"""The listening-and-thinking half of a call: one Gemini Live session.

Mic audio streams in continuously. Gemini's server-side voice activity
detection decides when the caller has finished a turn, and the reply streams
back as text for the TTS voice to speak. Events:

  SpeechStarted / SpeechEnded   the caller started / stopped talking
  Heard(text)                   transcription of the caller (for logs and tests)
  Reply(text)                   a piece of the character's reply
  ReplyDone                     the reply is complete
  Interrupted                   the caller talked over the reply; it was abandoned
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

log = logging.getLogger(__name__)


@dataclass
class SpeechStarted:
    t: float = field(default_factory=time.monotonic)


@dataclass
class SpeechEnded:
    t: float = field(default_factory=time.monotonic)


@dataclass
class Heard:
    text: str


@dataclass
class Reply:
    text: str


@dataclass
class ReplyDone:
    pass


@dataclass
class Interrupted:
    pass


class BrainError(Exception):
    pass


class GeminiLiveBrain:
    def __init__(self, cfg, role):
        self.cfg = cfg
        self.role = role
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.session = None
        self._context = None
        self._resume_handle = None

    def _config(self):
        sensitivity = types.AutomaticActivityDetection(
            # Low start sensitivity ignores line noise and breaths; low end
            # sensitivity waits out mid-sentence pauses instead of cutting in.
            start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
            prefix_padding_ms=self.cfg.vad_prefix_ms,
            silence_duration_ms=self.cfg.vad_silence_ms,
        )
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction=self.role.system_instruction(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=sensitivity,
                activity_handling=(types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS if self.cfg.barge_in
                                   else types.ActivityHandling.NO_INTERRUPTION),
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()),
            session_resumption=types.SessionResumptionConfig(handle=self._resume_handle),
        )

    async def connect(self):
        started = time.monotonic()
        context = self.client.aio.live.connect(model=self.cfg.live_model, config=self._config())
        try:
            session = await asyncio.wait_for(context.__aenter__(), self.cfg.connect_timeout)
        except TimeoutError:
            raise BrainError(f"Gemini Live didn't connect within {self.cfg.connect_timeout}s") from None
        old_context, self._context, self.session = self._context, context, session
        log.info("Gemini Live connected in %d ms%s", (time.monotonic() - started) * 1000,
                 " (resumed)" if self._resume_handle else "")
        if old_context:
            asyncio.create_task(self._close_context(old_context))

    async def close(self):
        context, self._context, self.session = self._context, None, None
        if context:
            await self._close_context(context)

    async def _close_context(self, context):
        try:
            await context.__aexit__(None, None, None)
        except Exception as e:
            log.debug("Error closing Live session: %s", e)

    async def send_audio(self, pcm):
        session = self.session
        if session is None:
            return
        try:
            await session.send_realtime_input(
                audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={self.cfg.mic_rate}"))
        except Exception as e:
            if session is self.session:
                raise BrainError(f"Sending audio failed: {e}") from e
            # The session was swapped out mid-send during a reconnect; drop the frame.

    async def events(self):
        while True:
            session = self.session
            try:
                async for message in session.receive():
                    for event in self._translate(message):
                        yield event
                    if message.go_away:
                        log.info("Live session ending in %s; resuming", message.go_away.time_left)
                        await self._resume()
                        break
            except BrainError:
                raise
            except Exception as e:
                if session is not self.session:
                    continue  # already reconnected
                if self._resume_handle:
                    log.warning("Live connection dropped (%s); resuming", e)
                    await self._resume()
                    continue
                raise BrainError(f"Live connection failed: {e}") from e

    async def _resume(self):
        if not self._resume_handle:
            raise BrainError("Live session ended and can't be resumed")
        await self.connect()

    def _translate(self, message):
        if message.session_resumption_update and message.session_resumption_update.resumable:
            self._resume_handle = message.session_resumption_update.new_handle or self._resume_handle
        activity = message.voice_activity
        if activity and activity.voice_activity_type:
            if activity.voice_activity_type == types.VoiceActivityType.ACTIVITY_START:
                yield SpeechStarted()
            elif activity.voice_activity_type == types.VoiceActivityType.ACTIVITY_END:
                yield SpeechEnded()
        content = message.server_content
        if not content:
            return
        if content.input_transcription and content.input_transcription.text:
            yield Heard(content.input_transcription.text)
        if content.interrupted:
            yield Interrupted()
        if content.model_turn:
            for part in content.model_turn.parts or []:
                if part.text and not part.thought:
                    yield Reply(part.text)
        if content.turn_complete and not content.interrupted:
            yield ReplyDone()
