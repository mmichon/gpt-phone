"""The listening-and-thinking half of a call.

Listening: a local Silero voice activity detector (phone/vad.py) decides when
the caller starts and stops talking, and exactly that audio (plus a little
pre-roll) goes to a Gemini Transcribe Live session, which returns interim
transcripts as they speak and a punctuated final one for each stretch of
speech. Gemini's own voice detection is off: in testing it ended turns early
and dropped callers' last words.

Deciding the caller is done: when they stop after a finished sentence, the
reply starts at once, from the interim transcript if it's complete (the final
one, ~0.3 s later, replaces it if the words differ). A pause mid-sentence gets
a grace period, longer after "and" or a comma.

Yielding: if the caller carries on before they've heard much of the reply
(mark_heard() says when they have), the reply is withdrawn and everything they
said is answered together, like two people who start talking at once. If
they've heard it, it's a real interruption: the reply stops, cut short.

Thinking: each turn goes to a fast Gemini text model, and the reply streams
back piece by piece for the character's voice to speak.

(Gemini Live's conversational models only answer in audio; speaking their
transcript made them miss or clip callers. See spikes/.)

Events:
  SpeechStarted / SpeechEnded   the caller started / stopped talking
  Heard(text)                   a final transcript of the caller
  Reply(text)                   a piece of the character's reply
  ReplyDone                     the reply is complete
  Retracted                     the caller carried on before hearing the reply; drop it quietly
  Interrupted                   the caller talked over the reply; it was cut short
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from . import vad

log = logging.getLogger(__name__)

LANGUAGE_CODES = {"en": "en-US", "pl": "pl-PL"}
SENTENCE_DONE = re.compile(r"[.?!…][\"')\]]*\s*$")
CONTINUING = re.compile(r"([,;:—–-]|\b(and|but|so|or|because|then|that|like|um+|uh+|if|when|with))\s*$", re.I)


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
class Retracted:
    pass


@dataclass
class Interrupted:
    pass


class BrainError(Exception):
    pass


def _turn(role, text):
    return types.Content(role=role, parts=[types.Part(text=text)])


def _words(text):
    return re.findall(r"[\w']+", text.lower())


@dataclass
class _Exchange:
    """One caller turn and the reply to it."""
    base: str                    # final transcripts from before the last stretch of speech
    segment: str                 # the last stretch: its interim transcript, then its final one
    segment_final: bool
    task: asyncio.Task = None
    said: str = ""               # reply text so far
    heard: bool = False          # has the caller heard enough of the reply for it to count?
    done: bool = False
    user_index: int = 0          # where the caller's turn sits in the history

    @property
    def asked(self):
        return f"{self.base} {self.segment}".strip()


class GeminiBrain:
    def __init__(self, cfg, role):
        self.cfg = cfg
        self.role = role
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.history = []            # completed and in-progress exchanges
        self.session = None
        self.detector = vad.SpeechDetector(cfg.mic_rate, cfg.vad_threshold, cfg.vad_start_ms,
                                           cfg.vad_silence_ms, cfg.vad_preroll_ms)
        self._context = None
        self._receiver = None
        self._events = asyncio.Queue()
        self._pending = ""           # final transcripts not yet answered
        self._interim = ""           # interim transcript of the current stretch of speech
        self._segment_open = False   # a stretch of speech whose final transcript hasn't arrived
        self._exchange = None
        self._turn_timer = None
        self._caller_talking = False
        self._background = set()

    # Listening

    def _listen_config(self):
        language = LANGUAGE_CODES.get(self.role.language, self.role.language)
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[language]),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)),
        )

    async def connect(self):
        self._spawn(self._warm_up())
        started = time.monotonic()
        context = self.client.aio.live.connect(model=self.cfg.listen_model, config=self._listen_config())
        try:
            session = await asyncio.wait_for(context.__aenter__(), self.cfg.connect_timeout)
        except TimeoutError:
            raise BrainError(f"Gemini didn't connect within {self.cfg.connect_timeout}s") from None
        old_context, old_receiver = self._context, self._receiver
        self._context, self.session = context, session
        self._receiver = asyncio.create_task(self._receive(session))
        log.info("Listening via %s (connected in %d ms)", self.cfg.listen_model, (time.monotonic() - started) * 1000)
        if old_receiver:
            old_receiver.cancel()
        if old_context:
            self._spawn(self._close_context(old_context))

    async def _warm_up(self):
        """Open the connection to the text model now, so the first reply doesn't pay for it."""
        try:
            await self.client.aio.models.get(model=self.cfg.text_model)
        except Exception as e:
            log.debug("Warm-up failed: %s", e)

    async def close(self):
        exchange_task = self._exchange.task if self._exchange else None
        for task in [self._receiver, exchange_task, *self._background]:
            if task:
                task.cancel()
        self._cancel_turn_timer()
        context, self._context, self.session = self._context, None, None
        if context:
            await self._close_context(context)

    async def _close_context(self, context):
        try:
            await context.__aexit__(None, None, None)
        except Exception as e:
            log.debug("Error closing the listening session: %s", e)

    async def send_audio(self, pcm):
        """Feed mic audio. Only speech (as judged locally) is sent to Gemini."""
        for action, data in self.detector.feed(pcm):
            if action == vad.START:
                self.speech_started()
                await self._send(activity_start=types.ActivityStart())
            elif action == vad.AUDIO:
                await self._send(audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={self.cfg.mic_rate}"))
            elif action == vad.END:
                await self._send(activity_end=types.ActivityEnd())
                self.speech_ended()

    async def _send(self, **message):
        session = self.session
        if session is None:
            return
        try:
            await session.send_realtime_input(**message)
        except Exception as e:
            if session is self.session:
                raise BrainError(f"Sending to Gemini failed: {e}") from e
            # The session was swapped for a fresh one mid-send; drop it.

    async def events(self):
        while True:
            event = await self._events.get()
            if isinstance(event, Exception):
                raise event
            yield event

    async def _receive(self, session):
        try:
            while True:
                async for message in session.receive():
                    self.handle(message)
                    if message.go_away:
                        log.info("Listening session ending in %s; starting a new one", message.go_away.time_left)
                        self._spawn(self._reconnect())
                        return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if session is self.session:
                log.warning("Listening session dropped (%s); reconnecting", e)
                self._spawn(self._reconnect())

    async def _reconnect(self):
        try:
            await self.connect()
        except Exception as e:
            self._events.put_nowait(BrainError(f"Lost the listening session: {e}"))

    def handle(self, message):
        """Process one message from the listening session."""
        content = message.server_content
        if not content:
            return
        if content.interim_input_transcription and content.interim_input_transcription.text:
            self._interim = content.interim_input_transcription.text.strip()
        if content.input_transcription and content.input_transcription.text:
            self.heard(content.input_transcription.text.strip())

    # The caller

    def speech_started(self):
        self._caller_talking = True
        self._segment_open = True
        self._interim = ""
        self._cancel_turn_timer()
        self._emit(SpeechStarted())
        exchange = self._exchange
        if exchange and not exchange.heard:
            self._retract(exchange)
        elif exchange and not exchange.done:
            self._interrupt(exchange)

    def speech_ended(self):
        log.debug("Speech ended; interim transcript so far: %r", self._interim)
        self._caller_talking = False
        self._emit(SpeechEnded())
        self._schedule_turn()

    def heard(self, text):
        """A final transcript of the last stretch of speech."""
        self._segment_open = False
        self._interim = ""
        self._emit(Heard(text))
        exchange = self._exchange
        if exchange and not exchange.segment_final:
            # The reply was started from the interim transcript; this is the real one.
            speculated, exchange.segment, exchange.segment_final = exchange.segment, text, True
            self.history[exchange.user_index] = _turn("user", exchange.asked)
            if _words(text) != _words(speculated) and not exchange.said:
                log.info("Final transcript differs from the interim one; restarting the reply")
                exchange.task.cancel()
                exchange.task = asyncio.create_task(self._write_reply(exchange))
            return
        self._pending = f"{self._pending} {text}".strip()
        self._schedule_turn()

    def mark_heard(self):
        """The caller has now heard enough of the current reply that it counts as said."""
        if self._exchange:
            self._exchange.heard = True

    # Deciding the caller is done

    def _candidate(self):
        """What we'd answer now, and whether it rests on an interim transcript."""
        if self._segment_open and self._interim and SENTENCE_DONE.search(self._interim):
            return f"{self._pending} {self._interim}".strip(), True
        if self._segment_open:
            return "", False  # wait for the final transcript
        return self._pending, False

    def _schedule_turn(self):
        if self._caller_talking or (self._exchange and not self._exchange.done):
            return
        text, speculative = self._candidate()
        if not text:
            return
        self._cancel_turn_timer()
        if SENTENCE_DONE.search(text):
            delay = 0
        elif CONTINUING.search(text):
            delay = self.cfg.turn_grace_continuing_s  # "...and", "...station,": more is coming
        else:
            delay = self.cfg.turn_grace_s     # paused mid-sentence: they may carry on
        self._turn_timer = asyncio.get_running_loop().call_later(delay, self._start_reply)

    def _cancel_turn_timer(self):
        if self._turn_timer:
            self._turn_timer.cancel()
            self._turn_timer = None

    # Thinking

    def _start_reply(self):
        self._turn_timer = None
        text, speculative = self._candidate()
        if not text:
            return
        if speculative:
            exchange = _Exchange(base=self._pending, segment=self._interim, segment_final=False)
        else:
            exchange = _Exchange(base=self._pending, segment="", segment_final=True)
        self._pending = ""
        log.debug("Answering from the %s transcript: %s", "interim" if speculative else "final", exchange.asked)
        exchange.user_index = len(self.history)
        self.history.append(_turn("user", exchange.asked))
        self._exchange = exchange
        exchange.task = asyncio.create_task(self._write_reply(exchange))

    async def _write_reply(self, exchange):
        try:
            async for piece in self.stream_reply(self.history[:exchange.user_index + 1]):
                exchange.said += piece
                self._emit(Reply(piece))
            exchange.done = True
            self.history.append(_turn("model", exchange.said))
            self._emit(ReplyDone())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._events.put_nowait(BrainError(f"Writing the reply failed: {e}"))

    async def stream_reply(self, history):
        thinking = types.ThinkingConfig(thinking_level=self.cfg.text_thinking) if self.cfg.text_thinking else None
        config = types.GenerateContentConfig(
            system_instruction=self.role.system_instruction(), thinking_config=thinking,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        stream = await asyncio.wait_for(
            self.client.aio.models.generate_content_stream(
                model=self.cfg.text_model, contents=list(history), config=config),
            self.cfg.first_token_timeout)
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _retract(self, exchange):
        """The caller carried on before hearing the reply: forget it, and answer everything together."""
        exchange.task.cancel()
        del self.history[exchange.user_index:]
        self._exchange = None
        self._pending = exchange.base if not exchange.segment_final else exchange.asked
        # (If its final transcript is still on the way, heard() will add it to _pending.)
        if exchange.said:
            self._emit(Retracted())

    def _interrupt(self, exchange):
        """The caller talked over a reply they'd heard: keep what was said, cut short."""
        exchange.task.cancel()
        self.history.append(_turn("model", exchange.said.rstrip() + " —"))
        self._exchange = None
        self._emit(Interrupted())

    def _emit(self, event):
        self._events.put_nowait(event)

    def _spawn(self, coro):
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
