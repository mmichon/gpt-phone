"""The listening-and-thinking half of a call.

Listening: a local Silero voice activity detector (phone/vad.py) decides when
the caller starts and stops talking, and exactly that audio (plus a little
pre-roll) goes to a Gemini Transcribe Live session, which returns interim
transcripts as they speak and a punctuated final one for each stretch of
speech. Gemini's own voice detection is off: in testing it ended turns early
and dropped callers' last words. A new stretch isn't opened with Gemini until
the previous one's transcript is back (its audio waits in a queue): stretches
opened back to back sometimes lost a whole sentence.

Deciding the caller is done: when they stop after a finished sentence, the
reply starts at once, from the interim transcript if it's complete (the final
one, ~0.3 s later, replaces it if the words differ). A pause mid-sentence gets
a grace period, longer after "and" or a comma. A reply to a question plays as
soon as it's ready; a reply to a statement is held until the caller has been
quiet a little longer (Reply.hold_until), because storytellers pause after
full stops too.

Yielding: if the caller carries on before they've heard much of the reply
(mark_heard() says when they have), the reply is withdrawn and everything they
said is answered together, like two people who start talking at once. If
they've heard it, it's a real interruption: the reply stops, cut short.

Not hearing anything: if a stretch's final transcript never comes (short words
sometimes come back empty), its interim transcript stands in for it; with no
words at all, the character says it couldn't make that out. The same goes for
speech so chopped up (by the echo canceller, talking over the character) that it
only shows up as scattered near-misses and never starts a stretch. And if the caller
says nothing for a while after a reply, the character speaks up again (nudge()).
Both go to the text model as a [bracketed note] instead of the caller's words.

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
FINAL_WAIT_S = 0.8  # longest a new stretch of speech waits for the previous transcript
TRANSCRIPT_TIMEOUT_S = 1.5  # after this, a stretch's final transcript isn't coming
MISSED_WAIT_S = 1.0  # quiet after chopped-up speech before asking the caller to say it again
UNCLEAR = "[The caller said something short, but the line crackled and you couldn't make it out.]"
SILENCE = "[A few seconds of silence on the line; the caller hasn't answered.]"
SENTENCE_DONE = re.compile(r"[.?!…][\"')\]]*\s*$")
CONTINUING = re.compile(r"([,;:—–-]|\b(and|but|so|or|because|then|that|like|um+|uh+|if|when|with))\s*$", re.I)


@dataclass
class SpeechStarted:
    t: float = field(default_factory=time.monotonic)


@dataclass
class SpeechEnded:
    t: float = field(default_factory=time.monotonic)  # when the caller's last word ended


@dataclass
class Heard:
    text: str


@dataclass
class Reply:
    text: str
    hold_until: float = None   # don't play this reply before this monotonic time


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
    hold_until: float = None     # see Reply.hold_until
    note: bool = False           # answering a [note] about the call, not the caller's words

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
        self._speech_ended_at = 0.0
        self._background = set()
        self._outbox = asyncio.Queue()   # messages for Gemini, sent in order by _sender
        self._final_due = asyncio.Event()  # clear while a stretch's final transcript is awaited
        self._final_due.set()
        self._final_timer = None
        self._overdue_timer = None
        self._awaiting_final = False  # a stretch has ended and its final transcript hasn't come
        self._drop_late_final = False  # ...and we gave up on it, so if it turns up, ignore it
        self._blips = []                # recent near-misses: (time, seconds, peak)
        self._missed_at = None          # when near-misses last added up to probable speech
        self._sender = None

    # Listening

    def _listen_config(self):
        language = LANGUAGE_CODES.get(self.role.language, self.role.language)
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[language]),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)),
        )

    async def connect(self, attempts=3):
        """Open the listening session, retrying: a slow or failed connect is usually a blip."""
        self._spawn(self._warm_up())
        for attempt in range(1, attempts + 1):
            try:
                return await self._connect_once()
            except Exception as e:
                if attempt == attempts or "API key" in str(e):
                    raise BrainError(f"Couldn't connect to Gemini: {e}") from e
                log.warning("Connecting to Gemini failed (%s); retrying", e or type(e).__name__)
                await asyncio.sleep(0.5 * attempt)

    async def _connect_once(self):
        if self._sender is None:
            self._sender = asyncio.create_task(self._send_loop())
        started = time.monotonic()
        context = self.client.aio.live.connect(model=self.cfg.listen_model, config=self._listen_config())
        try:
            session = await asyncio.wait_for(context.__aenter__(), self.cfg.connect_timeout)
        except TimeoutError:
            raise TimeoutError(f"no connection within {self.cfg.connect_timeout}s") from None
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
        for task in [self._receiver, self._sender, exchange_task, *self._background]:
            if task:
                task.cancel()
        self._cancel_turn_timer()
        if self._overdue_timer:
            self._overdue_timer.cancel()
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
                log.debug("Speech started")
                self.speech_started()
                self._outbox.put_nowait({"activity_start": types.ActivityStart()})
            elif action == vad.AUDIO:
                self._outbox.put_nowait(
                    {"audio": types.Blob(data=data, mime_type=f"audio/pcm;rate={self.cfg.mic_rate}")})
            elif action == vad.END:
                seconds, peak, db = self.detector.last_stretch
                log.info("Speech %.2f s (peak p=%.2f, loudest %.0f dBFS)", seconds, peak, db)
                self._outbox.put_nowait({"activity_end": types.ActivityEnd()})
                self.speech_ended()
            elif action == vad.IGNORED:
                seconds, peak, db = self.detector.last_stretch
                log.info("Possible speech ignored (peak p=%.2f, %d ms, loudest %.0f dBFS)", peak, seconds * 1000, db)
                self.near_miss(seconds, peak)

    async def _send_loop(self):
        try:
            while True:
                message = await self._outbox.get()
                if "activity_start" in message:
                    await self._final_due.wait()  # the previous stretch's transcript first
                await self._send(**message)
                if "activity_end" in message:
                    self._final_due.clear()
                    loop = asyncio.get_running_loop()
                    self._final_timer = loop.call_later(FINAL_WAIT_S, self._final_due.set)
                    if self._overdue_timer:
                        self._overdue_timer.cancel()
                    self._overdue_timer = loop.call_later(TRANSCRIPT_TIMEOUT_S, self._transcript_overdue)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._events.put_nowait(e if isinstance(e, BrainError) else BrainError(str(e)))

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
        self._drop_late_final = False
        self._blips, self._missed_at = [], None
        self._cancel_turn_timer()
        self._emit(SpeechStarted())
        exchange = self._exchange
        if exchange and not exchange.heard:
            log.debug("Caller carried on before hearing the reply to %r; withdrawing it", exchange.asked)
            self._retract(exchange)
        elif exchange and not exchange.done:
            log.debug("Caller talked over the reply to %r", exchange.asked)
            self._interrupt(exchange)

    def speech_ended(self):
        log.debug("Speech ended; interim transcript so far: %r", self._interim)
        self._caller_talking = False
        self._speech_ended_at = time.monotonic()
        self._awaiting_final = True
        # Detection lags the last word by the silence it takes to be sure speech has ended.
        self._emit(SpeechEnded(t=self._speech_ended_at - self.cfg.vad_silence_ms / 1000))
        self._schedule_turn()

    def heard(self, text):
        """A final transcript of the last stretch of speech."""
        log.debug("Final transcript: %r", text)
        if self._drop_late_final:
            log.info("Transcript arrived too late; ignoring it: %r", text)
            return
        self._awaiting_final = False
        self._final_due.set()
        if self._final_timer:
            self._final_timer.cancel()
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

    def _transcript_overdue(self):
        """A stretch of speech ended a while ago and its final transcript hasn't come."""
        self._overdue_timer = None
        if not self._awaiting_final:
            return
        self._awaiting_final = False
        self._final_due.set()
        if self._caller_talking:
            return  # they carried on; the next transcript will get things moving
        seconds = self.detector.last_stretch[0]
        interim = self._interim
        if interim:
            log.info("No transcript for a %.2f s stretch; using the interim one: %r", seconds, interim)
            self.heard(interim)
            self._drop_late_final = True
            return
        log.info("No transcript for a %.2f s stretch", seconds)
        self._drop_late_final = True
        self._segment_open = False
        if self._pending:
            self._schedule_turn()
        elif not self._exchange or self._exchange.done:
            self._answer_note(UNCLEAR)

    def near_miss(self, seconds, peak):
        """Something that nearly counted as speech. Several close together probably were."""
        now = time.monotonic()
        self._blips = [b for b in self._blips if now - b[0] < 2.0] + [(now, seconds, peak)]
        if sum(b[1] for b in self._blips) >= 0.09 and max(b[2] for b in self._blips) >= 0.5:
            self._missed_at = now

    def ask_to_repeat(self):
        """If the caller seems to have said something that never came through, and has
        since gone quiet, have the character ask them to say it again. Returns whether it did."""
        if not self._missed_at or time.monotonic() - self._missed_at < MISSED_WAIT_S or not self._idle():
            return False
        self._blips, self._missed_at = [], None
        log.info("Caller's words seem to have been lost; asking them to repeat")
        self._answer_note(UNCLEAR)
        return True

    def _idle(self):
        exchange = self._exchange  # None before the first exchange: the greeting was the last word
        return not (self._caller_talking or self._segment_open or self._pending or self._turn_timer
                    or (exchange and not exchange.done))

    def nudge(self):
        """The caller has gone quiet after a reply (or the greeting): have the character speak up again.
        Returns whether it did (not while anything else is going on)."""
        if not self._idle():
            return False
        log.info("Caller quiet after the reply; nudging them")
        self._answer_note(SILENCE)
        return True

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
        if not exchange.asked.rstrip("\"') ").endswith("?"):
            exchange.hold_until = self._speech_ended_at + self.cfg.statement_hold_s
        log.debug("Answering from the %s transcript: %s", "interim" if speculative else "final", exchange.asked)
        self._begin(exchange)

    def _answer_note(self, note):
        self._begin(_Exchange(base=note, segment="", segment_final=True, note=True))

    def _begin(self, exchange):
        exchange.user_index = len(self.history)
        self.history.append(_turn("user", exchange.asked))
        self._exchange = exchange
        exchange.task = asyncio.create_task(self._write_reply(exchange))

    async def _write_reply(self, exchange):
        try:
            async for piece in self.stream_reply(self.history[:exchange.user_index + 1]):
                exchange.said += piece
                self._emit(Reply(piece, exchange.hold_until))
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
        if exchange.note:
            self._pending = ""
        else:
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
