"""One phone call with a character. Everything runs at once:

  mic ──► brain                      (uplink: continuous, never paused)
  brain ──► chunker ──► speaker      (downlink: speaks while the reply is still being written)
  speaker ──► player                 (audio pump: plays while the rest is still being generated)
"""

import asyncio
import logging
import time

from . import audio
from .brain import GeminiBrain, Heard, Interrupted, Reply, ReplyDone, Retracted, SpeechEnded, SpeechStarted
from .tts import PhraseChunker, speakable

log = logging.getLogger(__name__)


class Speaker:
    """Speaks one reply at a time through a streaming TTS connection. cancel() stops it dead."""

    def __init__(self, tts, player, voice_id, model=None, language=None, on_first_audio=None):
        self.tts = tts
        self.player = player
        self.voice = (voice_id, model, language)
        self.on_first_audio = on_first_audio
        self._stream = None
        self._pending = None
        self._pumps = set()
        self.hold_until = None  # the current reply mustn't play before this (see brain.Reply)

    @property
    def busy(self):
        return self._stream is not None or bool(self._pumps) or self.player.playing

    def prewarm(self):
        """Open the next TTS connection early (while the caller is still talking)."""
        if self._stream is None and self._pending is None:
            self._pending = asyncio.create_task(self.tts.open(*self.voice))

    async def say(self, phrase):
        phrase = speakable(phrase)
        if not phrase:
            return
        if self._stream is None:
            self._stream = await self._take_stream()
            pump = asyncio.create_task(self._pump(self._stream, self.hold_until))
            self._pumps.add(pump)
            pump.add_done_callback(self._pumps.discard)
        await self._stream.send(phrase)

    async def end_reply(self):
        stream, self._stream = self._stream, None
        if stream:
            await stream.end()

    async def cancel(self):
        stream, self._stream = self._stream, None
        pending, self._pending = self._pending, None
        for task in [*self._pumps, pending]:
            if task:
                task.cancel()
        self.player.flush()
        if stream:
            await stream.close()
        if pending and pending.done() and not pending.cancelled() and not pending.exception():
            await pending.result().close()

    async def _take_stream(self):
        pending, self._pending = self._pending, None
        if pending:
            try:
                stream = await pending
                if stream.open:
                    return stream
            except Exception as e:
                log.debug("Pre-opened TTS stream failed: %s", e)
        return await self.tts.open(*self.voice)

    async def _pump(self, stream, hold_until=None):
        first = True
        try:
            async for pcm in stream.audio():
                if first:
                    if self.on_first_audio:
                        self.on_first_audio(time.monotonic())
                    if hold_until and (wait := hold_until - time.monotonic()) > 0:
                        await asyncio.sleep(wait)  # audio keeps arriving meanwhile; none is lost
                first = False
                self.player.write(pcm)
        finally:
            await stream.close()


class Call:
    def __init__(self, cfg, role, deps, observe=None, brain_factory=GeminiBrain):
        self.cfg = cfg
        self.role = role
        self.mic = deps.mic
        self.player = deps.player
        self.cache = deps.cache
        self.sounds = deps.sounds
        self.observe = observe or (lambda kind, **data: None)
        self.brain = brain_factory(cfg, role)
        self.speaker = Speaker(deps.tts, deps.player, role.voice_id, role.tts_model, role.language,
                               on_first_audio=self._on_first_audio)
        self._turn = 0
        self._marks = {}
        self._replying = False
        self._speech_end = None
        self._heard_at = None
        self._heard_timer = None
        self._last_activity = time.monotonic()
        self._caller_talking = False
        self._heard_count = 0
        self._mute_until = 0.0

    async def run(self):
        """Returns when the caller goes quiet for good; raises on failure. Hang-up cancels it."""
        log.info("Calling %s", self.role.name)
        self.observe("call_start", role=self.role.digit)
        connect = asyncio.create_task(self.brain.connect())
        self.player.on_start = self._on_play_start
        try:
            # Ring and greet from local audio while Gemini connects in parallel.
            await self.player.play(self.sounds.get(self.role.ringback))
            greeting = await self.cache.get(self.role.voice_id, self.role.tts_model,
                                            self.role.greeting, self.role.language)
            self.player.write(greeting)
            await connect
            self.mic.clear()
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._uplink())
                tasks.create_task(self._downlink())
                watch = tasks.create_task(self._watch_silence())
                await watch
                raise _CallOver
        except* _CallOver:
            pass
        finally:
            self.player.on_start = None
            connect.cancel()
            await self.speaker.cancel()
            await self.brain.close()
            self.observe("call_end", role=self.role.digit)

    async def _uplink(self):
        muted = audio.silence(self.cfg.mic_rate, self.cfg.mic_block_ms / 1000)
        async for frame in self.mic.frames():
            if not self.cfg.barge_in:
                # Half duplex: the character can't hear itself (or the caller) while talking.
                if self.player.playing:
                    self._mute_until = time.monotonic() + 0.3
                if time.monotonic() < self._mute_until:
                    frame = muted
            await self.brain.send_audio(frame)

    async def _downlink(self):
        chunker = PhraseChunker()
        async for event in self.brain.events():
            now = time.monotonic()
            match event:
                case SpeechStarted():
                    self._caller_talking = True
                    self.observe("speech_started")
                    if self.cfg.barge_in and self.speaker.busy:
                        await self._interrupt(chunker)
                    self.speaker.prewarm()
                case SpeechEnded(t=t):
                    # Only speech that turns into words counts as the caller being there
                    # (see Heard): a noise that transcribes to nothing mustn't hold off the nudge.
                    self._caller_talking = False
                    self._speech_end = t
                    self.observe("speech_ended")
                case Heard(text=text):
                    self._last_activity = self._heard_at = now
                    self._heard_count += 1
                    log.info("Caller: %s", text.strip())
                    self.observe("heard", text=text)
                case Reply(text=text, hold_until=hold_until):
                    if not self._replying:
                        self._start_reply(now)
                        self.speaker.hold_until = hold_until
                    self.observe("reply", text=text)
                    for phrase in chunker.feed(text):
                        await self.speaker.say(phrase)
                case ReplyDone():
                    for phrase in chunker.finish():
                        await self.speaker.say(phrase)
                    await self.speaker.end_reply()
                    self._replying = False
                    self.observe("reply_done")
                case Interrupted():
                    self._replying = False
                    self.observe("interrupted")
                    await self._interrupt(chunker)
                case Retracted():
                    # The caller carried on before hearing it; the brain will answer it all together.
                    self._replying = False
                    self.observe("retracted")
                    await self._interrupt(chunker)

    async def _interrupt(self, chunker):
        chunker.finish()
        if self._heard_timer:
            self._heard_timer.cancel()
            self._heard_timer = None
        await self.speaker.cancel()
        self.observe("output_stopped")

    def _start_reply(self, now):
        # Latency is measured from when the caller stopped talking: the server's
        # end-of-speech signal if it sent one, else the last transcribed words.
        self._replying = True
        self._turn += 1
        self._marks = {"eos": self._speech_end or self._heard_at or now, "text": now}
        self._speech_end = self._heard_at = None

    def _on_first_audio(self, t):
        self._marks.setdefault("tts", t)

    def _on_play_start(self, t):
        self.observe("play_start")
        if "text" in self._marks and "play" not in self._marks:
            # Once enough of the reply has played, the caller has heard it: talking now interrupts it.
            self._heard_timer = asyncio.get_running_loop().call_later(self.cfg.heard_after_s, self.brain.mark_heard)
        marks = self._marks
        if "text" in marks and "play" not in marks:
            marks["play"] = t
            ms = {k: round((marks[k] - marks["eos"]) * 1000) for k in ("text", "tts", "play") if k in marks}
            log.info("Turn %d latency from the caller's last word: text %s ms, TTS audio %s ms, playing %s ms",
                     self._turn, ms.get("text"), ms.get("tts"), ms.get("play"))
            self.observe("latency", turn=self._turn, **ms)

    async def _watch_silence(self):
        prompted = nudged = False
        heard = 0
        while True:
            await asyncio.sleep(0.25)
            if self.speaker.busy or self._caller_talking:
                self._last_activity = time.monotonic()
                continue
            if self.brain.ask_to_repeat():
                continue
            if heard != self._heard_count:
                heard, nudged = self._heard_count, False  # the caller spoke: they can be nudged again
            quiet = time.monotonic() - self._last_activity
            if quiet > self.cfg.nudge_s and not nudged:
                nudged = True  # once per silence: a nudge, then "still there?", then hang up
                if self.brain.nudge():
                    continue
            if quiet > self.cfg.give_up_s:
                log.info("Caller silent for %ds; ending call", quiet)
                return
            if quiet > self.cfg.still_there_s and not prompted:
                prompted = True
                pcm = await self.cache.get(self.role.voice_id, self.role.tts_model,
                                           self.role.still_there, self.role.language)
                self.player.write(pcm)
            elif quiet < self.cfg.still_there_s:
                prompted = False


class _CallOver(Exception):
    pass
