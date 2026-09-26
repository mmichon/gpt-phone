"""The phone's state machine: idle → operator → dialing → call → idle.

Lifting the handset starts a session task; hanging up cancels it wherever it
is, which stops all audio at once. A failure anywhere in a session plays the
busy message instead of crashing, so the phone is never silent.
"""

import asyncio
import logging
from dataclasses import dataclass

from . import audio
from .call import Call
from .hardware import DialStart, Digit, OffHook, OnHook

log = logging.getLogger(__name__)

PROMPT_REPEATS = 3
DIAL_WAIT_S = 15.0
DIGIT_WAIT_S = 5.0     # after the dial starts moving, the digit must arrive within this


@dataclass
class Deps:
    mic: object
    player: object
    tts: object
    cache: object
    sounds: object


class Sounds:
    """Decoded prompt sounds (ringbacks from sounds/) and synthesized tones."""

    def __init__(self, sounds_dir, rate):
        self.dir = sounds_dir
        self.rate = rate
        self._decoded = {}
        self.reorder = audio.reorder_tone(rate)
        self.dial_tone = audio.tone(rate, (350, 440), DIAL_WAIT_S)

    def get(self, name):
        if name not in self._decoded:
            try:
                self._decoded[name] = audio.decode(self.dir / name, self.rate)
            except Exception as e:
                log.warning("Can't load sound %s: %s", name, e)
                self._decoded[name] = audio.silence(self.rate, 0.5)
        return self._decoded[name]


class Switchboard:
    def __init__(self, cfg, directory, hardware, deps, status=None, observe=None, call_factory=Call):
        self.cfg = cfg
        self.directory = directory
        self.hw = hardware
        self.deps = deps
        self.player = deps.player
        self.cache = deps.cache
        self.status = status or (lambda text: None)
        self.observe = observe or (lambda kind, **data: None)
        self.call_factory = call_factory
        self._session = None
        self._dial = asyncio.Queue()

    def start(self):
        self.hw.start(asyncio.get_running_loop(), self.handle)
        self._set_status("idle")

    def handle(self, event):
        """Hardware events arrive here, on the event loop."""
        self.observe(type(event).__name__.lower(), **vars(event))
        match event:
            case OffHook():
                if self._session and not self._session.done() and not self._session.cancelling():
                    return  # already in a session
                log.info("Handset lifted")
                previous = self._session
                self._dial = asyncio.Queue()
                self._session = asyncio.create_task(self._run_session(previous))
            case OnHook():
                log.info("Handset hung up")
                if self._session:
                    self._session.cancel()
                self.player.flush()
                self._set_status("idle")
            case DialStart() | Digit():
                self._dial.put_nowait(event)

    async def wait_idle(self):
        if self._session:
            await asyncio.gather(self._session, return_exceptions=True)

    async def _run_session(self, previous):
        if previous and not previous.done():
            await asyncio.wait([previous], timeout=2)  # let the last call finish hanging up
        try:
            await self._session_body()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("Call failed: %s", e)
            self.observe("call_failed", error=repr(e))
            await self._busy()

    async def _session_body(self):
        self._set_status("operator")
        digit = await self._operator()
        if digit is None:
            log.info("Nothing dialed")
            return await self._reorder()
        role = self.directory.roles.get(digit)
        if role is None:
            log.info("Dialed %d: no such number", digit)
            op = self.directory.operator
            await self.player.play(await self.cache.get(op.voice_id, op.tts_model, op.wrong_number))
            return await self._reorder()
        log.info("Dialed %d: %s", digit, role.name)
        self._set_status(f"in call: {role.name}")
        await self.call_factory(self.cfg, role, self.deps, observe=self.observe).run()
        self._set_status("call ended")
        await self._reorder()

    async def _operator(self):
        op = self.directory.operator
        prompt = op.greeting
        for _ in range(PROMPT_REPEATS):
            pcm = self.cache.cached(op.voice_id, op.tts_model, prompt)
            if pcm is None:
                try:
                    pcm = await self.cache.get(op.voice_id, op.tts_model, prompt)
                except Exception as e:
                    log.warning("Operator prompt unavailable (%s); playing dial tone", e)
                    pcm = self.deps.sounds.dial_tone
            digit = await self._prompt_for_digit(pcm)
            if digit == 0:
                prompt = self.directory.listing()
            elif digit is not None:
                return digit
        return None

    async def _prompt_for_digit(self, pcm):
        """Play a prompt (dialing cuts it off) and return the digit dialed, or None."""
        while not self._dial.empty():
            self._dial.get_nowait()
        prompt = asyncio.create_task(self.player.play(pcm))
        try:
            event = await self._dial_event_while(prompt)
            if event is None:
                event = await asyncio.wait_for(self._dial.get(), DIAL_WAIT_S)
            prompt.cancel()
            while not isinstance(event, Digit):
                event = await asyncio.wait_for(self._dial.get(), DIGIT_WAIT_S)
            return event.digit
        except TimeoutError:
            return None
        finally:
            prompt.cancel()

    async def _dial_event_while(self, task):
        get = asyncio.create_task(self._dial.get())
        done, _ = await asyncio.wait({get, task}, return_when=asyncio.FIRST_COMPLETED)
        if get in done:
            return get.result()
        get.cancel()
        return None

    async def _busy(self):
        op = self.directory.operator
        pcm = self.cache.cached(op.voice_id, op.tts_model, op.busy)
        if pcm:
            await self.player.play(pcm)
        await self._reorder()

    async def _reorder(self):
        """Fast busy until the caller hangs up (a minute of it, then silence)."""
        self._set_status("off hook: reorder")
        for _ in range(6):
            await self.player.play(self.deps.sounds.reorder)

    def _set_status(self, text):
        self.status(text)
        self.observe("status", text=text)
