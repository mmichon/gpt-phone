"""End-to-end rig: the real switchboard, Gemini, ElevenLabs and audio devices,
with a scripted caller.

- ScriptedHardware lifts the handset, dials and hangs up on command.
- MixMic is the real (echo-cancelled) mic with the caller's recorded lines
  mixed in digitally, so real line noise and real earpiece echo are present
  while the caller's words stay deterministic.
- The player's tap records every frame played, with timestamps.
- Every switchboard and call event is timestamped for assertions and reports.
"""

import asyncio
import difflib
import io
import os
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np

from phone import audio
from phone.hardware import DialStart, Digit, OffHook, OnHook

FIXTURES = Path(__file__).parent.parent / "fixtures" / "caller"


def load_line(name, rate=16000, trim=False):
    """A caller line. With trim, the recording's own leading and trailing silence
    is cut (keeping 50 ms), so pauses placed between lines are exact."""
    with wave.open(str(FIXTURES / f"{name}.wav")) as w:
        assert w.getframerate() == rate and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = w.readframes(w.getnframes())
    if trim:
        samples = np.frombuffer(pcm, dtype=np.int16)
        loud = np.flatnonzero(np.abs(samples) > 500)
        margin = rate // 20
        pcm = samples[max(0, loud[0] - margin):loud[-1] + margin].tobytes()
    return pcm


class ScriptedHardware:
    def __init__(self):
        self.off_hook = False

    def start(self, loop, emit):
        self.emit = emit

    def resync(self):
        pass

    def close(self):
        pass

    def lift(self):
        self.off_hook = True
        self.emit(OffHook())

    def hang_up(self):
        self.off_hook = False
        self.emit(OnHook())

    async def dial(self, digit):
        self.emit(DialStart())
        await asyncio.sleep(0.1 * (digit or 10) + 0.2)  # a real dial takes about this long
        self.emit(Digit(digit))


class MixMic(audio.Mic):
    """The real mic, plus caller audio mixed in as if spoken into the handset."""

    def __init__(self, *args, caller_gain=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.caller_gain = caller_gain
        self._pending = bytearray()
        self._current = None
        self.spans = []  # (start, end, name) of each caller line, in monotonic time

    def say(self, pcm, name="line"):
        """Queue a caller line; it starts in the next mic block."""
        self._pending += pcm
        self._current = [None, None, name]
        self.spans.append(self._current)

    @property
    def talking(self):
        return bool(self._pending)

    def _put(self, pcm):
        if self._pending:
            now = time.monotonic()
            n = min(len(pcm), len(self._pending))
            if self._current[0] is None:
                self._current[0] = now
            mic = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
            caller = np.frombuffer(bytes(self._pending[:n]), dtype=np.int16).astype(np.int32)
            mic[:n // 2] += (caller * self.caller_gain).astype(np.int32)
            pcm = np.clip(mic, -32768, 32767).astype(np.int16).tobytes()
            del self._pending[:n]
            if not self._pending:
                self._current[1] = now  # this block (ending now) held the caller's last words
        super()._put(pcm)


class OutputTap:
    """Records what the speaker actually played, with the time each block played."""

    def __init__(self, rate):
        self.rate = rate
        self.blocks = []
        self._lock = threading.Lock()

    def __call__(self, t, pcm):
        if audio.rms(pcm) > 30:  # ignore digital near-silence
            with self._lock:
                self.blocks.append((t, pcm))

    def between(self, t0, t1=float("inf")):
        with self._lock:
            return b"".join(pcm for t, pcm in self.blocks if t0 <= t < t1)

    def times(self, t0=0.0, t1=float("inf")):
        with self._lock:
            return [t for t, _ in self.blocks if t0 <= t < t1]

    def seconds(self, t0=0.0, t1=float("inf")):
        return len(self.between(t0, t1)) / 2 / self.rate


class Observer:
    """Collects (time, kind, data) from the switchboard and call."""

    def __init__(self):
        self.events = []
        self._changed = asyncio.Event()

    def __call__(self, kind, **data):
        self.events.append((time.monotonic(), kind, data))
        self._changed.set()

    def of(self, kind, after=0.0):
        return [(t, d) for t, k, d in self.events if k == kind and t >= after]

    async def wait_for(self, kind, after=0.0, timeout=30.0, where=None):
        deadline = time.monotonic() + timeout
        while True:
            for t, d in self.of(kind, after):
                if where is None or where(d):
                    return t, d
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no {kind!r} event within {timeout}s")
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except TimeoutError:
                pass

    def text(self, kind, after=0.0, before=float("inf")):
        pieces = [d.get("text", "") for t, d in self.of(kind, after) if t < before]
        # Reply pieces carry their own spacing; transcripts of separate stretches don't.
        return " ".join(p.strip() for p in pieces) if kind == "heard" else "".join(pieces)

    def reply_that_stood(self, after=0.0):
        """The reply text since `after`, minus drafts that were withdrawn or cut short."""
        cutoffs = [t for kind in ("retracted", "interrupted") for t, _ in self.of(kind, after)]
        return self.text("reply", after=max(cutoffs, default=after))


def words(text):
    return re.findall(r"[\w']+", text.lower())


def similarity(a, b):
    return difflib.SequenceMatcher(None, words(a), words(b)).ratio()


POLISH_WORDS = {"jest", "nie", "się", "tak", "jak", "co", "to", "mam", "masz", "dobrze", "cześć", "dzień"}


def looks_polish(text):
    return bool(re.search(r"[ąćęłńóśźż]", text.lower())) or len(set(words(text)) & POLISH_WORDS) >= 2


async def transcribe(pcm, rate, api_key):
    """What did the phone actually say? (Catches silent or garbled TTS.)"""
    from google import genai
    from google.genai import types

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=os.environ.get("PHONE_TRANSCRIBE_MODEL", "gemini-3.8-flash"),
        contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"),
                  "Transcribe this speech exactly, in its original language. Output only the words."])
    return (response.text or "").strip()
