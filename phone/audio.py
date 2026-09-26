"""Long-lived microphone and speaker streams, plus prompt sounds.

Both streams stay open for the life of the process, so a call never waits for
a device to open and never misses the caller's first syllable. Audio is raw
16-bit mono PCM throughout. If a device disappears, the stream goes unhealthy
and the process exits so systemd can restart it with a fresh PortAudio.
"""

import asyncio
import collections
import logging
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_BYTES = 2


class Player:
    """A speaker stream fed from a byte buffer. flush() silences it immediately."""

    def __init__(self, rate, device=None, tap=None):
        self.rate = rate
        self.tap = tap           # tap(t, pcm) is called from the audio thread with audio as it plays
        self.on_start = None     # on_start(t) is called on the loop when audio starts after silence
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._was_playing = False
        self._loop = None
        self._drained = asyncio.Event()
        self._drained.set()
        self._device = device
        self._stream = None

    def open(self):
        self._loop = asyncio.get_running_loop()
        self._stream = sd.RawOutputStream(
            samplerate=self.rate, channels=1, dtype="int16", device=self._device,
            blocksize=self.rate // 50, latency="low", callback=self._callback)
        self._stream.start()
        log.info("Speaker open: %s at %d Hz", self._stream.device, self.rate)

    def close(self):
        if self._stream:
            self._stream.close()

    @property
    def healthy(self):
        return self._stream is not None and self._stream.active

    @property
    def playing(self):
        return not self._drained.is_set()

    def write(self, pcm):
        if not pcm:
            return
        with self._lock:
            self._buffer += pcm
        self._drained.clear()

    def flush(self):
        with self._lock:
            self._buffer.clear()
        self._drained.set()

    async def wait_drained(self):
        await self._drained.wait()

    async def play(self, pcm):
        """Play a whole sound. Cancelling stops it at once."""
        try:
            self.write(pcm)
            await self.wait_drained()
        except asyncio.CancelledError:
            self.flush()
            raise

    def _callback(self, outdata, frames, time_info, status):
        want = frames * SAMPLE_BYTES
        with self._lock:
            chunk = bytes(self._buffer[:want])
            del self._buffer[:want]
            empty = not self._buffer
        outdata[:len(chunk)] = chunk
        outdata[len(chunk):] = b"\x00" * (want - len(chunk))
        now = time.monotonic()
        if chunk:
            if self.tap:
                self.tap(now, chunk)
            if not self._was_playing:
                self._was_playing = True
                if self.on_start:
                    self._loop.call_soon_threadsafe(self.on_start, now)
        if empty and self._was_playing:
            self._was_playing = False
            self._loop.call_soon_threadsafe(self._maybe_drained)

    def _maybe_drained(self):
        # A write may have landed between the audio thread emptying the buffer and now.
        with self._lock:
            empty = not self._buffer
        if empty:
            self._drained.set()


class Mic:
    """A microphone stream delivering fixed-size PCM blocks to one async consumer."""

    def __init__(self, rate, block_ms, device=None, max_backlog_s=2.0):
        self.rate = rate
        self.block = rate * block_ms // 1000
        self._device = device
        self._frames = collections.deque(maxlen=int(max_backlog_s * 1000 / block_ms))
        self._ready = asyncio.Event()
        self._stream = None

    def open(self):
        self._loop = asyncio.get_running_loop()
        self._stream = sd.RawInputStream(
            samplerate=self.rate, channels=1, dtype="int16", device=self._device,
            blocksize=self.block, latency="low", callback=self._callback)
        self._stream.start()
        log.info("Mic open: %s at %d Hz", self._stream.device, self.rate)

    def close(self):
        if self._stream:
            self._stream.close()

    @property
    def healthy(self):
        return self._stream is not None and self._stream.active

    def clear(self):
        self._frames.clear()

    async def frames(self):
        while True:
            while self._frames:
                yield self._frames.popleft()
            self._ready.clear()
            await self._ready.wait()

    def _callback(self, indata, frames, time_info, status):
        self._loop.call_soon_threadsafe(self._put, bytes(indata))

    def _put(self, pcm):
        self._frames.append(pcm)
        self._ready.set()


def decode(path, rate):
    """Decode any audio file ffmpeg understands into mono 16-bit PCM at rate."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"],
        check=True, capture_output=True, timeout=30).stdout


def tone(rate, freqs, seconds, on=None, off=None, volume=0.15):
    """A telephone call-progress tone: summed sines, optionally cadenced on/off."""
    t = np.arange(int(rate * seconds)) / rate
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    if on and off:
        wave *= (t % (on + off)) < on
    return (wave * volume * 32767).astype(np.int16).tobytes()


def reorder_tone(rate, seconds=10.0):
    """Fast busy: the network couldn't complete the call."""
    return tone(rate, (480, 620), seconds, on=0.25, off=0.25)


def silence(rate, seconds):
    return b"\x00" * (int(rate * seconds) * SAMPLE_BYTES)


def rms(pcm):
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
