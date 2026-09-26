"""Spike: which VAD settings hear every caller line, whole, without cutting in?

Runs the real GeminiBrain over a fixed script of caller lines, with the gaps
between lines filled by digital zeros or by recorded-level line noise.

    PHONE_VAD_SILENCE_MS=500 PHONE_TURN_GRACE_S=0.8 python spikes/vad_matrix.py zeros|noise
"""

import asyncio
import logging
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from phone.brain import GeminiBrain, Heard, Interrupted, Reply, ReplyDone, Retracted, SpeechEnded, SpeechStarted
from phone.config import Config
from phone.roles import Role
from tests.e2e.harness import load_line

ROLE = Role(digit=1, name="test", voice_id="-", greeting="Hello, who's this?",
            persona="You are a friendly old telephone operator.")
SCRIPT = [["q1"], ["q2"], ["q3"], ["long_1", 1.2, "long_2", 1.2, "long_3"], ["q4"]]
RATE = 16000


def gap(seconds, noise):
    n = int(seconds * RATE)
    if not noise:
        return b"\0\0" * n
    rng = np.random.default_rng()
    return (rng.standard_normal(n) * 9).astype(np.int16).tobytes()  # the handset's hiss after noise suppression


async def main(noise):
    logging.basicConfig(level=logging.WARNING, format="%(relativeCreated)8.0f %(message)s")
    logging.getLogger("phone.brain").setLevel(logging.DEBUG)
    cfg = Config.from_env()
    brain = GeminiBrain(cfg, ROLE)
    await brain.connect()
    t0 = time.monotonic()
    log = []

    async def listen():
        replying = False
        async for e in brain.events():
            log.append((time.monotonic() - t0, e))
            # Pretend the reply is spoken: TTS starts ~0.3 s after the text, heard 0.5 s later.
            if isinstance(e, Reply) and not replying:
                replying = True
                asyncio.get_running_loop().call_later(0.8, brain.mark_heard)
            elif isinstance(e, (ReplyDone, Interrupted, Retracted)):
                replying = False

    listener = asyncio.create_task(listen())
    block = RATE * cfg.mic_block_ms // 1000 * 2
    spans = []
    for parts in SCRIPT:
        pcm = b"".join(gap(p, noise) if isinstance(p, float) else load_line(p) for p in parts)
        if noise:  # the handset hiss is under the speech too
            speech = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
            speech += np.frombuffer(gap(len(speech) / RATE, True), dtype=np.int16)[:len(speech)]
            pcm = np.clip(speech, -32768, 32767).astype(np.int16).tobytes()
        pcm += gap(7.0, noise)
        start = time.monotonic() - t0
        speech_bytes = len(pcm) - int(7.0 * RATE) * 2
        end, sent = None, 0.0
        for i in range(0, len(pcm), block):
            await brain.send_audio(pcm[i:i + block])
            if end is None and i + block >= speech_bytes:
                end = time.monotonic() - t0
            sent += block / 2 / RATE
            await asyncio.sleep(max(0.0, start + sent - (time.monotonic() - t0)))  # real time, no drift
        spans.append(("+".join(p for p in parts if isinstance(p, str)), start, end))
    listener.cancel()
    await brain.close()

    print(f"VAD local threshold={cfg.vad_threshold} silence={cfg.vad_silence_ms}ms grace={cfg.turn_grace_s}s "
          f"text={cfg.text_model} gaps={'noise' if noise else 'zeros'}")
    for name, start, end in spans:
        window = [(t, e) for t, e in log if start <= t < end + 7.0]
        heard = " ".join(e.text.strip() for t, e in window if isinstance(e, Heard))
        replies = [t for t, e in window if isinstance(e, Reply)]
        early = [t for t in replies if t < end]
        first_after = min((t for t in replies if t >= end), default=None)
        starts = sum(isinstance(e, SpeechStarted) for _, e in window)
        withdrawn = sum(isinstance(e, (Retracted, Interrupted)) for _, e in window)
        vad_end = max((t for t, e in window if isinstance(e, SpeechEnded) and t <= (first_after or 1e9)), default=None)
        final = max((t for t, e in window if isinstance(e, Heard) and t <= (first_after or 1e9)), default=None)
        ms = lambda a, b: f"{(b - a) * 1000:5.0f}" if a is not None and b is not None else "    -"
        print(f"  {name:22} starts={starts} withdrawn={withdrawn} early_reply={'YES' if early else 'no':3} "
              f"audio end→vad end {ms(end, vad_end)} →final {ms(vad_end, final)} →text {ms(final, first_after)} "
              f"= {ms(end, first_after)} ms  heard={heard!r}")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] == ["noise"]))
