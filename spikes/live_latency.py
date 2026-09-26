"""Spike: does gemini-3.8-live work in text mode, and how fast does it answer?

Streams caller WAVs into a real GeminiLiveBrain at real-time pace and prints a
timeline: when the caller's audio ended, when the server detected end of
speech, and when the first reply text arrived.

    python spikes/live_latency.py [line ...]    (default: hello q1 q2 long)
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")
from phone.brain import GeminiLiveBrain, Heard, Interrupted, Reply, ReplyDone, SpeechEnded, SpeechStarted
from phone.config import Config
from phone.roles import Role
from tests.e2e.harness import load_line

ROLE = Role(digit=1, name="test", voice_id="-", greeting="Hello, who's this?",
            persona="You are a friendly old telephone operator.")
LINES = {"long": ["long_1", 1.2, "long_2", 1.2, "long_3"]}


async def main(names):
    cfg = Config.from_env()
    brain = GeminiLiveBrain(cfg, ROLE)
    t0 = time.monotonic()
    await brain.connect()
    print(f"connected in {(time.monotonic() - t0) * 1000:.0f} ms (model {cfg.live_model})")
    stamp = lambda: f"{(time.monotonic() - t0):7.2f}s"
    ended = {}

    async def listen():
        first = None
        async for event in brain.events():
            match event:
                case SpeechStarted() | SpeechEnded() | Interrupted():
                    print(stamp(), type(event).__name__)
                case Heard(text=text):
                    print(stamp(), "heard:", text)
                case Reply(text=text):
                    if first is None:
                        first = time.monotonic()
                        lag = (first - ended["t"]) * 1000 if "t" in ended else float("nan")
                        print(stamp(), f"FIRST TEXT {lag:.0f} ms after the caller's audio ended")
                    print(stamp(), "reply:", repr(text))
                case ReplyDone():
                    print(stamp(), "reply done")
                    first = None

    listener = asyncio.create_task(listen())
    block = cfg.mic_rate * cfg.mic_block_ms // 1000 * 2
    for name in names:
        parts = LINES.get(name, [name])
        pcm = b"".join(b"\0\0" * int(p * cfg.mic_rate) if isinstance(p, float) else load_line(p) for p in parts)
        pcm += b"\0\0" * cfg.mic_rate * 6  # then silence while it answers
        print(stamp(), f"--- caller says {name} ({(len(pcm) / 2 / cfg.mic_rate) - 6:.1f}s)")
        for i in range(0, len(pcm), block):
            await brain.send_audio(pcm[i:i + block])
            if i + block >= len(pcm) - cfg.mic_rate * 6 * 2 and "t" not in ended:
                ended["t"] = time.monotonic()
            await asyncio.sleep(cfg.mic_block_ms / 1000)
        ended.clear()
    listener.cancel()
    await brain.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["hello", "q1", "q2", "long"]))
