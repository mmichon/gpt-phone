"""Spike: ElevenLabs streaming time-to-first-audio from the Pi, per model and voice.

    python spikes/tts_latency.py
"""

import asyncio
import statistics
import sys
import time

sys.path.insert(0, ".")
from phone.config import Config
from phone.tts import ElevenLabsTTS

VOICES = {
    "mike (clone)": ("hxlXQOvYq9uG0OG8ebdh", None, "Well, well, well. Look who finally called me back."),
    "elf (library)": ("542jzeOaLKbcpZhWfJDa", None, "Hi there! Do you want to play a guessing game?"),
    "fred (clone, pl)": ("K3q7w8KTKFSCTujD5PFK", "pl", "Cześć, jak się masz? Co u ciebie słychać?"),
}
MODELS = ["eleven_flash_v2_5", "eleven_turbo_v2_5"]


async def once(tts, voice, language, text):
    t0 = time.monotonic()
    stream = await tts.open(voice, language=language)
    connected = time.monotonic()
    await stream.send(text)
    await stream.end()
    first = None
    async for _ in stream.audio():
        first = first or time.monotonic()
    await stream.close()
    return (connected - t0) * 1000, (first - connected) * 1000


async def main():
    cfg = Config.from_env()
    for model in MODELS:
        tts = ElevenLabsTTS(cfg.elevenlabs_api_key, model, cfg.out_rate)
        for name, (voice, language, text) in VOICES.items():
            runs = [await once(tts, voice, language, text) for _ in range(3)]
            print(f"{model:20} {name:18} connect {statistics.median(r[0] for r in runs):5.0f} ms   "
                  f"text→first audio {statistics.median(r[1] for r in runs):5.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
