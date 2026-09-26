"""Spike: with local VAD driving Gemini Transcribe Live's activity boundaries,
does every stretch of speech get a final transcript? Plays a caller line in
real time through the same SpeechDetector the brain uses.

    python spikes/segment_trace.py LINE [MIN_GAP_MS]
MIN_GAP_MS delays a new activity_start until that long after the previous
activity_end (buffering the audio), to test whether close boundaries lose text.
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")
from google import genai
from google.genai import types

from phone import vad
from tests.e2e.harness import load_line

RATE = 16000


async def main(line, min_gap_ms="0"):
    min_gap = int(min_gap_ms) / 1000
    pcm = b"\0\0" * RATE + load_line(line) + b"\0\0" * RATE * 3
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=["en-US"]),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)))
    detector = vad.SpeechDetector()
    out = []
    async with genai.Client().aio.live.connect(model="gemini-3.5-transcribe-live", config=config) as session:
        t0 = time.monotonic()
        stamp = lambda: f"{time.monotonic() - t0:5.2f}"

        async def receive():
            while True:
                async for m in session.receive():
                    sc = m.server_content
                    if sc and sc.input_transcription and sc.input_transcription.text:
                        out.append(f"{stamp()}   FINAL {sc.input_transcription.text!r}")

        receiver = asyncio.create_task(receive())
        last_end, held = -10.0, []
        sent = 0.0
        for i in range(0, len(pcm), 1024):
            for action, data in detector.feed(pcm[i:i + 1024]):
                now = time.monotonic()
                if action == vad.START:
                    wait = last_end + min_gap - now
                    if wait > 0:
                        out.append(f"{stamp()} (holding start {wait * 1000:.0f} ms)")
                        await asyncio.sleep(wait)
                    out.append(f"{stamp()} activity_start")
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                elif action == vad.AUDIO:
                    await session.send_realtime_input(audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000"))
                elif action == vad.END:
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                    last_end = time.monotonic()
                    out.append(f"{stamp()} activity_end")
            sent += 1024 / 2 / RATE
            await asyncio.sleep(max(0.0, sent - (time.monotonic() - t0)))
        await asyncio.sleep(2)
        receiver.cancel()
    print(f"=== {line} min_gap={min_gap_ms}ms")
    print("\n".join(out))


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
