"""Spike: trace what Gemini Transcribe Live sends (voice activity, interim and
final transcripts) for a caller script, at a given VAD setting and noise floor.

    python spikes/transcribe_trace.py START END SILENCE_MS NOISE_RMS
    e.g. python spikes/transcribe_trace.py high low 700 9
"""

import asyncio
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from google import genai
from google.genai import types

from tests.e2e.harness import load_line

RATE = 16000


async def main(start, end, silence_ms, noise_rms):
    parts = [("q2", 2.0), ("long_1", 1.2), ("long_2", 1.2), ("long_3", 2.0), ("q3", 4.0)]
    pcm, spans, t = [], [], 0.0
    for name, gap in parts:
        line = np.frombuffer(load_line(name), np.int16)
        spans.append(f"{name} {t:.1f}-{t + len(line) / RATE:.1f}s")
        pcm += [line, np.zeros(int(gap * RATE), np.int16)]
        t += len(line) / RATE + gap
    audio = np.concatenate(pcm).astype(np.int32)
    audio += (np.random.default_rng(1).standard_normal(len(audio)) * float(noise_rms)).astype(np.int32)
    audio = np.clip(audio, -32768, 32767).astype(np.int16).tobytes()
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=["en-US"]),
        realtime_input_config=types.RealtimeInputConfig(automatic_activity_detection=types.AutomaticActivityDetection(
            start_of_speech_sensitivity=types.StartSensitivity[f"START_SENSITIVITY_{start.upper()}"],
            end_of_speech_sensitivity=types.EndSensitivity[f"END_SENSITIVITY_{end.upper()}"],
            prefix_padding_ms=200, silence_duration_ms=int(silence_ms))))
    out = []
    async with genai.Client().aio.live.connect(model="gemini-3.5-transcribe-live", config=config) as session:
        t0 = time.monotonic()

        async def receive():
            while True:
                async for m in session.receive():
                    t = time.monotonic() - t0
                    if m.voice_activity:
                        out.append(f"{t:5.2f} {m.voice_activity.voice_activity_type.name}")
                    sc = m.server_content
                    if sc and sc.input_transcription and sc.input_transcription.text:
                        out.append(f"{t:5.2f}   FINAL {sc.input_transcription.text!r}")

        receiver = asyncio.create_task(receive())
        block, sent = 1280, 0
        for i in range(0, len(audio), block):  # paced against the clock, not accumulated sleeps
            await session.send_realtime_input(audio=types.Blob(data=audio[i:i + block], mime_type="audio/pcm;rate=16000"))
            sent += block / 2 / RATE
            await asyncio.sleep(max(0, sent - (time.monotonic() - t0)))
        receiver.cancel()
    print(f"=== start={start} end={end} silence={silence_ms}ms noise_rms={noise_rms}   [{'; '.join(spans)}]")
    print("\n".join(out))


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
