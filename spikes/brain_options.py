"""Spike: which Gemini path gets reply *text* to ElevenLabs fastest?

A) gemini-3.5-transcribe-live (streaming STT, server VAD) → gemini-3.8-flash (streaming text)
B) gemini-3.8-live in audio mode, using its output transcription as the text (audio discarded)

Streams the same caller lines at real-time pace and reports, per line, the time
from the end of the caller's audio to the first reply text.

    python spikes/brain_options.py [A|B ...]
"""

import asyncio
import statistics
import sys
import time

sys.path.insert(0, ".")
from google import genai
from google.genai import types

from phone.config import Config
from tests.e2e.harness import load_line

LINES = ["hello", "q1", "q2", "q3", "q4", "q5"]
SYSTEM = "You are a friendly old telephone operator on a phone call. Reply in one or two short spoken sentences."
cfg = Config.from_env()
client = genai.Client()
BLOCK = cfg.mic_rate * cfg.mic_block_ms // 1000 * 2


async def stream_line(session, pcm, trailing_s=5.0):
    pcm = pcm + b"\0\0" * int(cfg.mic_rate * trailing_s)
    speech_end = None
    for i in range(0, len(pcm), BLOCK):
        await session.send_realtime_input(audio=types.Blob(data=pcm[i:i + BLOCK], mime_type="audio/pcm;rate=16000"))
        if speech_end is None and i + BLOCK >= len(pcm) - int(cfg.mic_rate * trailing_s) * 2:
            speech_end = time.monotonic()
        await asyncio.sleep(cfg.mic_block_ms / 1000)
    return speech_end


def vad():
    return types.RealtimeInputConfig(automatic_activity_detection=types.AutomaticActivityDetection(
        start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
        end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
        prefix_padding_ms=200, silence_duration_ms=cfg.vad_silence_ms))


async def option_a():
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"], realtime_input_config=vad(),
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=["en-US"]))
    history, results = [], []
    async with client.aio.live.connect(model="gemini-3.5-transcribe-live", config=config) as session:
        for name in LINES:
            final = asyncio.get_running_loop().create_future()

            async def listen():
                text = ""
                while True:
                    async for m in session.receive():
                        sc = m.server_content
                        if m.voice_activity:
                            print(f"    [{time.monotonic() - t0:5.2f}] voice activity {m.voice_activity.voice_activity_type}")
                        if sc and sc.input_transcription and sc.input_transcription.text:
                            text += sc.input_transcription.text
                            print(f"    [{time.monotonic() - t0:5.2f}] final: {sc.input_transcription.text!r} "
                                  f"finished={sc.input_transcription.finished}")
                            if not final.done():
                                final.set_result((time.monotonic(), text))
                        if sc and sc.turn_complete:
                            print(f"    [{time.monotonic() - t0:5.2f}] turn_complete")

            t0 = time.monotonic()
            listener = asyncio.create_task(listen())
            sender = asyncio.create_task(stream_line(session, load_line(name)))
            t_final, text = await asyncio.wait_for(final, 20)
            speech_end = await sender if sender.done() else None
            history.append(types.Content(role="user", parts=[types.Part(text=text)]))
            stream = await client.aio.models.generate_content_stream(
                model="gemini-3.8-flash", contents=history,
                config=types.GenerateContentConfig(system_instruction=SYSTEM,
                                                   thinking_config=types.ThinkingConfig(thinking_level="low")))
            first, reply = None, ""
            async for chunk in stream:
                if chunk.text:
                    first = first or time.monotonic()
                    reply += chunk.text
            history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
            speech_end = speech_end or await sender
            listener.cancel()
            results.append((first - speech_end) * 1000)
            print(f"A {name:5} transcript {(t_final - speech_end) * 1000:5.0f} ms, first text "
                  f"{(first - speech_end) * 1000:5.0f} ms  heard={text.strip()!r}  reply={reply.strip()[:60]!r}")
    return results


async def option_b():
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"], system_instruction=SYSTEM, realtime_input_config=vad(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig())
    results = []
    async with client.aio.live.connect(model="gemini-3.8-live", config=config) as session:
        for name in LINES:
            first = {}

            async def listen():
                while True:
                    async for m in session.receive():
                        sc = m.server_content
                        if not sc:
                            continue
                        if sc.model_turn and "audio" not in first:
                            first["audio"] = time.monotonic()
                        if sc.output_transcription and sc.output_transcription.text and "text" not in first:
                            first["text"] = time.monotonic()
                            first["reply"] = sc.output_transcription.text

            listener = asyncio.create_task(listen())
            speech_end = await stream_line(session, load_line(name))
            listener.cancel()
            if "text" in first:
                results.append((first["text"] - speech_end) * 1000)
                print(f"B {name:5} first audio {(first['audio'] - speech_end) * 1000:5.0f} ms, first text "
                      f"{(first['text'] - speech_end) * 1000:5.0f} ms  reply={first['reply']!r}")
            else:
                print(f"B {name:5} no reply")
    return results


async def main(options):
    for option in options:
        results = await {"A": option_a, "B": option_b}[option]()
        if results:
            print(f"==> {option}: median first text {statistics.median(results):.0f} ms after caller audio ended\n")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["A", "B"]))
