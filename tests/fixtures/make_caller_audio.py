"""Render the synthetic caller's lines (lines.yaml) to 16 kHz WAVs with Gemini TTS.

Run once, or after editing lines.yaml, then commit the WAVs:
    GEMINI_API_KEY=... python tests/fixtures/make_caller_audio.py [--force]
"""

import base64
import os
import sys
import wave
from pathlib import Path

import yaml
from google import genai

HERE = Path(__file__).parent / "caller"
MODEL = os.environ.get("PHONE_FIXTURE_TTS_MODEL", "gemini-3.8-flash-lite-tts")
RATE = 16000


def render(client, voice, text, style=None):
    annotations = [{"type": "speech_metadata", "style": style}] if style else []
    interaction = client.interactions.create(
        model=MODEL,
        input=[{"type": "user_input",
                "content": [{"type": "text", "text": text, "annotations": annotations}]}],
        response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": RATE},
        generation_config={"speech_config": [{"voice": voice}]},
    )
    return base64.b64decode(interaction.output_audio.data)


def main():
    force = "--force" in sys.argv
    spec = yaml.safe_load((HERE / "lines.yaml").read_text())
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for name, line in spec["lines"].items():
        path = HERE / f"{name}.wav"
        if path.exists() and not force:
            continue
        line = line if isinstance(line, dict) else {"text": line}
        pcm = render(client, spec["voice"], line["text"], line.get("style"))
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        print(f"{name}: {len(pcm) / 2 / RATE:.1f}s")


if __name__ == "__main__":
    main()
