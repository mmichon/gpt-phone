"""Spike: time to first token of Gemini text models, for a phone-call persona.

    python spikes/ttft.py MODEL [THINKING_LEVEL]
"""

import asyncio
import statistics
import sys
import time

from google import genai
from google.genai import types

sys.path.insert(0, ".")
from phone.roles import Role

SYSTEM = Role(digit=3, name="An ornery old prospector", voice_id="-", greeting="Well if it aint a caller on this here phone. What's your name, stranger?",
              persona="You are an old gold prospector. You offer unsolicited advice. You try to give people directions even when "
                      "they don't ask for them. Recount your life and crazy stories to the caller. Don't lead every sentence "
                      "with hey or hello. Never use emoji or unicode characters in your responses.").system_instruction()
HISTORY = [("user", "Hi, my name is Sam."), ("model", "Well howdy, Sam! Pull up a stump and set a spell. You sound like a city slicker to me."),
           ("user", "I am, I live in San Francisco."), ("model", "San Francisco! Why, I panned for gold not fifty miles east of there, back in my younger days.")]
QUESTIONS = ["What is your favorite color?", "Where do you live?", "Do you like music?",
             "What did you have for breakfast?", "How old are you?", "Tell me about your mule."]


async def main(model, level=None):
    client = genai.Client()
    label = level or "default"
    times = []
    for question in QUESTIONS:
        extra = {"thinking_config": types.ThinkingConfig(thinking_level=level)} if level else {}
        config = types.GenerateContentConfig(system_instruction=SYSTEM,
                                             automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), **extra)
        contents = [types.Content(role=r, parts=[types.Part(text=t)]) for r, t in HISTORY] + \
                   [types.Content(role="user", parts=[types.Part(text=question)])]
        start = time.monotonic()
        try:
            async for chunk in await client.aio.models.generate_content_stream(
                    model=model, contents=contents, config=config):
                if chunk.text:
                    times.append((time.monotonic() - start) * 1000)
                    break
        except Exception as e:
            print(f"{model:28} {label:8} FAIL {str(e)[:80]}")
            return
    print(f"{model:28} {label:8} median {statistics.median(times):5.0f} ms "
          f"(min {min(times):4.0f}, max {max(times):4.0f})")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
