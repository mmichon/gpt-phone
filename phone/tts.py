"""Text to speech: ElevenLabs over a streaming WebSocket, a disk cache for fixed
prompts, and helpers that turn streamed LLM text into speakable phrases."""

import asyncio
import base64
import hashlib
import json
import logging
import re

import websockets
from websockets.protocol import State

log = logging.getLogger(__name__)

# Models that accept language_code to force the output language.
LANGUAGE_CODE_MODELS = {"eleven_flash_v2_5", "eleven_turbo_v2_5"}


class TTSError(Exception):
    pass


class ElevenLabsStream:
    """One utterance. Send phrases as they arrive; read PCM as it is generated."""

    def __init__(self, ws):
        self._ws = ws
        self._ended = False

    @property
    def open(self):
        return self._ws.state is State.OPEN

    async def send(self, phrase):
        await self._ws.send(json.dumps({"text": phrase.strip() + " "}))

    async def end(self):
        """No more text: generate what's left, then finish."""
        if not self._ended:
            self._ended = True
            await self._ws.send(json.dumps({"text": ""}))

    async def audio(self):
        carry = b""
        async for message in self._ws:
            data = json.loads(message)
            if data.get("error") or (data.get("message") and not data.get("audio")):
                raise TTSError(f"ElevenLabs: {data}")
            if data.get("audio"):
                pcm = carry + base64.b64decode(data["audio"])
                cut = len(pcm) - len(pcm) % 2  # keep whole 16-bit samples
                carry = pcm[cut:]
                yield pcm[:cut]
            if data.get("isFinal"):
                return

    async def close(self):
        await self._ws.close()


class ElevenLabsTTS:
    URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"

    def __init__(self, api_key, model, rate=24000, connect_timeout=5.0):
        self.api_key = api_key
        self.model = model
        self.rate = rate
        self.connect_timeout = connect_timeout

    async def open(self, voice_id, model=None, language=None):
        model = model or self.model
        params = {
            "model_id": model,
            "output_format": f"pcm_{self.rate}",
            "auto_mode": "true",          # we send whole phrases; don't wait to buffer more
            "inactivity_timeout": "120",  # streams are opened early, while the caller talks
        }
        if language and model in LANGUAGE_CODE_MODELS:
            params["language_code"] = language
        url = self.URL.format(voice_id=voice_id) + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        ws = await websockets.connect(
            url, additional_headers={"xi-api-key": self.api_key},
            open_timeout=self.connect_timeout, max_size=None)
        # The voice's own saved settings apply, since we send no voice_settings.
        await ws.send(json.dumps({"text": " "}))
        return ElevenLabsStream(ws)

    async def synthesize(self, text, voice_id, model=None, language=None):
        stream = await self.open(voice_id, model, language)
        try:
            await stream.send(text)
            await stream.end()
            return b"".join([pcm async for pcm in stream.audio()])
        finally:
            await stream.close()


class PromptCache:
    """Fixed phrases rendered once and kept on disk, so they play instantly and offline."""

    def __init__(self, tts, cache_dir, rate):
        self.tts = tts
        self.dir = cache_dir / "prompts"
        self.rate = rate

    def path(self, voice_id, model, text):
        key = f"{voice_id}|{model or self.tts.model}|{self.rate}|{text}"
        return self.dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".pcm")

    def cached(self, voice_id, model, text):
        path = self.path(voice_id, model, text)
        return path.read_bytes() if path.exists() else None

    async def get(self, voice_id, model, text, language=None):
        pcm = self.cached(voice_id, model, text)
        if pcm is None:
            pcm = await self.tts.synthesize(text, voice_id, model, language)
            if not pcm:
                raise TTSError(f"No audio for {text!r}")
            path = self.path(voice_id, model, text)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(pcm)
            tmp.replace(path)
        return pcm

    async def warm(self, prompts):
        """Render anything missing. Returns how many prompts are still unavailable."""
        missing = 0
        for voice_id, model, text in prompts:
            try:
                await self.get(voice_id, model, text)
            except Exception as e:
                missing += 1
                log.warning("Couldn't pre-render %r: %s", text[:40], e)
        return missing


ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "mt", "no"}
SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s")
CLAUSE_END = re.compile(r"[,;:—–]\s")
STAGE_DIRECTION = re.compile(r"[\[(][^\])]{0,40}[\])]|(?<!\*)\*(?!\*)[^*]{1,40}(?<!\*)\*(?!\*)")
UNSPEAKABLE = re.compile(r"[*_#`~>|]|[\U0001F000-\U0001FAFF☀-➿️]")


def speakable(text):
    """Strip markdown, emoji and stage directions like *laughs* or (winks) that TTS would read out."""
    text = STAGE_DIRECTION.sub("", text)
    text = UNSPEAKABLE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class PhraseChunker:
    """Groups streamed reply text into phrases to send to TTS as each completes.

    The first phrase is released at the first clause break so speech starts
    quickly; after that, phrases end at sentences, which sound more natural.
    """

    FIRST_MIN = 12   # characters before a clause break can end the first phrase
    MAX = 200        # past this, break at any space

    def __init__(self):
        self.buffer = ""
        self.first = True

    def feed(self, text):
        self.buffer += text
        phrases = []
        while (cut := self._cut()) is not None:
            phrase, self.buffer = self.buffer[:cut].strip(), self.buffer[cut:]
            if phrase:
                phrases.append(phrase)
                self.first = False
        return phrases

    def finish(self):
        phrase, self.buffer, self.first = self.buffer.strip(), "", True
        return [phrase] if phrase else []

    def _cut(self):
        for m in SENTENCE_END.finditer(self.buffer):
            word = self.buffer[:m.start()].rsplit(None, 1)[-1:] or [""]
            if word[0].lower().rstrip(".") not in ABBREVIATIONS:
                return m.end()
        if self.first:
            m = CLAUSE_END.search(self.buffer, self.FIRST_MIN)
            if m:
                return m.end()
        if len(self.buffer) > self.MAX:
            space = self.buffer.rfind(" ", 0, self.MAX)
            if space > 0:
                return space + 1
        return None
