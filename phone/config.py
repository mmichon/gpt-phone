"""Runtime configuration, read from the environment (systemd EnvironmentFile or .env)."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent


def _env(name, default=None):
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_bool(name, default):
    value = _env(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    gemini_api_key: str | None
    elevenlabs_api_key: str | None

    # Models
    listen_model: str = "gemini-3.5-transcribe-live"   # streaming speech-to-text
    text_model: str = "gemini-3.5-flash-lite"          # writes the replies; fast first token
    text_thinking: str | None = "minimal"
    tts_model: str = "eleven_flash_v2_5"

    # Files
    roles_file: Path = REPO_DIR / "roles.yaml"
    sounds_dir: Path = REPO_DIR / "sounds"
    cache_dir: Path = Path.home() / ".cache" / "gpt-phone"

    # Hardware (BCM pin numbers)
    hook_gpio: int = 14
    dial_gpio: int = 15

    # Audio
    mic_rate: int = 16000        # what Gemini Live expects
    out_rate: int = 24000        # what ElevenLabs pcm_24000 produces
    mic_block_ms: int = 32       # one Silero VAD frame (512 samples)

    # Conversation behavior
    barge_in: bool = True        # let callers interrupt the character (needs echo cancellation)
    vad_threshold: float = 0.5   # Silero speech probability that counts as speech
    vad_start_ms: int = 96       # this much speech starts a turn (so clicks don't)
    vad_silence_ms: int = 500    # this much silence ends a stretch of speech
    vad_preroll_ms: int = 320    # audio kept from just before speech was confirmed
    turn_grace_s: float = 0.8    # extra wait when a pause comes mid-sentence
    turn_grace_continuing_s: float = 1.2  # ...or right after "and", a comma, etc.
    heard_after_s: float = 0.5   # once this much of a reply has played, the caller has heard it
    still_there_s: float = 20.0  # caller silence before "are you still there?"
    give_up_s: float = 120.0     # caller silence before the call is dropped

    # Timeouts (seconds)
    connect_timeout: float = 5.0
    first_token_timeout: float = 10.0
    tts_connect_timeout: float = 5.0

    @classmethod
    def from_env(cls):
        defaults = cls(gemini_api_key=None, elevenlabs_api_key=None)
        return cls(
            gemini_api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
            elevenlabs_api_key=_env("ELEVENLABS_API_KEY") or _env("ELEVENLABS_KEY"),
            listen_model=_env("PHONE_LISTEN_MODEL", defaults.listen_model),
            text_model=_env("PHONE_TEXT_MODEL", defaults.text_model),
            text_thinking=_env("PHONE_TEXT_THINKING", defaults.text_thinking),
            tts_model=_env("PHONE_TTS_MODEL", defaults.tts_model),
            roles_file=Path(_env("PHONE_ROLES_FILE", defaults.roles_file)).expanduser(),
            cache_dir=Path(_env("PHONE_CACHE_DIR", defaults.cache_dir)).expanduser(),
            barge_in=_env_bool("PHONE_BARGE_IN", defaults.barge_in),
            vad_threshold=float(_env("PHONE_VAD_THRESHOLD", defaults.vad_threshold)),
            vad_silence_ms=int(_env("PHONE_VAD_SILENCE_MS", defaults.vad_silence_ms)),
            turn_grace_s=float(_env("PHONE_TURN_GRACE_S", defaults.turn_grace_s)),
        )
