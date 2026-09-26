"""How Gemini Live server messages become call events."""

from google.genai import types

from phone.brain import GeminiLiveBrain, Heard, Interrupted, Reply, ReplyDone, SpeechEnded, SpeechStarted
from phone.config import Config

from .fakes import directory


def brain():
    return GeminiLiveBrain(Config(gemini_api_key="test", elevenlabs_api_key=None), directory().roles[1])


def events(b, **server_content):
    message = types.LiveServerMessage(server_content=types.LiveServerContent(**server_content))
    return [type(e).__name__ if not hasattr(e, "text") else (type(e).__name__, e.text) for e in b._translate(message)]


def activity(b, kind):
    message = types.LiveServerMessage(voice_activity=types.VoiceActivity(voice_activity_type=kind))
    return [type(e).__name__ for e in b._translate(message)]


def test_reply_text_comes_from_the_output_transcription_not_the_audio():
    b = brain()
    audio = types.Content(parts=[types.Part(inline_data=types.Blob(data=b"\0\0", mime_type="audio/pcm"))])
    assert events(b, model_turn=audio, output_transcription=types.Transcription(text="Hello there ")) == \
        [("Reply", "Hello there ")]
    assert events(b, turn_complete=True) == ["ReplyDone"]


def test_interrupted_turn_is_not_reported_done():
    assert events(brain(), interrupted=True, turn_complete=True) == ["Interrupted"]


def test_server_voice_activity_marks_speech():
    b = brain()
    assert activity(b, types.VoiceActivityType.ACTIVITY_START) == ["SpeechStarted"]
    assert events(b, input_transcription=types.Transcription(text="hi")) == [("Heard", "hi")]
    assert activity(b, types.VoiceActivityType.ACTIVITY_END) == ["SpeechEnded"]


def test_without_server_voice_activity_first_words_mark_speech_start():
    b = brain()
    assert events(b, interim_input_transcription=types.Transcription(text="wa")) == ["SpeechStarted"]
    assert events(b, input_transcription=types.Transcription(text="wait")) == [("Heard", "wait")]
    events(b, output_transcription=types.Transcription(text="Yes? "))
    assert events(b, input_transcription=types.Transcription(text="again")) == ["SpeechStarted", ("Heard", "again")]


def test_config_asks_for_audio_with_transcripts_both_ways():
    config = brain()._config()
    assert config.response_modalities == [types.Modality.AUDIO]
    assert config.input_audio_transcription is not None
    assert config.output_audio_transcription is not None
