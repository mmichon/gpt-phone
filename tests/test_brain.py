"""Turn-taking in the brain: when the caller is done, and what happens if they carry on."""

import asyncio
import dataclasses

import pytest
from google.genai import types

from phone.brain import SILENCE, UNCLEAR, GeminiBrain
from phone.config import Config

from .fakes import directory


class ScriptedBrain(GeminiBrain):
    """A brain whose replies are released piece by piece by the test."""

    def __init__(self, role_digit=1, **cfg):
        config = dataclasses.replace(Config(gemini_api_key="test", elevenlabs_api_key=None),
                                     turn_grace_s=0.3, turn_grace_continuing_s=0.6, **cfg)
        super().__init__(config, directory().roles[role_digit])
        self.pieces = asyncio.Queue()
        self.asked = []

    async def stream_reply(self, history):
        self.asked.append(history[-1].parts[0].text)
        while (piece := await self.pieces.get()) is not None:
            yield piece

    def interim(self, text):
        self.handle(types.LiveServerMessage(server_content=types.LiveServerContent(
            interim_input_transcription=types.Transcription(text=text))))

    def final(self, text):
        self.handle(types.LiveServerMessage(server_content=types.LiveServerContent(
            input_transcription=types.Transcription(text=text))))

    def say(self, text, interim=None, final=True):
        """A stretch of speech: start, interim transcript, end, then (as in reality) the final one."""
        self.speech_started()
        if interim:
            self.interim(interim)
        self.speech_ended()
        if final:
            self.final(text)

    def reply(self, *pieces):
        for piece in pieces:
            self.pieces.put_nowait(piece)

    def drain(self):
        events = []
        while not self._events.empty():
            events.append(type(self._events.get_nowait()).__name__)
        return events

    def texts(self):
        return [(c.role, c.parts[0].text) for c in self.history]


async def tick(seconds=0.01):
    await asyncio.sleep(seconds)


async def test_a_finished_sentence_is_answered_at_once():
    brain = ScriptedBrain()
    brain.say("Where do you live?")
    await tick()
    assert brain.asked == ["Where do you live?"]
    brain.reply("Up in the hills. ", None)
    await tick()
    assert brain.drain() == ["SpeechStarted", "SpeechEnded", "Heard", "Reply", "ReplyDone"]
    assert brain.texts() == [("user", "Where do you live?"), ("model", "Up in the hills. ")]


async def test_a_complete_interim_transcript_starts_the_reply_before_the_final_one():
    brain = ScriptedBrain()
    brain.say("Where do you live?", interim="Where do you live?", final=False)
    await tick()
    assert brain.asked == ["Where do you live?"]
    brain.final("Where do you live?")
    await tick()
    assert brain.asked == ["Where do you live?"], "same words: no need to start over"


async def test_a_final_transcript_that_differs_restarts_the_reply():
    brain = ScriptedBrain()
    brain.say("", interim="Where do you leave?", final=False)
    await tick()
    brain.final("Where do you live?")
    await tick()
    assert brain.asked == ["Where do you leave?", "Where do you live?"]
    assert brain.texts() == [("user", "Where do you live?")]


async def test_an_unfinished_interim_waits_for_the_final_transcript():
    brain = ScriptedBrain()
    brain.say("", interim="Where do you", final=False)
    await tick()
    assert brain.asked == []
    brain.final("Where do you live?")
    await tick()
    assert brain.asked == ["Where do you live?"]


async def test_a_pause_mid_sentence_waits_for_the_rest():
    brain = ScriptedBrain()
    brain.say("So I was walking down by the old train station")
    await tick(0.15)
    assert brain.asked == [], "shouldn't answer during a mid-sentence pause"
    brain.say("and I saw a dog in a hat. What was it doing?")
    await tick()
    assert brain.asked == ["So I was walking down by the old train station and I saw a dog in a hat. What was it doing?"]


async def test_a_long_pause_mid_sentence_is_eventually_answered():
    brain = ScriptedBrain()
    brain.say("I was wondering whether it")
    await tick(0.4)
    assert brain.asked == ["I was wondering whether it"]


async def test_a_trailing_and_or_comma_waits_longer():
    brain = ScriptedBrain()
    brain.say("I went to the store and")
    await tick(0.4)
    assert brain.asked == []
    await tick(0.3)
    assert brain.asked == ["I went to the store and"]


async def test_carrying_on_before_hearing_the_reply_withdraws_it():
    """The transcriber put a period at a thinking pause; the caller goes on before hearing the answer."""
    brain = ScriptedBrain()
    brain.say("So the other day I was at the station.")
    await tick()
    brain.reply("Oh, the station! ")  # reply text is streaming, but hasn't been heard yet
    await tick()
    brain.say("And I saw a dog in a hat. Why?")
    await tick()
    assert "Retracted" in brain.drain()
    assert brain.asked[-1] == "So the other day I was at the station. And I saw a dog in a hat. Why?"
    assert brain.texts() == [("user", "So the other day I was at the station. And I saw a dog in a hat. Why?")]


async def test_a_finished_but_unheard_reply_is_withdrawn_too():
    brain = ScriptedBrain()
    brain.say("Hi.")
    await tick()
    brain.reply("Hello there! ", None)
    await tick()
    brain.say("It's Sam. Who's this?")
    await tick()
    assert brain.asked[-1] == "Hi. It's Sam. Who's this?"
    assert brain.texts() == [("user", "Hi. It's Sam. Who's this?")]


async def test_talking_over_a_heard_reply_interrupts_it():
    brain = ScriptedBrain()
    brain.say("Tell me a story.")
    await tick()
    brain.reply("Once upon a time, ")
    await tick()
    brain.mark_heard()
    brain.say("Wait, stop!")
    await tick()
    events = brain.drain()
    assert "Interrupted" in events and "ReplyDone" not in events
    assert brain.texts() == [("user", "Tell me a story."), ("model", "Once upon a time, —"),
                             ("user", "Wait, stop!")]


async def test_a_heard_and_finished_reply_stays_when_the_caller_goes_on():
    brain = ScriptedBrain()
    brain.say("Hi.")
    await tick()
    brain.reply("Hello there! ", None)
    await tick()
    brain.mark_heard()
    brain.say("How are you?")
    await tick()
    assert brain.texts() == [("user", "Hi."), ("model", "Hello there! "), ("user", "How are you?")]


def test_listening_is_text_only_with_local_voice_detection():
    config = ScriptedBrain(role_digit=7)._listen_config()
    assert config.response_modalities == [types.Modality.TEXT]
    assert config.realtime_input_config.automatic_activity_detection.disabled
    assert config.input_audio_transcription.language_codes == ["en-US"]
    polish = dataclasses.replace(directory().roles[7], language="pl")
    brain = GeminiBrain(Config(gemini_api_key="test", elevenlabs_api_key=None), polish)
    assert brain._listen_config().input_audio_transcription.language_codes == ["pl-PL"]


class Session:
    """Records what the brain sends to Gemini."""

    def __init__(self):
        self.sent = []

    async def send_realtime_input(self, **message):
        self.sent.append(next(iter(message)))


async def test_a_new_stretch_waits_for_the_previous_transcript():
    brain = ScriptedBrain()
    brain.session = Session()
    brain._sender = asyncio.create_task(brain._send_loop())
    brain._outbox.put_nowait({"activity_start": None})
    brain._outbox.put_nowait({"audio": None})
    brain._outbox.put_nowait({"activity_end": None})
    brain._outbox.put_nowait({"activity_start": None})  # the caller carries on right away
    brain._outbox.put_nowait({"audio": None})
    await tick(0.05)
    assert brain.session.sent == ["activity_start", "audio", "activity_end"], "held until the transcript"
    brain.final("Hi there.")
    await tick()
    assert brain.session.sent[3:] == ["activity_start", "audio"]
    brain._sender.cancel()


async def test_replies_to_statements_are_held_and_to_questions_are_not():
    brain = ScriptedBrain(statement_hold_s=0.9)
    brain.say("So I was at the station.")
    await tick()
    assert brain._exchange.hold_until is not None
    assert brain._exchange.hold_until - brain._speech_ended_at == pytest.approx(0.9)
    brain.mark_heard()
    brain.reply("Oh? ", None)
    await tick()
    brain.say("What was it doing there?")
    await tick()
    assert brain._exchange.hold_until is None


async def test_a_missing_final_transcript_falls_back_to_the_interim_one():
    brain = ScriptedBrain()
    brain.say("", interim="Yes", final=False)
    await tick()
    assert brain.asked == [], "waiting for the final transcript"
    brain._transcript_overdue()
    await tick(0.4)
    assert brain.asked == ["Yes"]
    brain.final("Yes.")  # turns up after all: already answered
    await tick(0.4)
    assert brain.asked == ["Yes"] and brain.texts() == [("user", "Yes")]


async def test_speech_with_no_words_at_all_is_asked_about():
    brain = ScriptedBrain()
    brain.say("", final=False)
    brain._transcript_overdue()
    await tick()
    assert brain.asked == [UNCLEAR]
    brain.reply("Say again? ", None)
    await tick()
    brain.mark_heard()
    brain.say("Yes.")
    await tick()
    assert brain.texts() == [("user", UNCLEAR), ("model", "Say again? "), ("user", "Yes.")]


async def test_a_transcript_that_arrives_in_time_is_used_as_usual():
    brain = ScriptedBrain()
    brain.say("Yes.")
    brain._transcript_overdue()
    await tick()
    assert brain.asked == ["Yes."]


async def test_carrying_on_before_hearing_a_note_reply_drops_the_note():
    brain = ScriptedBrain()
    brain.say("", final=False)
    brain._transcript_overdue()
    await tick()
    brain.say("I said yes.")
    await tick()
    assert brain.asked[-1] == "I said yes."
    assert brain.texts() == [("user", "I said yes.")]


async def test_a_quiet_caller_is_nudged_only_when_nothing_else_is_going_on():
    brain = ScriptedBrain()
    brain.say("Hi.")
    await tick()
    assert not brain.nudge(), "still replying"
    brain.reply("Hello! Who's this? ", None)
    await tick()
    assert brain.nudge()
    await tick()
    assert brain.asked[-1] == SILENCE
    assert [role for role, _ in brain.texts()] == ["user", "model", "user"]


async def test_a_caller_quiet_after_the_greeting_is_nudged():
    brain = ScriptedBrain()
    assert brain.nudge()
    await tick()
    assert brain.asked == [SILENCE]
