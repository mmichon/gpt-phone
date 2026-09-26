"""The call pipeline with a scripted brain and a fake TTS: streaming, barge-in, hang-up."""

import asyncio

import pytest

from phone.brain import BrainError, Heard, Interrupted, Reply, ReplyDone, SpeechEnded, SpeechStarted
from phone.call import Call
from phone.config import Config
from phone.switchboard import Deps

from .fakes import FakeCache, FakePlayer, FakeSounds, directory

log = []  # ordered record of what happened, across fakes


class FakeBrain:
    def __init__(self, cfg, role, fail_connect=False):
        self.script = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.fail_connect = fail_connect

    async def connect(self):
        if self.fail_connect:
            raise BrainError("no connection")

    async def close(self):
        self.closed = True

    async def send_audio(self, pcm):
        self.sent.append(pcm)

    def mark_heard(self):
        self.heard = True

    def ask_to_repeat(self):
        return False

    def nudge(self):
        self.nudges = getattr(self, "nudges", 0) + 1
        return True

    async def events(self):
        while True:
            event = await self.script.get()
            log.append(("brain", type(event).__name__))
            yield event


class FakeStream:
    def __init__(self):
        self.phrases = []
        self.audio_q = asyncio.Queue()
        self.closed = False
        self.open = True

    async def send(self, phrase):
        log.append(("tts", phrase))
        self.phrases.append(phrase)
        self.audio_q.put_nowait(phrase.encode())

    async def end(self):
        self.audio_q.put_nowait(None)

    async def audio(self):
        while (pcm := await self.audio_q.get()) is not None:
            yield pcm

    async def close(self):
        self.closed = True
        self.open = False


class FakeTTS:
    def __init__(self):
        self.streams = []

    async def open(self, voice_id, model=None, language=None):
        stream = FakeStream()
        self.streams.append(stream)
        return stream


class FakeMic:
    def clear(self):
        pass

    async def frames(self):
        while True:
            await asyncio.sleep(0.01)
            yield b"\x01\x00" * 4


@pytest.fixture
def setup():
    log.clear()
    cfg = Config(None, None, still_there_s=3600, give_up_s=3600)
    role = directory().roles[7]
    tts, player = FakeTTS(), FakePlayer()
    deps = Deps(mic=FakeMic(), player=player, tts=tts, cache=FakeCache(), sounds=FakeSounds())
    observed = []
    brains = []

    def make(cfg=cfg, fail_connect=False):
        def brain_factory(c, r):
            brains.append(FakeBrain(c, r, fail_connect))
            return brains[-1]
        return Call(cfg, role, deps, observe=lambda kind, **d: observed.append((kind, d)),
                    brain_factory=brain_factory)

    return make, brains, tts, player, observed


async def start(call):
    task = asyncio.create_task(call.run())
    await asyncio.sleep(0.05)  # ringback + greeting + connect
    return task


async def test_speech_starts_before_the_reply_is_finished(setup):
    make, brains, tts, player, _ = setup
    task = await start(make())
    brain = brains[0]
    for event in [SpeechStarted(), SpeechEnded(), Reply("Well hello there, "), Reply("honey child. ")]:
        brain.script.put_nowait(event)
    await asyncio.sleep(0.02)
    assert ("tts", "Well hello there,") in log  # spoken while the reply is still streaming
    brain.script.put_nowait(Reply("What's your name?"))
    brain.script.put_nowait(ReplyDone())
    await asyncio.sleep(0.02)
    assert tts.streams[0].phrases == ["Well hello there,", "honey child.", "What's your name?"]
    assert b"What's your name?" in player.played
    task.cancel()


async def test_caller_talking_over_the_reply_stops_it(setup):
    make, brains, tts, player, observed = setup
    task = await start(make())
    brain = brains[0]
    brain.script.put_nowait(Reply("This is a long story about the gold rush. "))
    await asyncio.sleep(0.02)
    flushes = player.flushes
    brain.script.put_nowait(SpeechStarted())
    await asyncio.sleep(0.02)
    assert player.flushes > flushes
    assert tts.streams[0].closed
    assert ("output_stopped", {}) in observed
    brain.script.put_nowait(Interrupted())
    brain.script.put_nowait(Reply("Oh, sorry. What was that? "))
    brain.script.put_nowait(ReplyDone())
    await asyncio.sleep(0.02)
    assert tts.streams[-1].phrases == ["Oh, sorry.", "What was that?"]
    task.cancel()


async def test_half_duplex_mutes_the_mic_while_speaking(setup):
    make, brains, tts, player, _ = setup
    task = await start(make(cfg=Config(None, None, barge_in=False)))
    player.playing = True
    await asyncio.sleep(0.05)
    assert brains[0].sent[-1] == b"\x00" * len(brains[0].sent[-1])
    task.cancel()


async def test_latency_is_reported_per_turn(setup):
    make, brains, tts, player, observed = setup
    call = make()
    task = await start(call)
    brains[0].script.put_nowait(SpeechEnded())
    brains[0].script.put_nowait(Reply("Hi. "))
    await asyncio.sleep(0.02)
    player.on_start(asyncio.get_running_loop().time())  # the audio thread saw playback begin
    latency = [d for kind, d in observed if kind == "latency"]
    assert latency and latency[0]["turn"] == 1 and {"text", "tts", "play"} <= latency[0].keys()
    task.cancel()


async def test_hanging_up_closes_everything(setup):
    make, brains, tts, player, _ = setup
    task = await start(make())
    brains[0].script.put_nowait(Reply("Let me tell you, "))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert brains[0].closed
    assert all(s.closed for s in tts.streams)


async def test_connection_failure_raises(setup):
    make, *_ = setup
    with pytest.raises(BrainError):
        await make(fail_connect=True).run()


async def test_silent_caller_is_prompted_then_dropped(setup):
    make, brains, tts, player, _ = setup
    cfg = Config(None, None, still_there_s=1, give_up_s=2.5)
    await asyncio.wait_for(make(cfg=cfg).run(), timeout=5)
    assert b"Hello? Are you still there?" in player.played


async def test_a_reply_counts_as_heard_once_enough_has_played(setup):
    make, brains, tts, player, _ = setup
    import dataclasses
    task = await start(make(cfg=dataclasses.replace(Config(None, None, still_there_s=3600, give_up_s=3600),
                                                    heard_after_s=0.05)))
    brain = brains[0]
    brain.heard = False
    brain.script.put_nowait(SpeechEnded())
    brain.script.put_nowait(Reply("Hello there. "))
    await asyncio.sleep(0.02)
    player.on_start(asyncio.get_running_loop().time())
    assert not brain.heard
    await asyncio.sleep(0.08)
    assert brain.heard
    task.cancel()


async def test_a_retracted_reply_is_silenced(setup):
    from phone.brain import Retracted
    make, brains, tts, player, observed = setup
    task = await start(make())
    brains[0].script.put_nowait(Reply("Oh, the station! "))
    await asyncio.sleep(0.02)
    flushes = player.flushes
    brains[0].script.put_nowait(Retracted())
    await asyncio.sleep(0.02)
    assert player.flushes > flushes and tts.streams[0].closed
    assert ("retracted", {}) in observed
    task.cancel()


async def test_a_held_reply_does_not_play_early(setup):
    import time
    make, brains, tts, player, _ = setup
    task = await start(make())
    hold = time.monotonic() + 0.15
    brains[0].script.put_nowait(Reply("Go on. ", hold_until=hold))
    brains[0].script.put_nowait(ReplyDone())
    await asyncio.sleep(0.05)
    assert b"Go on." not in player.played, "held: shouldn't play yet"
    await asyncio.sleep(0.2)
    assert b"Go on." in player.played
    task.cancel()


async def test_a_quiet_caller_is_nudged_once_then_asked_if_they_are_there(setup):
    make, brains, tts, player, _ = setup
    cfg = Config(None, None, nudge_s=1, still_there_s=2.5, give_up_s=3600)
    task = await start(make(cfg=cfg))
    brain = brains[0]
    await asyncio.sleep(1.1)
    assert getattr(brain, "nudges", 0) == 1
    await asyncio.sleep(2)
    assert brain.nudges == 1, "once per silence"
    assert b"Hello? Are you still there?" in player.played
    task.cancel()


async def test_speech_without_words_does_not_reset_the_silence(setup):
    make, brains, tts, player, _ = setup
    cfg = Config(None, None, nudge_s=1, still_there_s=3600, give_up_s=3600)
    task = await start(make(cfg=cfg))
    brain = brains[0]
    await asyncio.sleep(0.5)
    brain.script.put_nowait(SpeechStarted())
    brain.script.put_nowait(SpeechEnded())  # a noise: no Heard follows
    await asyncio.sleep(0.7)
    assert getattr(brain, "nudges", 0) == 1
    brain.script.put_nowait(Heard("Yes."))  # real words: they can be nudged again later
    await asyncio.sleep(2.1)
    assert brain.nudges == 2
    task.cancel()
