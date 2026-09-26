"""End-to-end calls on the real phone: run on the Pi with `deploy/deploy.sh --test`.

The handset plays everything (quietly); a synthetic caller is mixed into the
real, echo-cancelled mic.
"""

import asyncio
import dataclasses
import statistics
import time

import pytest

from phone import tts as tts_module
from phone.config import Config
from phone.roles import load_directory

from .conftest import Rig
from .harness import load_line, looks_polish, similarity, transcribe

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio(loop_scope="session")]

HALLUCINATIONS = ["thank you for watching", "thanks for watching", "subscribe", "have a great weekend",
                  "see you next time"]


def role_digits():
    try:
        return sorted(load_directory(Config.from_env().roles_file).roles)
    except FileNotFoundError:
        return []


def an_english_role(directory):
    return next(d for d, r in sorted(directory.roles.items()) if r.language == "en")


async def converse(rig, *parts, timeout=30):
    """Say something, wait for the full reply to play. Returns a dict describing the turn."""
    start, end = await rig.say(*parts)
    done, _ = await rig.obs.wait_for("reply_done", after=start, timeout=timeout)
    await rig.wait_quiet(after=done)
    first_audio = min(rig.tap.times(end), default=None)
    latency = [d for t, d in rig.obs.of("latency", after=start)]
    return {
        "start": start, "end": end,
        "heard": rig.obs.text("heard", after=start),
        "reply": rig.obs.text("reply", after=start),
        "first_reply_t": min((t for t, _ in rig.obs.of("reply", after=start)), default=None),
        "latency_ms": round((first_audio - end) * 1000) if first_audio else None,
        "internal": latency[0] if latency else None,
        "played": rig.tap.between(end),
    }


@pytest.mark.parametrize("digit", role_digits())
async def test_every_role_answers(rig, directory, report, digit):
    role = directory.roles[digit]
    t0 = await rig.connect_to(digit)
    assert rig.tap.seconds(t0) > 1.0, "ringback and greeting should be audible"

    turn = await converse(rig, "polish_hello" if role.language == "pl" else "hello")
    report.add(f"role {digit}", **{k: turn[k] for k in ("heard", "reply", "latency_ms", "internal")})
    assert turn["reply"].strip(), "the character should reply"
    assert len(turn["played"]) / 2 / rig.cfg.out_rate > 0.5, "the reply should be audible"
    spoken = await transcribe(turn["played"], rig.cfg.out_rate, rig.cfg.gemini_api_key)
    assert similarity(spoken, turn["reply"]) >= 0.6, f"played {spoken!r} but meant {turn['reply']!r}"
    if role.language == "pl":
        assert looks_polish(turn["reply"]), f"{role.name} should answer in Polish"


async def test_long_utterance_is_not_cut_off(rig, directory, report):
    await rig.connect_to(an_english_role(directory))
    turn = await converse(rig, "long_1", 1.2, "long_2", 1.2, "long_3", timeout=40)
    report.add("long utterance", **{k: turn[k] for k in ("heard", "reply", "latency_ms")})
    assert turn["first_reply_t"] > turn["end"], "the character replied before the caller finished"
    heard = turn["heard"].lower()
    assert "train" in heard and ("hat" in heard or "doing there" in heard), \
        f"the transcript should span the whole utterance: {heard!r}"
    assert not rig.obs.of("interrupted", after=turn["start"])


async def test_silence_and_noise_are_not_heard_as_speech(rig, directory, report):
    t0 = await rig.connect_to(an_english_role(directory))
    await asyncio.sleep(10)
    await rig.say("cough")
    await asyncio.sleep(20)
    heard = rig.obs.text("heard", after=t0)
    report.add("silence", heard=heard)
    assert not rig.obs.of("reply", after=t0), f"the character replied to silence/noise (heard {heard!r})"
    assert rig.tap.seconds(t0 + 1) < 0.5, "nothing should play during silence"
    assert not any(h in heard.lower() for h in HALLUCINATIONS), f"hallucinated speech: {heard!r}"


async def test_character_does_not_interrupt_itself(rig, directory):
    await rig.connect_to(an_english_role(directory))
    start, _ = await rig.say("story_request")
    await rig.wait_playing(after=start, seconds=1)
    done, _ = await rig.obs.wait_for("reply_done", after=start, timeout=60)
    await rig.wait_quiet(after=done)
    assert rig.tap.seconds(start) > 5, "expected a long story"
    assert not rig.obs.of("interrupted", after=start), "echo from the earpiece interrupted the character"
    assert not rig.obs.of("output_stopped", after=start)


async def test_caller_can_interrupt(rig, directory, report):
    await rig.connect_to(an_english_role(directory))
    start, _ = await rig.say("story_request")
    await rig.wait_playing(after=start, seconds=2)
    rig.mic.say(load_line("interrupt"), "interrupt")
    await asyncio.sleep(0.1)
    spoke_at = rig.mic.spans[-1][0]
    stopped, _ = await rig.obs.wait_for("output_stopped", after=spoke_at, timeout=3)
    detected, _ = await rig.obs.wait_for("speech_started", after=spoke_at - 0.5, timeout=3)
    report.add("barge-in", latency_ms=round((stopped - spoke_at) * 1000))
    assert stopped - detected <= 0.3, "audio should stop within 300 ms of the interruption being detected"
    assert stopped - spoke_at <= 0.8, "audio should stop within 800 ms of the caller starting to talk"

    while rig.mic.talking:
        await asyncio.sleep(0.05)
    end = rig.mic.spans[-1][1]
    await rig.obs.wait_for("reply_done", after=end, timeout=30)
    answer_t = min(t for t, _ in rig.obs.of("reply", after=end))
    assert not rig.tap.times(stopped + 0.15, answer_t), "the old reply kept playing after the interruption"
    assert rig.obs.text("reply", after=end).strip(), "the character should answer the interruption"


async def test_hanging_up_mid_reply_stops_everything(rig, directory):
    await rig.connect_to(an_english_role(directory))
    tasks_before = set(asyncio.all_tasks())
    start, _ = await rig.say("story_request")
    await rig.wait_playing(after=start, seconds=1)
    hung_up = time.monotonic()
    rig.hw.hang_up()
    await asyncio.wait_for(rig.board.wait_idle(), 1.0)
    await asyncio.sleep(0.5)
    assert not rig.tap.times(hung_up + 0.1), "audio kept playing after hang-up"
    assert rig.obs.of("status", after=hung_up)[-1][1]["text"] == "idle"
    assert rig.obs.of("call_end", after=hung_up)
    leaked = [t for t in asyncio.all_tasks() - tasks_before if not t.done()
              and any(name in repr(t.get_coro()) for name in ("Call.", "Speaker.", "GeminiLive"))]
    assert not leaked, f"tasks left running after hang-up: {leaked}"


async def test_latency(rig, directory, report):
    latencies = []
    english = [d for d, r in sorted(directory.roles.items()) if r.language == "en"][:2]
    for digit in english:
        await rig.connect_to(digit)
        for line in ("q1", "q2", "q3", "q4", "q5"):
            turn = await converse(rig, line)
            report.add(f"latency role {digit}", turn=line,
                       **{k: turn[k] for k in ("heard", "reply", "latency_ms", "internal")})
            if turn["latency_ms"] is not None:
                latencies.append(turn["latency_ms"])
        await rig.hang_up()
        await asyncio.sleep(1)
    p50 = statistics.median(latencies)
    p90 = statistics.quantiles(latencies, n=10)[-1]
    report.add("latency summary", latency_ms=f"p50 {p50:.0f}, p90 {p90:.0f}, n={len(latencies)}")
    assert p50 <= 1500, f"p50 {p50:.0f} ms"
    assert p90 <= 2500, f"p90 {p90:.0f} ms"


async def test_api_failure_plays_busy_message(cfg, directory, devices):
    bad = dataclasses.replace(cfg, gemini_api_key="not-a-real-key")
    rig = Rig(bad, directory, devices)
    t0 = await rig.connect_to(an_english_role(directory))
    await rig.obs.wait_for("call_failed", after=t0, timeout=15)
    await rig.obs.wait_for("status", after=t0, timeout=30, where=lambda d: d["text"] == "off hook: reorder")
    assert rig.tap.seconds(t0) > 2, "the busy message and reorder tone should play"
    await rig.hang_up()
    t1 = time.monotonic()
    rig.hw.lift()
    await rig.obs.wait_for("status", after=t1, where=lambda d: d["text"] == "operator")
    await rig.hang_up()


async def test_operator_works_offline(rig, monkeypatch):
    monkeypatch.setattr(tts_module.ElevenLabsTTS, "URL", "wss://127.0.0.1:9/{voice_id}")
    t0 = time.monotonic()
    rig.hw.lift()
    await rig.wait_playing(after=t0, seconds=1.5, timeout=5)
    await rig.hw.dial(0)
    t1 = time.monotonic()
    await rig.wait_playing(after=t1, seconds=2, timeout=10)
