import asyncio
import functools

import pytest

from phone import switchboard as sb
from phone.config import Config
from phone.hardware import DialStart, Digit, OffHook, OnHook

from .fakes import FakeCache, FakeCall, FakeHardware, FakePlayer, FakeSounds, directory


@pytest.fixture(autouse=True)
def fast_timeouts(monkeypatch):
    monkeypatch.setattr(sb, "DIAL_WAIT_S", 0.05)
    monkeypatch.setattr(sb, "DIGIT_WAIT_S", 0.05)
    FakeCall.instances.clear()


def board(call=FakeCall, offline=False, greeting="dial please"):
    player = FakePlayer()
    deps = sb.Deps(mic=None, player=player, tts=None, cache=FakeCache(offline), sounds=FakeSounds())
    statuses = []
    b = sb.Switchboard(Config(None, None), directory(greeting), FakeHardware(), deps,
                       status=statuses.append, call_factory=call)
    b.start()
    return b, player, statuses


async def settle(seconds=0.02):
    await asyncio.sleep(seconds)


async def dial(b, digit):
    b.handle(DialStart())
    b.handle(Digit(digit))
    await settle()


async def test_dialing_a_role_starts_a_call_with_it():
    b, player, statuses = board()
    b.handle(OffHook())
    await settle()
    assert player.texts() == ["dial please"]
    await dial(b, 7)
    assert [c.role.name for c in FakeCall.instances] == ["God"]
    assert statuses[-1] == "in call: God"


async def test_zero_reads_the_directory_then_accepts_a_digit():
    b, player, _ = board()
    b.handle(OffHook())
    await settle()
    await dial(b, 0)
    assert player.texts()[-1] == "For The Elf, dial 1. For God, dial 7."
    await dial(b, 1)
    assert FakeCall.instances[-1].role.name == "The Elf"


async def test_unknown_number_plays_wrong_number_then_reorder():
    b, player, statuses = board()
    b.handle(OffHook())
    await settle()
    await dial(b, 4)
    await settle(0.05)
    assert "wrong number" in player.texts()
    assert statuses[-1] == "off hook: reorder"
    assert not FakeCall.instances


async def test_hanging_up_cancels_the_call_and_silences_audio():
    b, player, statuses = board()
    b.handle(OffHook())
    await settle()
    await dial(b, 7)
    flushes = player.flushes
    b.handle(OnHook())
    await b.wait_idle()
    assert FakeCall.instances[-1].cancelled
    assert player.flushes > flushes
    assert statuses[-1] == "idle"


async def test_a_failed_call_plays_the_busy_message_instead_of_crashing():
    b, player, statuses = board(call=functools.partial(FakeCall, fail=ConnectionError("API down")))
    b.handle(OffHook())
    await settle()
    await dial(b, 1)
    await settle(0.05)
    assert "all circuits busy" in player.texts()
    assert statuses[-1] == "off hook: reorder"


async def test_dialing_cuts_off_the_operator_prompt():
    b, player, _ = board(greeting="dial please " * 50)  # 'plays' for ~0.6 s
    b.handle(OffHook())
    await settle()
    b.handle(DialStart())
    await settle()
    assert not player.playing
    b.handle(Digit(1))
    await settle()
    assert FakeCall.instances


async def test_no_dialing_repeats_the_prompt_then_gives_up():
    b, player, statuses = board()
    b.handle(OffHook())
    await settle(0.4)
    assert player.texts().count("dial please") == sb.PROMPT_REPEATS
    assert statuses[-1] == "off hook: reorder"


async def test_operator_falls_back_to_dial_tone_when_offline():
    b, player, _ = board(offline=True)
    b.handle(OffHook())
    await settle()
    assert player.played[0] == FakeSounds.dial_tone
    await dial(b, 7)
    assert FakeCall.instances[-1].role.name == "God"


async def test_a_second_off_hook_during_a_session_is_ignored():
    b, _, _ = board()
    b.handle(OffHook())
    await settle()
    first = b._session
    b.handle(OffHook())
    assert b._session is first


async def test_call_ending_by_itself_leads_to_reorder_not_idle():
    b, player, statuses = board(call=functools.partial(FakeCall, duration=0))
    b.handle(OffHook())
    await settle()
    await dial(b, 1)
    await settle(0.05)
    assert statuses[-1] == "off hook: reorder"


async def test_a_paused_role_is_unlisted_and_gets_the_busy_message():
    import dataclasses
    b, player, statuses = board()
    b.directory.roles[7] = dataclasses.replace(b.directory.roles[7], paused=True)
    assert b.directory.listing() == "For The Elf, dial 1."
    b.handle(OffHook())
    await settle()
    await dial(b, 7)
    await settle(0.05)
    assert not FakeCall.instances
    assert "all circuits busy" in player.texts()
