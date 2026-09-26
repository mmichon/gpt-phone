import asyncio
import dataclasses
import datetime
import json
import os
import time
from pathlib import Path

import pytest
import pytest_asyncio

from phone.audio import Player
from phone.config import Config
from phone.roles import load_directory
from phone.switchboard import Deps, Sounds, Switchboard
from phone.tts import ElevenLabsTTS, PromptCache

from .harness import MixMic, Observer, OutputTap, ScriptedHardware, load_line

REPORTS = Path(__file__).parent / "reports"


def pytest_collection_modifyitems(config, items):
    if not (os.environ.get("GEMINI_API_KEY") and os.environ.get("ELEVENLABS_API_KEY")):
        skip = pytest.mark.skip(reason="needs GEMINI_API_KEY and ELEVENLABS_API_KEY (run via deploy/deploy.sh --test)")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def cfg():
    # The rig handles "still there?" prompts itself, so tests can sit in silence.
    return dataclasses.replace(Config.from_env(), still_there_s=600, give_up_s=900)


@pytest.fixture(scope="session")
def directory(cfg):
    return load_directory(cfg.roles_file)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def devices(cfg, directory):
    """Real audio devices and TTS, shared by all tests; fixed prompts pre-rendered."""
    mic = MixMic(cfg.mic_rate, cfg.mic_block_ms)
    tap = OutputTap(cfg.out_rate)
    player = Player(cfg.out_rate, tap=tap)
    mic.open()
    player.open()
    tts = ElevenLabsTTS(cfg.elevenlabs_api_key, cfg.tts_model, cfg.out_rate, cfg.tts_connect_timeout)
    cache = PromptCache(tts, cfg.cache_dir, cfg.out_rate)
    missing = await cache.warm(directory.prompts())
    assert missing == 0, "couldn't pre-render prompts; is ElevenLabs reachable and paid?"
    yield mic, player, tap, tts, cache, Sounds(cfg.sounds_dir, cfg.out_rate)
    mic.close()
    player.close()


class Rig:
    def __init__(self, cfg, directory, devices):
        self.cfg = cfg
        self.directory = directory
        self.mic, self.player, self.tap, self.tts, self.cache, self.sounds = devices
        self.hw = ScriptedHardware()
        self.obs = Observer()
        self.board = Switchboard(cfg, directory, self.hw,
                                 Deps(self.mic, self.player, self.tts, self.cache, self.sounds),
                                 observe=self.obs)
        self.board.start()

    async def connect_to(self, digit):
        """Lift, dial, and wait until the character's greeting has finished."""
        t0 = time.monotonic()
        self.hw.lift()
        await self.obs.wait_for("status", after=t0, where=lambda d: d["text"] == "operator")
        await asyncio.sleep(0.5)
        await self.hw.dial(digit)
        await self.obs.wait_for("call_start", after=t0, timeout=10)
        await self.wait_quiet(after=t0)
        return t0

    async def say(self, *parts):
        """The caller speaks: line names, with numbers as pauses in seconds. Returns (start, end)."""
        pcm = b""
        for part in parts:
            pcm += (b"\x00\x00" * int(part * self.cfg.mic_rate) if isinstance(part, (int, float))
                    else load_line(part, self.cfg.mic_rate))
        self.mic.say(pcm, "+".join(p for p in parts if isinstance(p, str)))
        await asyncio.sleep(len(pcm) / 2 / self.cfg.mic_rate)
        while self.mic.talking or self.mic.spans[-1][1] is None:
            await asyncio.sleep(0.02)
        start, end, _ = self.mic.spans[-1]
        return start, end

    async def wait_quiet(self, after, quiet=1.0, timeout=60.0):
        """Wait until the phone has been silent for `quiet` seconds."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            last = max(self.tap.times(after), default=after)
            if not self.player.playing and time.monotonic() - last >= quiet:
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("the phone never went quiet")

    async def wait_playing(self, after, seconds, timeout=30.0):
        """Wait until at least `seconds` of audio has played since `after`."""
        deadline = time.monotonic() + timeout
        while self.tap.seconds(after) < seconds:
            if time.monotonic() > deadline:
                raise TimeoutError(f"less than {seconds}s of audio played")
            await asyncio.sleep(0.05)

    async def hang_up(self):
        self.hw.hang_up()
        await self.board.wait_idle()


@pytest_asyncio.fixture(loop_scope="session")
async def rig(cfg, directory, devices):
    r = Rig(cfg, directory, devices)
    yield r
    if r.hw.off_hook:
        await r.hang_up()
    await asyncio.sleep(0.5)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, test, **data):
        self.rows.append({"test": test, **data})


@pytest.fixture(scope="session")
def report():
    r = Report()
    yield r
    if not r.rows:
        return
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    (REPORTS / f"{stamp}.json").write_text(json.dumps(r.rows, indent=2, ensure_ascii=False))
    lines = [f"# E2E report {stamp}", "", "| test | turn | caller end → first audio (ms) | "
             "server eos → text / TTS / playing (ms) | heard | reply |", "|---|---|---|---|---|---|"]
    for row in r.rows:
        inner = row.get("internal") or {}
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            row["test"], row.get("turn", ""), row.get("latency_ms", ""),
            " / ".join(str(inner.get(k, "")) for k in ("text", "tts", "play")) if inner else "",
            row.get("heard", "").replace("|", "/")[:80], row.get("reply", "").replace("|", "/")[:120]))
    (REPORTS / f"{stamp}.md").write_text("\n".join(lines) + "\n")
