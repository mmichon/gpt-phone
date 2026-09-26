"""Entry point: python -m phone [--no-gpio] [--role N] [--dial-test [--record FILE --expect DIGITS]]"""

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import urllib.request

from .audio import Mic, Player
from .config import REPO_DIR, Config
from .roles import load_directory
from .switchboard import Deps, Sounds, Switchboard
from .tts import ElevenLabsTTS, PromptCache

log = logging.getLogger("phone")

HEARTBEAT_S = 5
SERVICE_RETRY_S = 60


def sd_notify(message):
    """Tell systemd about readiness, liveness and status (no-op outside systemd)."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(message.encode())
    except OSError as e:
        log.debug("sd_notify failed: %s", e)


class Status:
    """What the phone is doing plus any degraded service, published as systemd STATUS=."""

    def __init__(self):
        self.state = "starting"
        self.problems = {}

    def set(self, state):
        self.state = state
        self._publish()

    def problem(self, key, text=None):
        if text:
            self.problems[key] = text
        else:
            self.problems.pop(key, None)
        self._publish()

    def _publish(self):
        text = self.state + "".join(f" | degraded: {p}" for p in self.problems.values())
        sd_notify(f"STATUS={text}")


def _elevenlabs_subscription(api_key):
    request = urllib.request.Request("https://api.elevenlabs.io/v1/user/subscription",
                                     headers={"xi-api-key": api_key or ""})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


async def check_services(cfg, status):
    """Validate API keys (keeps retrying while the network is down)."""
    while True:
        ok = True
        try:
            tier = (await asyncio.to_thread(_elevenlabs_subscription, cfg.elevenlabs_api_key)).get("tier")
            if tier == "free":
                log.error("ElevenLabs account is on the free tier; library voices will be refused")
            status.problem("elevenlabs", "elevenlabs free tier" if tier == "free" else None)
        except Exception as e:
            ok = False
            log.error("ElevenLabs check failed: %s", e)
            status.problem("elevenlabs", f"elevenlabs: {e}")
        try:
            from google import genai
            client = genai.Client(api_key=cfg.gemini_api_key)
            await asyncio.wait_for(client.aio.models.get(model=cfg.live_model), 10)
            status.problem("gemini")
        except Exception as e:
            ok = False
            log.error("Gemini check failed: %s", e)
            status.problem("gemini", f"gemini: {e}")
        if ok:
            return
        await asyncio.sleep(SERVICE_RETRY_S)


async def warm_cache(cache, directory, status):
    while missing := await cache.warm(directory.prompts()):
        status.problem("prompts", f"{missing} prompts not cached")
        await asyncio.sleep(SERVICE_RETRY_S)
    status.problem("prompts")
    log.info("All fixed prompts are cached")


async def heartbeat(mic, player, hardware):
    while True:
        if not (mic.healthy and player.healthy):
            log.error("Audio device lost; exiting so systemd restarts us")
            raise SystemExit(1)
        hardware.resync()
        sd_notify("WATCHDOG=1")
        await asyncio.sleep(HEARTBEAT_S)


async def serve(cfg, args):
    status = Status()
    directory = load_directory(cfg.roles_file)
    log.info("Loaded %d roles from %s", len(directory.roles), cfg.roles_file)

    if args.no_gpio:
        from .hardware import KeyboardHardware
        hardware = KeyboardHardware(auto_dial=args.role)
    else:
        from .hardware import GpioHardware
        hardware = GpioHardware(cfg.hook_gpio, cfg.dial_gpio)

    mic = Mic(cfg.mic_rate, cfg.mic_block_ms)
    player = Player(cfg.out_rate)
    mic.open()
    player.open()
    tts = ElevenLabsTTS(cfg.elevenlabs_api_key, cfg.tts_model, cfg.out_rate, cfg.tts_connect_timeout)
    cache = PromptCache(tts, cfg.cache_dir, cfg.out_rate)
    sounds = Sounds(cfg.sounds_dir, cfg.out_rate)
    for role in directory.roles.values():
        sounds.get(role.ringback)

    board = Switchboard(cfg, directory, hardware, Deps(mic, player, tts, cache, sounds), status=status.set)
    board.start()
    sd_notify("READY=1")
    log.info("Ready")
    try:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(heartbeat(mic, player, hardware))
            tasks.create_task(check_services(cfg, status))
            tasks.create_task(warm_cache(cache, directory, status))
    finally:
        sd_notify("STOPPING=1")
        hardware.close()
        mic.close()
        player.close()


async def dial_test(cfg, record, expect):
    from .hardware import Digit, GpioHardware
    hardware = GpioHardware(cfg.hook_gpio, cfg.dial_gpio, record_edges=True)
    digits = []

    def show(event):
        print(event, flush=True)
        if isinstance(event, Digit):
            digits.append(event.digit)

    hardware.start(asyncio.get_running_loop(), show)
    print("Dial away (Ctrl-C to stop)." + (f" Expecting: {expect}" if expect else ""), flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        decoded = "".join(map(str, digits))
        print(f"Decoded: {decoded}" + (f"  (expected {expect}: {'OK' if decoded == expect else 'MISMATCH'})"
                                       if expect else ""))
        if record:
            with open(record, "w") as f:
                json.dump({"expected": expect or decoded, "edges": hardware.edges}, f)
            print(f"Saved {len(hardware.edges)} edges to {record}")
        hardware.close()


def main():
    parser = argparse.ArgumentParser(prog="phone", description=__doc__)
    parser.add_argument("--no-gpio", action="store_true", help="use the keyboard instead of the hook and dial")
    parser.add_argument("--role", type=int, help="with --no-gpio: pick up and dial this digit at start")
    parser.add_argument("--dial-test", action="store_true", help="print decoded digits from the rotary dial")
    parser.add_argument("--record", help="with --dial-test: save raw dial edges to this JSON file")
    parser.add_argument("--expect", help="with --dial-test: the digits you're going to dial")
    parser.add_argument("--log-level", default=os.environ.get("PHONE_LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_DIR / ".env")
    except ImportError:
        pass
    under_systemd = "JOURNAL_STREAM" in os.environ  # journald adds its own timestamps
    logging.basicConfig(
        level=args.log_level.upper(), stream=sys.stderr,
        format="%(levelname)-7s %(name)s: %(message)s" if under_systemd
        else "%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("websockets", "httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = Config.from_env()
    try:
        if args.dial_test:
            asyncio.run(dial_test(cfg, args.record, args.expect))
        else:
            asyncio.run(serve(cfg, args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
