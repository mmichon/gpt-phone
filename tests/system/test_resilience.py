"""Resilience of the deployed service, driven from a dev machine over SSH:
    pytest -m system tests/system          (PHONE_HOST overrides pi@phone.local)

These disrupt the phone on purpose (kill, freeze, cut the network, unplug the
USB audio in software, reboot) and check it recovers on its own.
"""

import os
import subprocess
import time

import pytest

pytestmark = pytest.mark.system

HOST = os.environ.get("PHONE_HOST", "pi@phone.local")
UNIT = "gpt-phone.service"


def remote(command, timeout=60, check=True):
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", HOST, command],
                            capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"{command!r} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def service():
    out = remote(f"systemctl --user show {UNIT} -p ActiveState -p MainPID -p StatusText -p NRestarts")
    return dict(line.split("=", 1) for line in out.splitlines())


def wait_until(predicate, timeout, interval=1.0, what="condition"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = service()
            if predicate(last):
                return last
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}; last state: {last}")


def idle(state):
    return state.get("ActiveState") == "active" and state.get("StatusText", "").startswith("idle")


def healthy_idle(state):
    return idle(state) and "degraded" not in state.get("StatusText", "")


@pytest.fixture(autouse=True)
def service_is_up():
    wait_until(idle, 60, what="the service to be idle before the test")


def test_killed_process_is_restarted():
    pid = service()["MainPID"]
    remote(f"kill -9 {pid}")
    state = wait_until(lambda s: idle(s) and s["MainPID"] != pid, 30, what="a restart after kill -9")
    assert state["MainPID"] != pid


def test_hung_process_is_restarted_by_the_watchdog():
    pid = service()["MainPID"]
    remote(f"kill -STOP {pid}")
    try:
        wait_until(lambda s: idle(s) and s["MainPID"] != pid, 60, what="the watchdog to restart a frozen process")
    finally:
        remote(f"kill -CONT {pid}", check=False)


BLOCKED_HOSTS = "api.elevenlabs.io generativelanguage.googleapis.com"
RESTORE_HOSTS = "sudo cp /etc/hosts.phone-test /etc/hosts && sudo rm -f /etc/hosts.phone-test"


def test_starts_degraded_without_network_and_recovers():
    """Like booting before Wi-Fi is up: the service must come up anyway and heal itself.
    The APIs are made unreachable by pointing their hostnames at localhost (the Pi has
    no firewall tools); a timer restores /etc/hosts even if this test dies midway."""
    remote(f"sudo cp /etc/hosts /etc/hosts.phone-test && "
           f"echo '127.0.0.1 {BLOCKED_HOSTS}' | sudo tee -a /etc/hosts >/dev/null && "
           f"sudo systemd-run --quiet --on-active=240 --unit phone-test-unblock "
           f"sh -c 'test -f /etc/hosts.phone-test && cp /etc/hosts.phone-test /etc/hosts && rm /etc/hosts.phone-test'")
    try:
        remote(f"systemctl --user restart {UNIT}")
        state = wait_until(lambda s: idle(s) and "degraded" in s["StatusText"], 60,
                           what="the service to report degraded")
        restarts = state["NRestarts"]
    finally:
        remote(RESTORE_HOSTS, check=False)
        remote("sudo systemctl stop phone-test-unblock.timer", check=False)
    state = wait_until(healthy_idle, 150, interval=5, what="recovery once the network is back")
    assert state["NRestarts"] == restarts, "it should recover without restarting"


def test_recovers_from_usb_audio_loss():
    usb = remote("for d in /sys/bus/usb/devices/*; do "
                 "[ \"$(cat $d/product 2>/dev/null)\" = 'USB Audio Device' ] && basename $d; done | head -1")
    assert usb, "C-Media USB audio device not found"
    remote(f"echo {usb} | sudo tee /sys/bus/usb/drivers/usb/unbind >/dev/null")
    time.sleep(5)
    remote(f"echo {usb} | sudo tee /sys/bus/usb/drivers/usb/bind >/dev/null")
    wait_until(idle, 90, what="the service to be idle after the USB audio came back")
    time.sleep(5)
    streams = remote("pactl list short source-outputs; pactl list short sink-inputs")
    sources = remote("pactl list short sources")
    assert "C-Media" in sources or "usb" in sources.lower(), "the USB mic didn't come back"
    assert streams, "the phone isn't streaming audio after recovery"


@pytest.mark.parametrize("n", range(3))
def test_comes_up_after_reboot(n):
    remote("sudo systemctl reboot", check=False, timeout=15)
    time.sleep(20)
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        try:
            remote("true", timeout=15)
            break
        except (RuntimeError, subprocess.TimeoutExpired):
            time.sleep(5)
    else:
        raise AssertionError("the phone didn't come back after reboot")
    wait_until(idle, 90, what="the service to be idle after boot")
    log = remote("journalctl -b --no-pager 2>/dev/null | grep -c 'pipewire-0.lock' || true", check=False)
    assert log.strip() in ("", "0"), "PipeWire lock conflicts at boot (a second PipeWire is being started)"
