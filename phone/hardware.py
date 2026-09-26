"""Hook switch and rotary dial: GPIO on the Pi, the keyboard on a Mac.

Hardware delivers events to a callback on the asyncio loop thread:
OffHook, OnHook, DialStart (the dial left rest, so stop any prompt), Digit.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OffHook:
    pass


@dataclass(frozen=True)
class OnHook:
    pass


@dataclass(frozen=True)
class DialStart:
    pass


@dataclass(frozen=True)
class Digit:
    digit: int


class DialDecoder:
    """Turns rotary-dial contact edges into digits.

    The dial line idles low. While the dial winds back it is high, dropping low
    for about 60 ms per pulse, and it goes low for good when the dial comes to
    rest. So each short low period between two highs is one pulse, and a low
    period longer than PULSE_MAX ends the digit. Ten pulses dial 0.
    """

    PULSE_MIN = 0.015  # shorter lows are contact bounce
    PULSE_MAX = 0.1    # longer lows end the digit (same threshold as the legacy read_dial)

    def __init__(self):
        self.level = 0
        self.dialing = False
        self.pulses = 0
        self.fell_at = None

    def edge(self, t, level):
        """Feed one contact edge at monotonic time t. Returns a list of events."""
        events = []
        if level == self.level:
            return events
        self.level = level
        if level:
            if self.dialing and self.fell_at is not None and t - self.fell_at >= self.PULSE_MAX:
                events += self._finish()  # missed the poll: previous digit ended at rest
            if not self.dialing:
                self.dialing = True
                self.pulses = 0
            elif self.fell_at is not None and t - self.fell_at >= self.PULSE_MIN:
                self.pulses += 1
                if self.pulses == 1:
                    events.append(DialStart())  # on a real pulse, so glitches can't cut prompts off
            self.fell_at = None
        else:
            self.fell_at = t
        return events

    def poll(self, now):
        """Call at least PULSE_MAX after the last falling edge to complete a digit."""
        if self.dialing and self.level == 0 and self.fell_at is not None \
                and now - self.fell_at >= self.PULSE_MAX:
            return self._finish()
        return []

    def _finish(self):
        pulses, self.pulses, self.dialing = self.pulses, 0, False
        if not 1 <= pulses <= 10:
            log.warning("Ignoring dial with %d pulses", pulses)
            return []
        return [Digit(0 if pulses == 10 else pulses)]


class GpioHardware:
    """The real phone: hook switch and dial wired to GPIO inputs with pull-downs."""

    HOOK_DEBOUNCE = 0.05

    def __init__(self, hook_pin, dial_pin, record_edges=False):
        from gpiozero import Button  # Pi-only dependency

        self._hook = Button(hook_pin, pull_up=False)
        self._dial = Button(dial_pin, pull_up=False)
        self._decoder = DialDecoder()
        self._hook_check = None
        self.edges = [] if record_edges else None

    @property
    def off_hook(self):
        return self._hook_state

    def start(self, loop, emit):
        self._loop = loop
        self._emit = emit
        self._hook_state = bool(self._hook.value)
        self._dial.when_activated = lambda: self._from_thread(self._on_dial_edge, time.monotonic(), 1)
        self._dial.when_deactivated = lambda: self._from_thread(self._on_dial_edge, time.monotonic(), 0)
        self._hook.when_activated = lambda: self._from_thread(self._on_hook_edge)
        self._hook.when_deactivated = lambda: self._from_thread(self._on_hook_edge)
        if self._hook_state:
            emit(OffHook())

    def resync(self):
        """Re-read the hook in case an edge was missed. Called periodically."""
        self._check_hook()

    def close(self):
        self._hook.close()
        self._dial.close()

    def _from_thread(self, fn, *args):
        self._loop.call_soon_threadsafe(fn, *args)

    def _on_dial_edge(self, t, level):
        if self.edges is not None:
            self.edges.append((round(t, 4), level))
        for event in self._decoder.edge(t, level):
            self._emit(event)
        if not level:
            self._loop.call_later(DialDecoder.PULSE_MAX + 0.01, self._poll_dial)

    def _poll_dial(self):
        for event in self._decoder.poll(time.monotonic()):
            self._emit(event)

    def _on_hook_edge(self):
        if self._hook_check:
            self._hook_check.cancel()
        self._hook_check = self._loop.call_later(self.HOOK_DEBOUNCE, self._check_hook)

    def _check_hook(self):
        state = bool(self._hook.value)
        if state != self._hook_state:
            self._hook_state = state
            self._emit(OffHook() if state else OnHook())

    def dump_edges(self, path):
        with open(path, "w") as f:
            json.dump(self.edges, f)


class KeyboardHardware:
    """Mac development stand-in: Enter toggles the hook, a digit then Enter dials it."""

    def __init__(self, auto_dial=None):
        self._auto_dial = auto_dial
        self._off_hook = False

    @property
    def off_hook(self):
        return self._off_hook

    def start(self, loop, emit):
        self._emit = emit
        loop.add_reader(sys.stdin, self._on_line)
        print("Keyboard phone: Enter = pick up / hang up, digit + Enter = dial.", file=sys.stderr)
        if self._auto_dial is not None:
            self._toggle()
            loop.call_later(0.5, self._dial, self._auto_dial)

    def resync(self):
        pass

    def close(self):
        asyncio.get_event_loop().remove_reader(sys.stdin)

    def _on_line(self):
        line = sys.stdin.readline().strip()
        if line.isdigit() and len(line) == 1:
            self._dial(int(line))
        elif line == "":
            self._toggle()

    def _toggle(self):
        self._off_hook = not self._off_hook
        print("[off hook]" if self._off_hook else "[on hook]", file=sys.stderr)
        self._emit(OffHook() if self._off_hook else OnHook())

    def _dial(self, digit):
        if self._off_hook:
            self._emit(DialStart())
            self._emit(Digit(digit))
