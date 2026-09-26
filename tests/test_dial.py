import json
from pathlib import Path

import pytest

from phone.hardware import DialDecoder, DialStart, Digit

FIXTURES = Path(__file__).parent / "fixtures" / "dial"


def pulse_train(pulses, start=0.0, make=0.04, brk=0.06, bounce=False):
    """Edges for one dialed digit: high while winding back, a ~60 ms low per pulse, low at rest."""
    edges, t = [(start, 1)], start + 0.05
    for _ in range(pulses):
        edges.append((t, 0))
        if bounce:  # contact chatter right after the break
            edges += [(t + 0.002, 1), (t + 0.003, 0)]
        t += brk
        edges.append((t, 1))
        t += make
    edges.append((t, 0))
    return edges, t


def decode(edges, settle=0.5):
    decoder, events = DialDecoder(), []
    for t, level in edges:
        events += decoder.edge(t, level)
    events += decoder.poll(edges[-1][0] + settle)
    return events


@pytest.mark.parametrize("pulses,digit", [(n, n % 10) for n in range(1, 11)])
def test_each_digit(pulses, digit):
    edges, _ = pulse_train(pulses)
    assert decode(edges) == [DialStart(), Digit(digit)]


def test_contact_bounce_is_ignored():
    edges, _ = pulse_train(7, bounce=True)
    assert decode(edges)[-1] == Digit(7)


def test_glitch_at_rest_is_not_a_digit():
    assert decode([(0.0, 1), (0.004, 0)]) == []


def test_digit_completes_only_after_the_dial_rests():
    edges, end = pulse_train(3)
    decoder = DialDecoder()
    for t, level in edges:
        decoder.edge(t, level)
    assert decoder.poll(end + 0.05) == []
    assert decoder.poll(end + 0.11) == [Digit(3)]


def test_back_to_back_digits_without_a_poll():
    first, end = pulse_train(4)
    second, _ = pulse_train(2, start=end + 0.3)
    digits = [e for e in decode(first + second) if isinstance(e, Digit)]
    assert digits == [Digit(4), Digit(2)]


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_recorded_dialing(path):
    """Edges recorded from the real dial with `python -m phone --dial-test --record`."""
    data = json.loads(path.read_text())
    decoder, digits = DialDecoder(), []
    for t, level in data["edges"]:
        digits += [e.digit for e in decoder.poll(t) + decoder.edge(t, level) if isinstance(e, Digit)]
    digits += [e.digit for e in decoder.poll(float("inf")) if isinstance(e, Digit)]
    assert "".join(map(str, digits)) == data["expected"]
