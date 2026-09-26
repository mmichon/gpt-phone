"""Local voice activity detection, with the real Silero model, on the caller recordings."""

import numpy as np
import pytest

from phone.vad import AUDIO, END, IGNORED, START, SpeechDetector

from .e2e.harness import load_line

RATE = 16000


def silence(seconds):
    return b"\0\0" * int(seconds * RATE)


def hiss(seconds, rms):
    return (np.random.default_rng(0).standard_normal(int(seconds * RATE)) * rms).astype(np.int16).tobytes()


def trimmed(line):
    """A recording without its leading and trailing silence."""
    samples = np.frombuffer(load_line(line), dtype=np.int16)
    loud = np.flatnonzero(np.abs(samples) > 500)
    return samples[loud[0]:loud[-1] + 1].tobytes()


def segments(pcm, block=1024, **kwargs):
    """Feed audio in mic-sized blocks. Returns [(start_s, end_s, bytes_forwarded)]."""
    detector, found, forwarded = SpeechDetector(**kwargs), [], 0
    for i in range(0, len(pcm), block):
        for action, data in detector.feed(pcm[i:i + block]):
            t = (i + block) / 2 / RATE
            if action == START:
                found.append([t, None, 0])
            elif action == AUDIO:
                found[-1][2] += len(data)
            elif action == END:
                found[-1][1] = t
    return [tuple(s) for s in found]


@pytest.mark.parametrize("line", ["q2", "long_1", "long_2"])
def test_a_spoken_line_is_one_segment_with_nothing_clipped(line):
    pcm = load_line(line)
    found = segments(pcm + silence(1))
    assert len(found) == 1, found
    start, end, forwarded = found[0]
    assert end is not None
    # Everything from just before speech was confirmed up to the end is forwarded.
    assert forwarded >= len(pcm) - int(0.6 * RATE) * 2


def test_a_long_pause_splits_speech_and_a_short_one_does_not():
    two = segments(trimmed("long_1") + silence(1.2) + trimmed("long_2") + silence(1))
    one = segments(trimmed("long_1") + silence(0.3) + trimmed("long_2") + silence(1))
    assert len(two) == 2 and len(one) == 1


@pytest.mark.parametrize("noise", ["silence", "suppressed hiss", "raw hiss", "cough"])
def test_noise_is_not_speech(noise):
    pcm = {"silence": silence(5), "suppressed hiss": hiss(5, 9), "raw hiss": hiss(5, 146),
           "cough": load_line("cough")}[noise]
    assert segments(pcm) == []


def test_the_first_syllable_is_kept():
    pcm = silence(0.5) + load_line("q1")
    (start, _, _), = segments(pcm)
    detector, out = SpeechDetector(), b""
    for i in range(0, len(pcm), 1024):
        out += b"".join(d for a, d in detector.feed(pcm[i:i + 1024]) if a == AUDIO)
    speech_onset = next(i for i in range(0, len(pcm), 2)
                        if abs(int.from_bytes(pcm[i:i + 2], "little", signed=True)) > 1000)
    assert len(out) >= len(pcm) - speech_onset - 2 * int(0.1 * RATE)


def test_a_stretch_reports_its_length_and_peak():
    detector = SpeechDetector()
    pcm = trimmed("q2") + silence(1)
    actions = [a for i in range(0, len(pcm), 1024) for a, _ in detector.feed(pcm[i:i + 1024])]
    assert END in actions
    seconds, peak, db = detector.last_stretch
    assert abs(seconds - len(trimmed("q2")) / 2 / RATE) < 0.4 and peak > 0.5 and -40 < db <= 0


def test_a_blip_too_short_to_start_speech_is_reported():
    detector = SpeechDetector(start_ms=10_000)  # nothing is long enough to start
    pcm = trimmed("q2") + silence(1)
    actions = [a for i in range(0, len(pcm), 1024) for a, _ in detector.feed(pcm[i:i + 1024])]
    assert START not in actions and IGNORED in actions
    assert detector.last_stretch[1] >= 0.3
