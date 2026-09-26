import numpy as np

from phone import audio


def test_scale_lowers_the_volume():
    pcm = np.array([1000, -1000, 32767, -32768], dtype=np.int16).tobytes()
    out = np.frombuffer(audio.scale(pcm, 0.8), dtype=np.int16)
    assert out.tolist() == [800, -800, 26213, -26214]


def test_scale_at_full_volume_is_a_no_op():
    pcm = b"\x01\x02\x03\x04"
    assert audio.scale(pcm, 1.0) is pcm
