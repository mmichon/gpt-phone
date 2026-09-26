"""Spike: how much of the earpiece leaks into the mouthpiece, with and without
PipeWire's echo canceller? Leave the handset on the hook (or lying still) and
stay quiet while it runs.

    python spikes/echo_test.py
"""

import subprocess
import time

import numpy as np
import sounddevice as sd

RATE = 24000
PAIRS = {
    "raw": ("alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo",
            "alsa_input.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.mono-fallback"),
    "echo-cancelled": ("phone_aec_sink", "phone_aec_source"),
}


def db(x):
    return 20 * np.log10(max(x, 1e-9))


def measure(seconds=4.0, level=0.3):
    noise = np.random.default_rng(0).standard_normal(int(RATE * seconds)).astype(np.float32)
    noise = np.convolve(noise, np.ones(8) / 8, mode="same")  # soften toward speech-band noise
    noise *= level / np.abs(noise).max()
    quiet = sd.rec(int(RATE * 2), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    recorded = sd.playrec(noise, samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    return db(rms(noise)), db(rms(quiet[RATE // 2:])), db(rms(recorded[RATE // 2:]))


def main():
    for name, (sink, source) in PAIRS.items():
        if subprocess.run(["pactl", "set-default-sink", sink]).returncode or \
                subprocess.run(["pactl", "set-default-source", source]).returncode:
            print(f"{name}: nodes not found, skipping")
            continue
        time.sleep(1)
        sd._terminate()
        sd._initialize()
        played, floor, echo = measure()
        print(f"{name:15} played {played:6.1f} dBFS   mic floor {floor:6.1f} dBFS   "
              f"mic during playback {echo:6.1f} dBFS   echo above floor {echo - floor:5.1f} dB")

    subprocess.run(["pactl", "set-default-sink", "phone_aec_sink"])
    subprocess.run(["pactl", "set-default-source", "phone_aec_source"])


if __name__ == "__main__":
    main()
