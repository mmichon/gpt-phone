"""Local voice activity detection: decides when the caller starts and stops talking.

Silero VAD scores each 32 ms of (echo-cancelled, noise-suppressed) mic audio.
Speech starts after a run of confident frames and ends after a stretch of
silence, so breaths, clicks and short pauses don't flip it. A pre-roll buffer
means the first syllable, which happens before speech is confirmed, is kept.
A blip that looked a bit like speech but never started a stretch is reported
as IGNORED, so missed words show up in the logs.
"""

import collections

from pysilero_vad import SileroVoiceActivityDetector

START, AUDIO, END, IGNORED = "start", "audio", "end", "ignored"
NEAR_MISS = 0.3  # a blip scoring this high but not starting speech is reported


class SpeechDetector:
    def __init__(self, rate=16000, threshold=0.5, start_ms=64, end_ms=500, preroll_ms=320):
        assert rate == 16000, "Silero VAD runs at 16 kHz"
        self._vad = SileroVoiceActivityDetector()
        self.chunk = self._vad.chunk_bytes()            # 512 samples = 32 ms
        self.chunk_ms = chunk_ms = self.chunk // 2 * 1000 // rate
        self.threshold = threshold
        self.start_chunks = max(1, start_ms // chunk_ms)
        self.end_chunks = max(1, end_ms // chunk_ms)
        self._preroll = collections.deque(maxlen=max(1, preroll_ms // chunk_ms))
        self._leftover = b""
        self.speaking = False
        self._run = 0  # consecutive chunks disagreeing with the current state
        self.last_probability = 0.0
        self._chunks = 0      # length of the current stretch, or of the current blip
        self._peak = 0.0      # ...and its highest speech probability
        self.last_stretch = (0.0, 0.0)  # (seconds, peak) of the last stretch or ignored blip

    def reset(self):
        self._vad.reset()
        self._preroll.clear()
        self._leftover = b""
        self.speaking = False
        self._run = 0
        self._chunks, self._peak = 0, 0.0

    def feed(self, pcm):
        """Feed mic audio. Returns a list of (START | AUDIO | END, bytes) actions:
        audio to forward is only produced while the caller is speaking."""
        actions = []
        data = self._leftover + pcm
        cut = len(data) - len(data) % self.chunk
        self._leftover = data[cut:]
        for i in range(0, cut, self.chunk):
            chunk = data[i:i + self.chunk]
            self.last_probability = p = self._vad(chunk)
            if not self.speaking:
                self._preroll.append(chunk)
                self._run = self._run + 1 if p >= self.threshold else 0
                if p >= NEAR_MISS:
                    self._chunks, self._peak = self._chunks + 1, max(self._peak, p)
                elif self._chunks:
                    self.last_stretch = (self._chunks * self.chunk_ms / 1000, self._peak)
                    self._chunks, self._peak = 0, 0.0
                    actions.append((IGNORED, b""))
                if self._run >= self.start_chunks:
                    self.speaking, self._run = True, 0
                    self._chunks, self._peak = len(self._preroll), p
                    actions += [(START, b""), (AUDIO, b"".join(self._preroll))]
                    self._preroll.clear()
            else:
                actions.append((AUDIO, chunk))
                self._chunks, self._peak = self._chunks + 1, max(self._peak, p)
                # A little hysteresis: only clearly non-speech frames count toward the end.
                self._run = self._run + 1 if p < self.threshold * 0.7 else 0
                if self._run >= self.end_chunks:
                    self.speaking, self._run = False, 0
                    speech = self._chunks - self.end_chunks  # without the trailing silence
                    self.last_stretch = (speech * self.chunk_ms / 1000, self._peak)
                    self._chunks, self._peak = 0, 0.0
                    actions.append((END, b""))
        return actions
