"""Local voice activity detection: decides when the caller starts and stops talking.

Silero VAD scores each 32 ms of (echo-cancelled, noise-suppressed) mic audio.
Speech starts after a run of confident frames and ends after a stretch of
silence, so breaths, clicks and short pauses don't flip it. A pre-roll buffer
means the first syllable, which happens before speech is confirmed, is kept.
"""

import collections

from pysilero_vad import SileroVoiceActivityDetector

START, AUDIO, END = "start", "audio", "end"


class SpeechDetector:
    def __init__(self, rate=16000, threshold=0.5, start_ms=96, end_ms=500, preroll_ms=320):
        assert rate == 16000, "Silero VAD runs at 16 kHz"
        self._vad = SileroVoiceActivityDetector()
        self.chunk = self._vad.chunk_bytes()            # 512 samples = 32 ms
        chunk_ms = self.chunk // 2 * 1000 // rate
        self.threshold = threshold
        self.start_chunks = max(1, start_ms // chunk_ms)
        self.end_chunks = max(1, end_ms // chunk_ms)
        self._preroll = collections.deque(maxlen=max(1, preroll_ms // chunk_ms))
        self._leftover = b""
        self.speaking = False
        self._run = 0  # consecutive chunks disagreeing with the current state
        self.last_probability = 0.0

    def reset(self):
        self._vad.reset()
        self._preroll.clear()
        self._leftover = b""
        self.speaking = False
        self._run = 0

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
                if self._run >= self.start_chunks:
                    self.speaking, self._run = True, 0
                    actions += [(START, b""), (AUDIO, b"".join(self._preroll))]
                    self._preroll.clear()
            else:
                actions.append((AUDIO, chunk))
                # A little hysteresis: only clearly non-speech frames count toward the end.
                self._run = self._run + 1 if p < self.threshold * 0.7 else 0
                if self._run >= self.end_chunks:
                    self.speaking, self._run = False, 0
                    actions.append((END, b""))
        return actions
