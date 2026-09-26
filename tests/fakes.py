"""In-memory stand-ins for hardware, audio and TTS, for fast switchboard tests."""

import asyncio

from phone.roles import Directory, Operator, Role


class FakeHardware:
    off_hook = False

    def start(self, loop, emit):
        self.emit = emit

    def resync(self):
        pass

    def close(self):
        pass


class FakePlayer:
    """Plays 'audio' where each byte lasts a millisecond."""

    def __init__(self):
        self.played = []
        self.flushes = 0
        self.playing = False
        self.on_start = None

    def write(self, pcm):
        self.played.append(pcm)

    def flush(self):
        self.flushes += 1

    async def wait_drained(self):
        pass

    async def play(self, pcm):
        self.played.append(pcm)
        self.playing = True
        try:
            await asyncio.sleep(len(pcm) / 1000)
        finally:
            self.playing = False

    def texts(self):
        return [p.decode() for p in self.played if not p.startswith(b"\0")]


class FakeCache:
    """'Renders' a phrase as its own UTF-8 bytes, so tests can see what was said."""

    def __init__(self, offline=False):
        self.offline = offline

    def cached(self, voice_id, model, text):
        return None if self.offline else text.encode()

    async def get(self, voice_id, model, text, language=None):
        if self.offline:
            raise ConnectionError("offline")
        return text.encode()


class FakeSounds:
    reorder = b"\0" * 20
    dial_tone = b"\0" * 20

    def get(self, name):
        return b"\0" * 5


class FakeCall:
    """Records the role it was created for; runs until told to finish, or fails."""

    instances = []

    def __init__(self, cfg, role, deps, observe=None, fail=None, duration=None):
        self.role = role
        self.fail = fail
        self.duration = duration
        self.cancelled = False
        FakeCall.instances.append(self)

    async def run(self):
        try:
            if self.fail:
                raise self.fail
            await asyncio.sleep(self.duration if self.duration is not None else 3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def directory(greeting="dial please"):
    return Directory(
        operator=Operator(voice_id="op", greeting=greeting, wrong_number="wrong number",
                          busy="all circuits busy"),
        roles={
            1: Role(digit=1, name="The Elf", voice_id="v1", greeting="hi from elf", persona="an elf"),
            7: Role(digit=7, name="God", voice_id="v7", greeting="hi from god", persona="god"),
        })
