"""Characters reachable by dialing a digit, plus the operator, loaded from roles.yaml."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PREAMBLE = """\
You are a character on a phone call. The caller is talking to you through a \
vintage rotary telephone, and everything you write is spoken aloud by a \
text-to-speech voice. So:
- Reply in one to three short, natural spoken sentences, unless the caller asks for a story.
- Write plain spoken words only: no emoji, markdown, lists, or stage directions.
- The caller's words come through a crackly phone line. If what they said seems \
garbled, cut off, or makes no sense in context, say you didn't catch that and \
ask them to repeat it, instead of guessing.
- The caller may interrupt you. If they do, respond to what they just said.
- The caller may be a young child. End every reply by clearly handing the turn \
back, with a question or an invitation like "Your turn!" or "What do you think?", \
so they never wonder whether you're still talking.
- In a game, when a round ends, say how it went and ask if they want to play again.
- Stay in character.
Today is {today}.

You already picked up the phone and said: "{greeting}"

Your character:
{persona}"""

DEFAULT_STILL_THERE = "Hello? Are you still there?"


@dataclass(frozen=True)
class Role:
    digit: int
    name: str          # spoken in the directory: "For <name>, dial <digit>."
    voice_id: str      # ElevenLabs voice
    greeting: str
    persona: str
    ringback: str = "dialtone.mp3"
    language: str = "en"
    still_there: str = DEFAULT_STILL_THERE
    tts_model: str | None = None
    paused: bool = False  # left out of the directory; dialing it gets the busy message

    def system_instruction(self, now=None):
        now = now or datetime.datetime.now()
        today = now.strftime("%A, %B %-d, %Y")
        return PREAMBLE.format(today=today, greeting=self.greeting, persona=self.persona.strip())


@dataclass(frozen=True)
class Operator:
    voice_id: str
    greeting: str = "Please dial a single digit to proceed. For a directory, please dial zero."
    wrong_number: str = "That number is disconnected. Please hang up and try again."
    busy: str = "We're sorry, all circuits are busy now. Please hang up and try your call again later."
    tts_model: str | None = None


@dataclass(frozen=True)
class Directory:
    operator: Operator
    roles: dict[int, Role] = field(default_factory=dict)

    def listing(self):
        """The spoken directory, e.g. 'For God, dial 7.'"""
        return " ".join(f"For {r.name}, dial {d}." for d, r in sorted(self.roles.items()) if not r.paused)

    def prompts(self):
        """Every fixed (voice_id, tts_model, text) phrase worth pre-rendering."""
        op = self.operator
        yield op.voice_id, op.tts_model, op.greeting
        yield op.voice_id, op.tts_model, op.wrong_number
        yield op.voice_id, op.tts_model, op.busy
        yield op.voice_id, op.tts_model, self.listing()
        for role in self.roles.values():
            if role.paused:
                continue
            yield role.voice_id, role.tts_model, role.greeting
            yield role.voice_id, role.tts_model, role.still_there


def load_directory(path):
    data = yaml.safe_load(Path(path).read_text())
    operator = Operator(**data["operator"])
    roles = {}
    for digit, spec in (data.get("roles") or {}).items():
        digit = int(digit)
        if not 1 <= digit <= 9:
            raise ValueError(f"{path}: role digit must be 1-9, got {digit}")
        roles[digit] = Role(digit=digit, **spec)
    return Directory(operator=operator, roles=roles)
