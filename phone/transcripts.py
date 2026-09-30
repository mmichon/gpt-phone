"""What was said on each call, kept on disk until it's been emailed (see digest.py).

Calls are appended to calls.jsonl, one JSON object per line. The file named `sent`
holds the byte offset up to which calls have gone out in a digest.
"""

import datetime
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class CallRecord:
    role_digit: int
    role_name: str
    started: str                  # ISO local time
    ended: str = ""
    lines: list = field(default_factory=list)   # [seconds into the call, "caller" | "character", text]
    _t0: float = field(default_factory=time.monotonic, repr=False, compare=False)

    @classmethod
    def begin(cls, role):
        return cls(role.digit, role.name, _now())

    def add(self, speaker, text):
        text = " ".join(text.split())
        if text:
            self.lines.append([round(time.monotonic() - self._t0, 1), speaker, text])

    def finish(self):
        self.ended = _now()

    @property
    def caller_spoke(self):
        return any(speaker == "caller" for _, speaker, _ in self.lines)

    @property
    def start_time(self):
        return datetime.datetime.fromisoformat(self.started)

    @property
    def seconds(self):
        if not self.ended:
            return 0
        return max(0, (datetime.datetime.fromisoformat(self.ended) - self.start_time).total_seconds())

    def to_json(self):
        data = asdict(self)
        data.pop("_t0")
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, line):
        return cls(**json.loads(line))


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class Journal:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.calls = self.dir / "calls.jsonl"
        self.marker = self.dir / "sent"

    def save(self, record):
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.calls.open("a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")

    def _sent_offset(self):
        try:
            return int(self.marker.read_text().strip() or 0)
        except (FileNotFoundError, ValueError):
            return 0

    def unsent(self):
        """The calls not yet emailed, and the offset to pass to mark_sent once they have been."""
        if not self.calls.exists():
            return [], 0
        records = []
        with self.calls.open("rb") as f:
            f.seek(self._sent_offset())
            for line in f:
                if line.strip():
                    try:
                        records.append(CallRecord.from_json(line.decode("utf-8")))
                    except (ValueError, TypeError) as e:
                        log.warning("Skipping an unreadable call record: %s", e)
            end = f.tell()
        return records, end

    def mark_sent(self, offset):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.marker.with_suffix(".tmp")
        tmp.write_text(str(offset))
        tmp.replace(self.marker)
