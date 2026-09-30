"""Call transcripts: kept on disk until emailed, and rendered as the daily email."""

import datetime

from phone import digest
from phone.transcripts import CallRecord, Journal


def record(digit, name, started, seconds, lines):
    start = datetime.datetime.fromisoformat(started)
    end = (start + datetime.timedelta(seconds=seconds)).isoformat(timespec="seconds")
    return CallRecord(digit, name, started, end, [[0, s, t] for s, t in lines])


SLOTH = record(3, "Storytime Sloth", "2026-10-03T21:40:00", 190, [
    ("character", "Hellooo, it's the sloth."),
    ("caller", "Tell me a story"), ("caller", "about a dragon."),
    ("character", "Once upon a time..."),
])
ELF = record(1, "Elf", "2026-10-04T00:15:00", 45, [
    ("character", "Merry greetings!"), ("caller", "<script>alert(1)</script> hi elf"),
])


def test_journal_keeps_calls_until_sent(tmp_path):
    journal = Journal(tmp_path)
    journal.save(SLOTH)
    records, offset = Journal(tmp_path).unsent()   # survives a restart
    assert [r.role_name for r in records] == ["Storytime Sloth"]
    assert records[0].lines == SLOTH.lines
    journal.mark_sent(offset)
    assert journal.unsent()[0] == []
    journal.save(ELF)
    assert [r.role_name for r in Journal(tmp_path).unsent()[0]] == ["Elf"]


def test_empty_journal_has_nothing_to_send(tmp_path):
    assert Journal(tmp_path / "none").unsent() == ([], 0)


def test_digest_summarizes_and_transcribes():
    subject, html, text = digest.build([ELF, SLOTH], ["Elf gets hacked", "A dragon, slowly"])
    assert subject == "📞 2 calls on the rotary phone · Sat, Oct 3"
    assert "3 min 55 s" in html                      # total talking
    assert html.index("A dragon, slowly") < html.index("Elf gets hacked")  # in call order
    assert "Tell me a story about a dragon." in html  # one bubble per turn
    assert "<script>" not in html and "&lt;script&gt;" in html
    for line in ["Caller: Tell me a story about a dragon.", "Storytime Sloth: Once upon a time...",
                 "Elf: Merry greetings!"]:
        assert line in text


def test_fallback_title_is_the_callers_first_words():
    assert digest.fallback_title(SLOTH) == "“Tell me a story”"


def test_next_run_is_the_coming_morning():
    at = datetime.datetime(2026, 10, 3, 23, 30)
    assert digest.next_run(at, 9) == datetime.datetime(2026, 10, 4, 9)
    assert digest.next_run(at.replace(hour=8), 9) == datetime.datetime(2026, 10, 3, 9)
