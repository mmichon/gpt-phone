"""The daily email: every call since the last one, as a readable transcript.

build() renders it (HTML with a plain-text twin), titles() asks Gemini for a short
headline per call, and send_digest() does the lot and marks the calls as sent.
Email clients ignore <style> blocks and most modern CSS, so styles are inline.
"""

import asyncio
import collections
import datetime
import html
import logging
import smtplib
from email.message import EmailMessage
from email.utils import formataddr

log = logging.getLogger(__name__)

TITLE_TIMEOUT_S = 15
TITLE_PROMPT = """\
Here is a transcript of a call on a novelty rotary phone, where guests at a party talk \
to an AI character ("{name}"). Write a playful 4 to 8 word title for the call, like a \
newspaper headline, about what the caller did or asked. Reply with the title only.

{transcript}"""

# Colors: a cream page, bakelite black and red, brass.
PAGE = "#f3ebdd"
CARD = "#fffaf1"
INK = "#2b2320"
MUTED = "#8a7b6c"
RULE = "#e6dac7"
RED = "#9c2b23"
CALLER_BUBBLE = "#e8dcc5"
SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
DIGIT_COLORS = ["#6b4f3a", "#2f6f5e", "#9c2b23", "#3d5a8a", "#b0762a",
                "#7a4a86", "#4f7a2f", "#a8456d", "#2f6f86", "#8a5a2f"]


def _color(digit):
    return DIGIT_COLORS[digit % len(DIGIT_COLORS)]


def _duration(seconds):
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds} s" if seconds else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def _clock(t):
    return t.strftime("%-I:%M %p").lower()


def _day(t):
    return t.strftime("%a, %b %-d")


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _merged(lines):
    """Consecutive lines from the same speaker, joined into one bubble."""
    turns = []
    for _, speaker, text in lines:
        if turns and turns[-1][0] == speaker:
            turns[-1][1] += " " + text
        else:
            turns.append([speaker, text])
    return turns


def fallback_title(record):
    first = next((text for _, speaker, text in record.lines if speaker == "caller"), "")
    first = first.strip().rstrip(".")
    if len(first) > 60:
        first = first[:57].rsplit(" ", 1)[0] + "…"
    return f"“{first}”" if first else f"A call with {record.role_name}"


def build(records, titles=None):
    """(subject, html, text) for a digest of these calls."""
    titles = titles or [fallback_title(r) for r in records]
    records_titles = sorted(zip(records, titles), key=lambda rt: rt[0].started)
    records = [r for r, _ in records_titles]

    days = collections.Counter(_day(r.start_time) for r in records)
    day = days.most_common(1)[0][0]
    by_role = collections.Counter(r.role_name for r in records)
    talk = sum(r.seconds for r in records)
    longest = max(records, key=lambda r: r.seconds)
    subject = f"📞 {_plural(len(records), 'call')} on the rotary phone · {day}"

    stats = [
        (str(len(records)), _plural(len(records), "call").split(" ", 1)[1]),
        (_duration(talk), "of talking"),
        (by_role.most_common(1)[0][0], "most popular"),
    ]
    first, last = records[0].start_time, records[-1].start_time
    span = _clock(first) if len(records) == 1 else f"{_clock(first)} to {_clock(last)}"
    if len(days) > 1:
        span = f"{_day(first)}, {_clock(first)} to {_day(last)}, {_clock(last)}"

    e = html.escape
    out = [f"""\
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{e(subject)}</title></head>
<body style="margin:0;padding:0;background:{PAGE};">
<div style="display:none;max-height:0;overflow:hidden;">{e(', '.join(t for _, t in records_titles[:3]))}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{PAGE};">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;">
<tr><td style="padding:8px 4px 20px;text-align:center;font-family:{SERIF};color:{INK};">
  <div style="font-size:13px;letter-spacing:3px;text-transform:uppercase;color:{RED};">The Rotary Phone</div>
  <div style="font-size:30px;line-height:1.2;margin-top:6px;">Who called, and what they said</div>
  <div style="font-family:{SANS};font-size:14px;color:{MUTED};margin-top:8px;">{e(span)}</div>
</td></tr>
<tr><td style="background:{CARD};border-radius:14px;border:1px solid {RULE};padding:18px 8px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>"""]
    for value, label in stats:
        out.append(f"""\
    <td width="33%" align="center" valign="top" style="padding:0 6px;">
      <div style="font-family:{SERIF};font-size:22px;color:{INK};">{e(value)}</div>
      <div style="font-family:{SANS};font-size:12px;color:{MUTED};text-transform:uppercase;letter-spacing:1px;margin-top:4px;">{e(label)}</div>
    </td>""")
    out.append("  </tr></table>")

    if len(by_role) > 1:
        top = by_role.most_common(1)[0][1]
        digits = {r.role_name: r.role_digit for r in records}
        out.append(f'  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
                   f'style="margin-top:16px;border-top:1px solid {RULE};">')
        for name, n in by_role.most_common():
            width = max(4, round(100 * n / top))
            out.append(f"""\
    <tr><td width="38%" style="padding:8px 8px 0 12px;font-family:{SANS};font-size:13px;color:{INK};">{e(name)}</td>
      <td style="padding:8px 12px 0 0;"><table role="presentation" width="{width}%" cellpadding="0" cellspacing="0"><tr>
        <td style="background:{_color(digits[name])};height:10px;border-radius:5px;font-size:0;line-height:0;">&nbsp;</td>
      </tr></table></td>
      <td width="24" style="padding:8px 12px 0 0;font-family:{SANS};font-size:13px;color:{MUTED};text-align:right;">{n}</td></tr>""")
        out.append("  </table>")
    out.append("</td></tr>")

    for record, title in records_titles:
        color = _color(record.role_digit)
        longest_note = " · longest call" if record is longest and len(records) > 1 else ""
        out.append(f"""\
<tr><td style="height:18px;"></td></tr>
<tr><td style="background:{CARD};border-radius:14px;border:1px solid {RULE};border-top:5px solid {color};padding:18px 18px 12px;">
  <div style="font-family:{SANS};font-size:12px;color:{MUTED};">
    <span style="display:inline-block;background:{color};color:#fff;border-radius:10px;padding:2px 9px;font-weight:600;">{e(record.role_name)}</span>
    &nbsp;{e(_clock(record.start_time))} · {e(_duration(record.seconds))}{longest_note}
  </div>
  <div style="font-family:{SERIF};font-style:italic;font-size:20px;line-height:1.3;color:{INK};margin:10px 0 12px;">{e(title)}</div>""")
        for speaker, text in _merged(record.lines):
            if speaker == "caller":
                out.append(f"""\
  <div style="text-align:right;margin:0 0 8px 48px;">
    <div style="display:inline-block;text-align:left;background:{CALLER_BUBBLE};color:{INK};border-radius:14px 14px 4px 14px;padding:8px 12px;font-family:{SANS};font-size:15px;line-height:1.4;">{e(text)}</div>
  </div>""")
            else:
                out.append(f"""\
  <div style="margin:0 48px 8px 0;">
    <div style="display:inline-block;background:#fff;border:1px solid {RULE};border-left:3px solid {color};color:{INK};border-radius:4px 14px 14px 14px;padding:8px 12px;font-family:{SANS};font-size:15px;line-height:1.4;">{e(text)}</div>
  </div>""")
        out.append("</td></tr>")

    out.append(f"""\
<tr><td style="padding:22px 4px 8px;text-align:center;font-family:{SANS};font-size:12px;color:{MUTED};">
  Sent by the rotary phone. Transcripts are machine-made, so the caller's words may be a little off.
</td></tr>
</table></td></tr></table></body></html>""")

    text = [subject, span, ""]
    for record, title in records_titles:
        text += ["─" * 40,
                 f"{record.role_name} · {_clock(record.start_time)} · {_duration(record.seconds)}",
                 title, ""]
        for speaker, line in _merged(record.lines):
            text.append(f"{'Caller' if speaker == 'caller' else record.role_name}: {line}")
        text.append("")
    return subject, "\n".join(out), "\n".join(text)


async def titles(cfg, records):
    """A short headline per call from Gemini, or the caller's first words if that fails."""
    from google import genai
    from google.genai import types
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    try:
        client = genai.Client(api_key=cfg.gemini_api_key)
    except Exception as e:
        log.warning("Can't title calls: %s", e)
        return [fallback_title(r) for r in records]
    limit = asyncio.Semaphore(4)

    async def one(record):
        transcript = "\n".join(f"{'Caller' if s == 'caller' else record.role_name}: {t}"
                               for s, t in _merged(record.lines))
        async with limit:
            try:
                response = await asyncio.wait_for(client.aio.models.generate_content(
                    model=cfg.text_model,
                    contents=TITLE_PROMPT.format(name=record.role_name, transcript=transcript[:6000]),
                    config=config),
                    TITLE_TIMEOUT_S)
                title = (response.text or "").strip().strip('"*').splitlines()[0].strip()
                if title:
                    return title
            except Exception as e:
                log.warning("Couldn't title a call: %s", e)
        return fallback_title(record)

    return await asyncio.gather(*(one(r) for r in records))


def send(cfg, subject, html_body, text_body):
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr(("Rotary Phone", cfg.smtp_user))
    message["To"] = cfg.email_to
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    with smtplib.SMTP_SSL(cfg.smtp_host, 465, timeout=30) as smtp:
        smtp.login(cfg.smtp_user, cfg.smtp_password)
        smtp.send_message(message)


async def send_digest(cfg, journal, preview=None):
    """Email every call not yet sent (or write the HTML to `preview` and send nothing).
    Returns how many calls it covered."""
    records, offset = journal.unsent()
    if not records:
        return 0
    subject, html_body, text_body = build(records, await titles(cfg, records))
    if preview:
        with open(preview, "w", encoding="utf-8") as f:
            f.write(html_body)
        return len(records)
    await asyncio.to_thread(send, cfg, subject, html_body, text_body)
    journal.mark_sent(offset)
    log.info("Emailed the transcripts of %s to %s", _plural(len(records), "call"), cfg.email_to)
    return len(records)


def next_run(now, hour):
    run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return run if run > now else run + datetime.timedelta(days=1)
