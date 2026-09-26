# Agent notes: gpt-phone

Rotary phone → AI characters. Python 3.11 asyncio app in `phone/`, deployed to a
Raspberry Pi 4 (`ssh pi@phone.local`, passwordless, passwordless sudo). See README.md
for architecture.

## Rules
- **The GitHub repo is public.** Never commit `roles.yaml` (personal persona details),
  `.env`, `deploy/env`, or anything with API keys. Grep staged diffs for `sk-`, `sk_`,
  `AIza`, `AQ.` (newer Gemini keys) before pushing. `roles.example.yaml` holds only non-personal characters.
- Wrap SSH, rsync and other network commands in `timeout`.
- Test on the Pi, not just the Mac: audio goes through PipeWire and a USB C-Media
  adapter, and GPIO only exists there.
- The legacy version is tagged `legacy-v1`; `deploy/rollback-legacy.sh` restores it on the Pi.

## Commands
- Unit tests (Mac or Pi): `.venv/bin/python -m pytest`
- Deploy: `deploy/deploy.sh` (add `--test` to run the e2e suite on the Pi)
- Resilience tests (from the Mac, disruptive): `.venv/bin/python -m pytest -m system tests/system`
- Local run: `.venv/bin/python -m phone --no-gpio --role N`
- Logs on the Pi: `journalctl --user -u gpt-phone -f`

## Layout on the Pi
- Code: `~/phone` (venv at `~/phone/.venv`, created with `--system-site-packages` for
  Debian's gpiozero/lgpio)
- Secrets and roles: `~/.config/gpt-phone/{env,roles.yaml}` (mode 600)
- Prompt cache: `~/.cache/gpt-phone/prompts/`
- Service: systemd **user** unit `gpt-phone.service` (linger enabled); echo canceller
  config in `~/.config/pipewire/pipewire.conf.d/60-echo-cancel.conf`
- Legacy files (after cutover): `~/legacy/`
