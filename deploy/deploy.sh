#!/usr/bin/env bash
# Deploy to the phone: sync code and roles, install dependencies, the user unit
# and the echo-cancel config, then (re)start the service.
#
#   deploy/deploy.sh             deploy and restart (after cutover)
#   deploy/deploy.sh --cutover   one time: retire the legacy system service, start the new one
#   deploy/deploy.sh --test      deploy, then run the end-to-end suite on the Pi
#
# PHONE_HOST overrides the target (default pi@phone.local).
set -euo pipefail

HOST=${PHONE_HOST:-pi@phone.local}
CUTOVER=0
TEST=0
for arg in "$@"; do
  case $arg in
    --cutover) CUTOVER=1 ;;
    --test) TEST=1 ;;
    *) echo "usage: $0 [--cutover] [--test]" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
remote() { timeout 900 ssh -o ConnectTimeout=10 "$HOST" "$@"; }

[[ -f roles.yaml ]] || { echo "roles.yaml is missing: copy roles.example.yaml and edit it" >&2; exit 1; }

echo "==> Syncing code to $HOST:~/phone"
timeout 300 rsync -az --delete \
  --exclude .venv --exclude __pycache__ --exclude '.lgd-nfy*' --exclude tests/e2e/reports \
  phone sounds tests deploy spikes pyproject.toml requirements.txt requirements-dev.txt "$HOST:phone/"
timeout 60 rsync -a roles.yaml "$HOST:.config/gpt-phone/roles.yaml"

echo "==> Installing dependencies, unit and echo-cancel config"
remote 'bash -s' <<'EOF'
set -euo pipefail
chmod 600 ~/.config/gpt-phone/roles.yaml ~/.config/gpt-phone/env
cd ~/phone
[[ -x .venv/bin/python ]] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q --disable-pip-version-check -r requirements.txt -r requirements-dev.txt
mkdir -p ~/.config/systemd/user ~/.config/pipewire/pipewire.conf.d
install -m 644 deploy/gpt-phone.service ~/.config/systemd/user/gpt-phone.service
if ! cmp -s deploy/60-echo-cancel.conf ~/.config/pipewire/pipewire.conf.d/60-echo-cancel.conf; then
  install -m 644 deploy/60-echo-cancel.conf ~/.config/pipewire/pipewire.conf.d/
  systemctl --user restart pipewire.service pipewire-pulse.service wireplumber.service
  sleep 3
fi
pactl set-default-sink phone_aec_sink || true
pactl set-default-source phone_aec_source || true
# Mic gain all the way up: the handset's mouthpiece is quiet, and soft words
# otherwise don't score as speech. Both the adapter's own gain and PipeWire's.
card=$(awk '/C-Media/ {print $1; exit}' /proc/asound/cards)
if [[ -n $card ]]; then
  amixer -q -c "$card" sset Mic capture 100% cap || true
  sudo alsactl store "$card" 2>/dev/null || true
fi
pactl set-source-volume alsa_input.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.mono-fallback 100% || true
pactl set-source-volume phone_aec_source 100% || true
systemctl --user daemon-reload
# Wi-Fi power saving on the Pi causes dropouts and latency spikes. The drop-in
# covers future connections; iw applies it now without reconnecting.
if [[ ! -f /etc/NetworkManager/conf.d/99-gpt-phone-wifi.conf ]]; then
  printf '[connection]\nwifi.powersave = 2\n' | sudo tee /etc/NetworkManager/conf.d/99-gpt-phone-wifi.conf >/dev/null
fi
sudo iw dev wlan0 set power_save off 2>/dev/null || true
EOF

if (( CUTOVER )); then
  echo "==> Cutover: retiring the legacy system service (files kept in ~/legacy for rollback)"
  remote 'bash -s' <<'EOF'
set -euo pipefail
sudo systemctl disable --now phone.service 2>/dev/null || true
mkdir -p ~/legacy && chmod 700 ~/legacy
for f in start-phone.sh gpt-phone.py roles.py creepy.mp3 dialtone.mp3 gcp-creds.json; do
  [[ -e ~/$f ]] && mv ~/$f ~/legacy/
done
chmod 600 ~/legacy/* 2>/dev/null || true
chmod 700 ~/legacy/start-phone.sh 2>/dev/null || true
systemctl --user enable gpt-phone.service
EOF
fi

# Only a cutover (which enables the unit) hands the phone to the new service;
# the legacy one merely being stopped (e.g. for a dial test) doesn't count.
enabled=$(remote 'systemctl --user is-enabled gpt-phone.service 2>/dev/null || true')
if [[ $enabled == enabled ]]; then
  can_start=1
else
  echo "!! gpt-phone isn't enabled yet, so it wasn't (re)started."
  echo "   Run with --cutover when ready (deploy/rollback-legacy.sh undoes it)."
  can_start=0
fi

if (( TEST )); then
  echo "==> Running end-to-end tests on the phone (service paused)"
  remote 'systemctl --user stop gpt-phone.service 2>/dev/null || true'
  set +e
  remote 'cd ~/phone && set -a && . ~/.config/gpt-phone/env && set +a &&
          PHONE_ROLES_FILE=~/.config/gpt-phone/roles.yaml .venv/bin/python -m pytest -m e2e tests/e2e -v'
  status=$?
  set -e
  mkdir -p tests/e2e/reports
  timeout 60 rsync -a "$HOST:phone/tests/e2e/reports/" tests/e2e/reports/ || true
  echo "==> Reports copied to tests/e2e/reports/"
fi

if (( can_start )); then
  echo "==> Restarting gpt-phone"
  remote 'systemctl --user restart gpt-phone.service && sleep 5 &&
          systemctl --user show gpt-phone.service -p ActiveState -p StatusText --no-pager'
fi

exit "${status:-0}"
