#!/usr/bin/env bash
# Undo the cutover: stop the new service and bring back the legacy phone.service
# with the files that deploy.sh --cutover moved into ~/legacy.
# PHONE_HOST overrides the target (default pi@phone.local).
set -euo pipefail

HOST=${PHONE_HOST:-pi@phone.local}

timeout 120 ssh -o ConnectTimeout=10 "$HOST" 'bash -s' <<'EOF'
set -euo pipefail
systemctl --user disable --now gpt-phone.service 2>/dev/null || true
if [[ -e ~/.config/pipewire/pipewire.conf.d/60-echo-cancel.conf ]]; then
  rm ~/.config/pipewire/pipewire.conf.d/60-echo-cancel.conf
  systemctl --user restart pipewire.service pipewire-pulse.service wireplumber.service
  sleep 3
  pactl set-default-sink alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo || true
  pactl set-default-source alsa_input.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.mono-fallback || true
fi
for f in start-phone.sh gpt-phone.py roles.py creepy.mp3 dialtone.mp3 gcp-creds.json; do
  [[ -e ~/legacy/$f && ! -e ~/$f ]] && cp -p ~/legacy/$f ~/
done
chmod 755 ~/start-phone.sh
sudo systemctl enable --now phone.service
sleep 5
systemctl status phone.service --no-pager | head -5
EOF
echo "Legacy service restored. Re-deploy the new one with: deploy/deploy.sh --cutover"
