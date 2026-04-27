#!/usr/bin/env bash
# One-shot setup for a fresh Debian/Ubuntu VM (e.g. a Google Cloud e2-micro).
# Idempotent — safe to re-run.
#
# Usage (run as the user that will own the bot, NOT root):
#   curl -sSL https://raw.githubusercontent.com/<USER>/<REPO>/main/deploy/setup.sh \
#     | bash -s -- https://github.com/<USER>/<REPO>.git
#
# Or, if you've already cloned the repo:
#   ./deploy/setup.sh https://github.com/<USER>/<REPO>.git

set -euo pipefail

REPO_URL="${1:-}"
if [[ -z "$REPO_URL" ]]; then
  echo "ERROR: pass the repo URL as the first argument." >&2
  echo "Usage: $0 <git-repo-url>" >&2
  exit 1
fi

PROJECT_DIR="$HOME/multi-agents-learning"

echo "==> Installing system packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git

echo "==> Cloning repo"
if [[ -d "$PROJECT_DIR/.git" ]]; then
  echo "    repo already exists at $PROJECT_DIR — pulling latest"
  git -C "$PROJECT_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

echo "==> Creating venv and installing Python deps"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> Setting up .env"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo
  echo "    Created .env from template. EDIT IT NOW:"
  echo "        nano $PROJECT_DIR/.env"
  echo "    Paste your real ANTHROPIC_API_KEY and TELEGRAM_TOKEN."
  echo
fi

echo "==> Installing systemd unit"
UNIT_SRC="$PROJECT_DIR/deploy/truth-committee-bot.service"
UNIT_DST="/etc/systemd/system/truth-committee-bot.service"
# Substitute __USER__ and __HOME__ placeholders with the actual values.
sudo sed -e "s|__USER__|$USER|g" -e "s|__HOME__|$HOME|g" \
  "$UNIT_SRC" | sudo tee "$UNIT_DST" > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable truth-committee-bot.service

echo
echo "==> Setup complete."
echo
echo "Next steps:"
echo "  1. Edit your secrets:    nano $PROJECT_DIR/.env"
echo "  2. Start the bot:        sudo systemctl start truth-committee-bot.service"
echo "  3. Check status:         sudo systemctl status truth-committee-bot.service"
echo "  4. Tail logs:            journalctl -u truth-committee-bot.service -f"
echo
