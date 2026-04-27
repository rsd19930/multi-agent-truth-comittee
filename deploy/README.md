# Deploying the Telegram bot to Google Cloud (Always Free e2-micro)

## What you'll get

- A free Linux VM running 24×7 in Google Cloud (`us-central1`, `us-east1`, or `us-west1`).
- The bot starts automatically on boot via systemd and restarts on crash.
- Logs viewable with `journalctl`.
- **Cost: $0/month** within the [Always Free quota](https://cloud.google.com/free) (1× e2-micro VM, 30 GB standard persistent disk).

A credit card is required at GCP signup for verification, but you won't be charged while staying within the always-free limits.

---

## Step 1 — One-time GCP setup (web console)

1. Sign in at https://console.cloud.google.com.
2. Create a new project (or reuse one) and select it. Note the project ID.
3. Enable the Compute Engine API:
   - Console → **APIs & Services** → **Library** → search "Compute Engine API" → **Enable**.
4. Create the VM:
   - Console → **Compute Engine** → **VM instances** → **Create instance**.
   - **Name:** `truth-committee-bot`
   - **Region:** `us-central1` *(must be one of `us-central1`, `us-east1`, or `us-west1` for Always Free)*
   - **Zone:** any zone in that region.
   - **Machine configuration:** Series `E2`, machine type `e2-micro`.
   - **Boot disk:** click *Change* → **OS:** Debian 12 → **Size:** 30 GB (free tier allowance).
   - **Firewall:** leave HTTP/HTTPS unchecked. The bot only makes outbound calls (Telegram + Anthropic), no inbound web traffic.
   - Click **Create**.
5. SSH in: from the VM list, click the **SSH** button next to your instance — a browser SSH session opens.

---

## Step 2 — On the VM: install and start the bot

In the SSH session, paste:

```bash
# Replace <REPO_URL> with your GitHub repo URL, e.g. https://github.com/you/multi-agents-learning.git
curl -sSL https://raw.githubusercontent.com/<USER>/<REPO>/main/deploy/setup.sh \
  | bash -s -- https://github.com/<USER>/<REPO>.git
```

The script will:
1. Install `python3`, `python3-venv`, `git` via `apt`.
2. Clone the repo to `~/multi-agents-learning`.
3. Create a `.venv` and install all dependencies from `requirements.txt`.
4. Copy `.env.example` to `.env` (you'll fill in real values next).
5. Install and enable the `truth-committee-bot` systemd service.

Then edit `.env` with your real keys:

```bash
nano ~/multi-agents-learning/.env
# Paste real values for ANTHROPIC_API_KEY and TELEGRAM_TOKEN.
# Save with Ctrl+O, Enter, Ctrl+X.
```

Start the bot:

```bash
sudo systemctl start truth-committee-bot.service
sudo systemctl status truth-committee-bot.service   # confirm it's "active (running)"
```

Watch logs in real time:

```bash
journalctl -u truth-committee-bot.service -f
```

Test it: open Telegram, find your bot, send `/start`. You should get the welcome message within a couple of seconds.

---

## Updating the bot later

```bash
cd ~/multi-agents-learning
git pull
.venv/bin/pip install -r requirements.txt   # only needed if requirements changed
sudo systemctl restart truth-committee-bot.service
```

## Stopping or disabling

```bash
sudo systemctl stop truth-committee-bot.service       # stop now (still re-enabled on boot)
sudo systemctl disable truth-committee-bot.service    # don't start on boot
```

## Common gotchas

- **"Failed to start" in `systemctl status`** → check `journalctl -u truth-committee-bot.service -n 50` for the actual error. Almost always a typo in `.env` or a missing `TELEGRAM_TOKEN`.
- **Bot starts but doesn't respond** → confirm the token is from BotFather and matches the bot username you're messaging. Each token belongs to exactly one bot.
- **Quota exceeded warnings on the GCP console** → make sure the VM is in `us-central1`/`us-east1`/`us-west1`, machine type is `e2-micro`, and disk is ≤30 GB. Anything else falls outside the always-free quota.
