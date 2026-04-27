# Multi-Agent Learning — Three Progressive Projects

A learning journey through multi-agent LLM orchestration, building progressively from a CrewAI sequential pipeline → a hierarchical CrewAI crew → a hand-rolled Anthropic ReAct loop deployed as an interactive Telegram bot.

All three projects use **Claude Haiku 4.5** as the underlying LLM.

> **🤖 Try the live Telegram bot:** [@TruthComitteeBot](https://t.me/TruthComitteeBot)
>
> Send it a controversial claim or a numeric question. The Chief Editor will dispatch the Investigator, hand the file to the Analyst, crunch numbers with the Calculator, and deliver a verdict — narrating each step. Running 24×7 on a Google Cloud free-tier VM.

## What's in here

| File | What it is | Stack |
|---|---|---|
| [`main.py`](main.py) | **Tech News Pipeline** — a 2-agent sequential CrewAI crew (Researcher → Writer) that drafts a blog post about Anthropic's Claude Managed Agents. | CrewAI + LangChain (DuckDuckGo search) |
| [`truth_committee.py`](truth_committee.py) | **The Truth Committee (CrewAI)** — a 3-agent **hierarchical** crew (Lead Investigator + Skeptical Analyst, managed by a Chief Editor) that produces a balanced Risk-Benefit Report on humanoid robotics in manufacturing by 2027, with confidence scores per claim. | CrewAI hierarchical process |
| [`TruthComitteeBot.py`](TruthComitteeBot.py) | **The Truth Committee (Telegram bot)** — same concept, rebuilt without a framework. Uses the Anthropic SDK directly with a hand-written ReAct loop. Multi-user, async, streams status updates as the committee deliberates. | python-telegram-bot v20+, AsyncAnthropic, ddgs |

## The progression

1. **Sequential** ([`main.py`](main.py)) — Researcher hands off to Writer in a fixed order. Linear pipeline. Easiest mental model.
2. **Hierarchical** ([`truth_committee.py`](truth_committee.py)) — A Chief Editor decides who works next and can loop back for revisions when the Analyst flags gaps.
3. **Native ReAct, no framework** ([`TruthComitteeBot.py`](TruthComitteeBot.py)) — Drop CrewAI entirely. Hand-written async ReAct loop with explicit tool dispatch, per-role iteration caps, programmatic backstops, and Telegram streaming for many concurrent users.

The three files together demonstrate why you might *use* a framework like CrewAI early on (fast to a prototype) and why you might eventually *drop* it (token bloat, opacity, less control over iteration logic).

## Local setup

```bash
git clone <this-repo-url>
cd <repo-folder>

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# Edit .env — paste your real ANTHROPIC_API_KEY (and TELEGRAM_TOKEN if running the bot)
```

Get keys:
- **Anthropic API key:** https://console.anthropic.com/settings/keys
- **Telegram bot token:** message [@BotFather](https://t.me/BotFather) on Telegram, `/newbot`, follow prompts.

## Running each project

```bash
.venv/bin/python main.py             # Tech News Pipeline (CrewAI sequential)
.venv/bin/python truth_committee.py  # Hierarchical Truth Committee (CrewAI)
.venv/bin/python TruthComitteeBot.py # Telegram bot (native Anthropic ReAct)
```

`main.py` and `truth_committee.py` run once and print to stdout. `TruthComitteeBot.py` is a long-running poller — Ctrl+C to stop.

## Deploying the bot to Google Cloud free tier

The Telegram bot is configured to deploy on a Google Cloud `e2-micro` VM (Always Free in `us-central1`, `us-east1`, or `us-west1`). See [`deploy/README.md`](deploy/README.md) for step-by-step instructions, including the systemd unit that keeps the bot running 24×7 across reboots.

## Architecture notes for `TruthComitteeBot.py`

The interesting bits, condensed:

- **No framework.** ~600 lines including comments. Direct AsyncAnthropic + python-telegram-bot.
- **ReAct loop** — `while iter < MAX_EDITOR_ITERATIONS`: call Anthropic with the Editor's system prompt and tools; on `stop_reason == "tool_use"`, dispatch each tool, append results, repeat; on `end_turn`, send the final reply.
- **Per-role caps** (`ROLE_CAPS`) — at most 3 investigator calls, 3 analyst calls, 3 calculator calls per user message. Once both investigator and analyst caps are hit, the next API call drops the tools list, forcing the model to synthesize a verdict.
- **Per-chat asyncio.Lock + history dict** — many concurrent users, but a single user's rapid double-message can't race their own history. Different chats run in parallel.
- **Status-first ordering** — the user sees `🕵️ Dispatching Investigator…` *before* the slow tool call returns, so the chat feels alive.
- **Safe calculator** — pure ast-based, never calls `eval()`. Whitelisted nodes only (no Names, Calls, Attributes).
- **Markdown → Telegram-HTML converter** so Claude's `**bold**` and `## headers` actually render in Telegram.
- **Concurrency gotcha** — `python-telegram-bot` v20+ processes updates *serially* by default. To get true parallelism across users, the Application is built with `concurrent_updates(True)`.

## File structure

```
.
├── main.py                       # Project 1: CrewAI sequential
├── truth_committee.py            # Project 2: CrewAI hierarchical
├── TruthComitteeBot.py           # Project 3: native Anthropic + Telegram
├── requirements.txt              # All deps for all three projects
├── .env.example                  # Template — copy to .env and fill in
├── deploy/
│   ├── README.md                 # GCP deployment walkthrough
│   ├── setup.sh                  # One-shot VM setup script
│   └── truth-committee-bot.service  # systemd unit
└── README.md                     # this file
```

## License

MIT — do whatever you like with it.
