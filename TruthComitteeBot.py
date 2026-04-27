"""
The Truth Committee — Telegram Bot (native Anthropic ReAct loop, no CrewAI).

Concept:
    A user sends a controversial statement or numeric question. A "Chief Editor"
    LLM agent orchestrates a small team — Investigator (web search), Analyst
    (skeptical critique), Calculator (deterministic math) — and replies with a
    synthesized, evidence-aware verdict. Status updates stream into Telegram
    while the committee deliberates so the user sees the drama unfold.

Architecture (one file, intentionally explicit):
    Telegram update  -->  handle_message()
                          └─ async run_committee(chat_id, text, send_status)
                              └─ ReAct loop on AsyncAnthropic.messages.create
                                  ├─ tool: delegate_to_sub_agent  -->  run_investigator | run_analyst | run_calculator
                                  └─ tool: calculator             -->  safe_eval

Concurrency:
    python-telegram-bot v20+ runs each handler as an asyncio task, so multiple
    users hit the bot in parallel for free. Per-user state lives in HISTORIES
    (chat_id -> message list) and is guarded by a per-chat asyncio.Lock to
    prevent two rapid messages from the same user from interleaving.
"""

from __future__ import annotations

import ast
import asyncio
import html as html_lib
import logging
import operator
import os
import re
from collections import defaultdict
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic
from ddgs import DDGS
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# override=True so the .env values beat any (possibly empty) shell vars — same
# gotcha hit during the CrewAI build of this project.
load_dotenv(override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")
if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your-telegram-bot-token-here":
    raise RuntimeError("TELEGRAM_TOKEN is not set. Add a real token to .env.")

MODEL = "claude-haiku-4-5"
MAX_EDITOR_ITERATIONS = 10       # outer ReAct loop cap (per user message)
MAX_INVESTIGATOR_ITERATIONS = 3  # inner search loop cap (per delegate call)
HISTORY_LIMIT = 10               # rolling messages retained per chat
PREVIEW_LIMIT = 280              # max chars of a delegated task shown to user

# Per-role hard caps. Once hit, the Editor's tools are stripped on the next
# API call, forcing it to synthesize a verdict from what it already has.
ROLE_CAPS = {"investigator": 3, "analyst": 3, "calculator": 3}

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("truth-committee")

client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# Per-chat conversation history (Anthropic message dicts) and per-chat locks.
HISTORIES: dict[int, list[dict[str, Any]]] = defaultdict(list)
CHAT_LOCKS: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


# ===========================================================================
# Tool 1: Safe calculator (ast-based; never calls eval())
# ===========================================================================
# Whitelisted AST operators only. Anything else (names, calls, attributes,
# subscripts, comprehensions) trips the visitor and raises.
_SAFE_OPS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate a pure arithmetic expression. No names, no calls, no attrs."""
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Disallowed binary op: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Disallowed unary op: {type(node.op).__name__}")
            return op(_eval(node.operand))
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    return _eval(tree)


# ===========================================================================
# Sub-agent: Investigator (LLM + web search, internal mini ReAct loop)
# ===========================================================================
INVESTIGATOR_SYSTEM = (
    "You are the INVESTIGATOR — a hard-boiled fact-finder. Terse. Sourced. "
    "Skeptical of vendor claims. When given a task, search the web (use the "
    "duckduckgo_search tool aggressively), gather 3-6 concrete facts, and "
    "return a tight bullet list. Each bullet must end with `[Source: <url or "
    "publication>]`. If you cannot verify a claim, say so explicitly. "
    "No hedging filler. Just the facts and where they came from."
)

INVESTIGATOR_TOOLS = [
    {
        "name": "duckduckgo_search",
        "description": "Search the web via DuckDuckGo. Returns text snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
    }
]


def _ddg_search_sync(query: str, max_results: int = 5) -> str:
    """Blocking DuckDuckGo call — wrapped in to_thread() at the call site."""
    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        return f"[search error: {e}]"
    if not results:
        return "[no results]"
    return "\n\n".join(
        f"- {r.get('title', '')}\n  {r.get('body', '')}\n  {r.get('href', '')}"
        for r in results
    )


async def run_investigator(task: str) -> str:
    """Single-shot Investigator: own ReAct mini-loop with web search."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    for _ in range(MAX_INVESTIGATOR_ITERATIONS):
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=INVESTIGATOR_SYSTEM,
            tools=INVESTIGATOR_TOOLS,
            messages=messages,
        )
        if resp.stop_reason == "end_turn":
            return _extract_text(resp)
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    if block.name == "duckduckgo_search":
                        # ddgs is synchronous; offload so we don't block the loop.
                        result = await asyncio.to_thread(
                            _ddg_search_sync, block.input["query"]
                        )
                    else:
                        result = f"[unknown tool: {block.name}]"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            continue
        # Any other stop_reason: bail with whatever text we have.
        return _extract_text(resp) or "[Investigator returned no text]"
    return _extract_text(resp) or "[Investigator hit iteration cap]"


# ===========================================================================
# Sub-agent: Analyst (LLM, no tools — critiques what's passed in)
# ===========================================================================
ANALYST_SYSTEM = (
    "You are the ANALYST — a hyper-logical skeptic who loves crunching "
    "numbers. Given findings or a claim, identify: (a) logical leaps, "
    "(b) unverified assertions, (c) numeric implausibilities, (d) what's "
    "MISSING. Quote concrete percentages, base rates, and orders of "
    "magnitude wherever possible. End with a Confidence Score (0.0-1.0) "
    "for the overall claim and a one-line recommendation: VERIFY FURTHER, "
    "ACCEPT, or REJECT."
)


async def run_analyst(task: str) -> str:
    """Single-shot Analyst: no tools, just sharp prose."""
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=ANALYST_SYSTEM,
        messages=[{"role": "user", "content": task}],
    )
    return _extract_text(resp) or "[Analyst returned no text]"


# ===========================================================================
# Sub-agent: Calculator (deterministic — no LLM)
# ===========================================================================
def run_calculator(expression: str) -> str:
    """Wraps safe_eval with a stable text format the Editor can quote."""
    try:
        value = safe_eval(expression)
        return f"Result: {expression} = {value}"
    except Exception as e:
        return f"Calculator error: {e}"


# ===========================================================================
# Editor: tool schemas, system prompt, and the orchestration loop
# ===========================================================================
EDITOR_SYSTEM = (
    "You are the CHIEF EDITOR of The Truth Committee. Theatrical, dramatic, "
    "engaging — you narrate the investigation as it unfolds.\n\n"
    "Your job: when the user makes a claim or asks a controversial question, "
    "you MUST run a real investigation before answering. Do not answer from "
    "training data alone.\n\n"
    "WORKFLOW — strict step counts, do not exceed:\n"
    "  1. INVESTIGATE (1st round, REQUIRED): one `delegate_to_sub_agent` "
    "     call with role='investigator' covering ALL the facts you need. "
    "     Write a thorough brief upfront — you only get one shot before the "
    "     critique.\n"
    "  2. CALCULATE (if numbers are involved): up to 3 `calculator` calls "
    "     total. Never compute numbers in your head.\n"
    "  3. CRITIQUE (1st round, REQUIRED): one `delegate_to_sub_agent` call "
    "     with role='analyst', passing the Investigator's findings. The "
    "     Analyst returns a Confidence Score (0.0-1.0) and a recommendation: "
    "     VERIFY FURTHER, ACCEPT, or REJECT.\n"
    "  4. ADDITIONAL PASSES (CONDITIONAL): if the Analyst returned VERIFY "
    "     FURTHER or Confidence < 0.8 AND named a SPECIFIC gap, run another "
    "     targeted investigator call followed by another analyst call. You "
    "     may do this up to a total of 3 investigator and 3 analyst calls. "
    "     Stop sooner if the Analyst returns ACCEPT or REJECT with "
    "     Confidence ≥ 0.8.\n"
    "  5. SYNTHESIZE: deliver the final verdict in your dramatic voice. "
    "     Acknowledge any remaining uncertainty. Cite the sources the "
    "     Investigator surfaced. Note in a final methodology line how many "
    "     investigator and analyst rounds the committee ran.\n\n"
    "HARD CAPS — exceeding these wastes the user's time:\n"
    "  • At most 3 investigator calls total.\n"
    "  • At most 3 analyst calls total.\n"
    "  • At most 3 calculator calls total.\n"
    "  • Once the analyst's second-round verdict is in, STOP delegating "
    "    and synthesize, even if some uncertainty remains. Acknowledge the "
    "    gap in your final answer rather than chasing it forever.\n\n"
    "Format the final reply in Markdown. Use **bold** for emphasis and "
    "## or ### for headers (these will render properly in Telegram). "
    "Keep the final reply under ~400 words. End with a clear bottom line."
)

EDITOR_TOOLS = [
    {
        "name": "delegate_to_sub_agent",
        "description": (
            "Delegate a task to a specialist on your committee. "
            "Roles: 'investigator' (web research, sourced facts), "
            "'analyst' (skeptical critique with confidence scores), "
            "'calculator' (deterministic arithmetic on the given expression)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_role": {
                    "type": "string",
                    "enum": ["investigator", "analyst", "calculator"],
                },
                "task": {
                    "type": "string",
                    "description": (
                        "The instruction or question for the sub-agent. "
                        "For the calculator, pass a pure arithmetic expression "
                        "like '0.30 * 4200000000000'."
                    ),
                },
            },
            "required": ["agent_role", "task"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a pure arithmetic expression deterministically. "
            "Supports + - * / // % ** and parentheses. No variables, no calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "e.g. '(0.3 * 4.2e12) / 1e9'",
                },
            },
            "required": ["expression"],
        },
    },
]


def _extract_text(resp) -> str:
    """Concatenate all `text` blocks from an Anthropic response."""
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _preview(task: str, limit: int = PREVIEW_LIMIT) -> str:
    """Flatten a delegated task into a single readable line for status UX."""
    one_line = re.sub(r"\s+", " ", task).strip()
    if len(one_line) <= limit:
        return one_line
    cut = one_line[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _pre_tool_status(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Status sent BEFORE a tool runs — describes the handoff."""
    if tool_name == "delegate_to_sub_agent":
        role = tool_input.get("agent_role", "?")
        task = tool_input.get("task", "")
        if role == "investigator":
            return (
                "🕵️ <b>The Editor dispatches the Investigator.</b>\n"
                f"<i>Brief:</i> {html_lib.escape(_preview(task), quote=False)}"
            )
        if role == "analyst":
            return (
                "🧠 <b>The Editor hands the file to the Analyst for critique…</b>"
            )
        if role == "calculator":
            return (
                f"📟 <b>The Editor consults the Calculator:</b> "
                f"<code>{html_lib.escape(_preview(task, 120), quote=False)}</code>"
            )
        return f"📨 <b>Delegating to {html_lib.escape(role)}…</b>"
    if tool_name == "calculator":
        expr = tool_input.get("expression", "?")
        return (
            f"📟 <b>The Editor crunches the numbers:</b> "
            f"<code>{html_lib.escape(_preview(expr, 120), quote=False)}</code>"
        )
    return f"🔧 <b>Tool:</b> {html_lib.escape(tool_name)}"


def _post_tool_status(
    tool_name: str, tool_input: dict[str, Any], is_error: bool
) -> str | None:
    """Status sent AFTER a tool returns — describes what came back. Optional."""
    if is_error:
        return "⚠️ <b>A spanner in the works</b> — the committee adapts and presses on."
    if tool_name == "delegate_to_sub_agent":
        role = tool_input.get("agent_role", "?")
        if role == "investigator":
            return (
                "📂 <b>The Investigator returns with their report.</b> "
                "The Editor reads it carefully…"
            )
        if role == "analyst":
            return (
                "📋 <b>The Analyst delivers the critique.</b> "
                "The Editor weighs the evidence…"
            )
        if role == "calculator":
            return "🧮 <b>The Calculator's numbers land on the desk.</b>"
    if tool_name == "calculator":
        return "🧮 <b>Numbers verified.</b>"
    return None  # Unknown tools: no post-status (pre-status was enough).


def _role_for_call(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """Map a tool invocation to a role key for cap-tracking. None = uncounted."""
    if tool_name == "calculator":
        return "calculator"
    if tool_name == "delegate_to_sub_agent":
        role = tool_input.get("agent_role")
        if role in ROLE_CAPS:
            return role
    return None


async def _execute_tool(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Run one tool. Returns (content, is_error)."""
    try:
        if name == "calculator":
            return run_calculator(tool_input["expression"]), False
        if name == "delegate_to_sub_agent":
            role = tool_input["agent_role"]
            task = tool_input["task"]
            if role == "investigator":
                return await run_investigator(task), False
            if role == "analyst":
                return await run_analyst(task), False
            if role == "calculator":
                return run_calculator(task), False
            return f"Unknown role: {role}", True
        return f"Unknown tool: {name}", True
    except Exception as e:
        log.exception("tool error")
        return f"Tool {name} raised: {e}", True


SendStatus = Callable[[str], Awaitable[None]]


async def run_committee(
    chat_id: int,
    user_text: str,
    send_status: SendStatus,
) -> str:
    """
    The Editor's outer ReAct loop. Runs until Claude returns end_turn or we
    hit MAX_EDITOR_ITERATIONS. Streams pre/post status messages around each
    tool call so the user sees the handoffs.
    """
    history = HISTORIES[chat_id]
    # Work on a copy so a partial run doesn't poison history if we crash.
    messages = list(history) + [{"role": "user", "content": user_text}]

    # Opening flourish: tells the user the committee is convening before any
    # LLM latency hits.
    try:
        await send_status(
            "🎭 <b>The Chief Editor reviews your claim and convenes the committee…</b>"
        )
    except Exception:
        log.exception("opening status send failed")

    # Track per-role usage so we can revoke tools once caps are hit.
    role_usage: dict[str, int] = {"investigator": 0, "analyst": 0, "calculator": 0}

    final_text: str | None = None
    for iteration in range(MAX_EDITOR_ITERATIONS):
        # Programmatic backstop. Once an investigator AND analyst round have
        # both happened twice, OR any single role has hit its cap, strip the
        # tools list — the model now has no choice but to synthesize text.
        # This is the hard guarantee that the loop terminates regardless of
        # what the model "wants" to do next.
        caps_exhausted = (
            role_usage["investigator"] >= ROLE_CAPS["investigator"]
            and role_usage["analyst"] >= ROLE_CAPS["analyst"]
        )
        tools_arg = [] if caps_exhausted else EDITOR_TOOLS

        resp = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=EDITOR_SYSTEM,
            tools=tools_arg,
            messages=messages,
        )

        if resp.stop_reason == "end_turn":
            final_text = _extract_text(resp) or "[Editor returned no text]"
            messages.append({"role": "assistant", "content": resp.content})
            break

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue

                # Track role usage. If THIS specific role is over cap, refuse
                # the call inline by returning an error tool_result that tells
                # the Editor to stop and synthesize.
                role = _role_for_call(block.name, block.input)
                if role is not None and role_usage[role] >= ROLE_CAPS[role]:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": (
                                f"[budget exhausted: the {role} cap of "
                                f"{ROLE_CAPS[role]} calls is already used. "
                                "Synthesize the final verdict now from what "
                                "you already have.]"
                            ),
                            "is_error": True,
                        }
                    )
                    continue

                # Pre-status FIRST so the user sees motion before we wait on the tool.
                try:
                    await send_status(_pre_tool_status(block.name, block.input))
                except Exception:
                    log.exception("pre-status send failed")

                content, is_err = await _execute_tool(block.name, block.input)
                if role is not None:
                    role_usage[role] += 1

                # Post-status: announces what came back.
                post = _post_tool_status(block.name, block.input, is_err)
                if post is not None:
                    try:
                        await send_status(post)
                    except Exception:
                        log.exception("post-status send failed")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                        "is_error": is_err,
                    }
                )

            # Empty-content guard: if the API said tool_use but returned no
            # tool_use blocks we could process, sending {"role":"user","content":[]}
            # would 400 with "user messages must have non-empty content". Bail
            # cleanly with whatever text the assistant already produced.
            if not tool_results:
                log.warning(
                    "tool_use response had no actionable tool_use blocks; "
                    "stopping loop to avoid empty-content API error"
                )
                final_text = _extract_text(resp) or (
                    "[Editor stopped: tool_use signaled but no tool calls "
                    "were emitted]"
                )
                break

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason — bail with whatever text we have.
        final_text = _extract_text(resp) or f"[Editor stopped: {resp.stop_reason}]"
        break
    else:
        final_text = (
            "🎬 The committee deliberated past its deadline. Here's the best we "
            "have so far — please ask again with a sharper question."
        )

    # Curtain call before the verdict.
    try:
        await send_status("🎬 <b>The committee delivers its verdict…</b>")
    except Exception:
        log.exception("curtain status send failed")

    # Commit to per-user history and trim. We store the original user turn and
    # the final assistant turn only (NOT every tool round-trip) — that keeps
    # follow-up context useful without ballooning tokens on the next message.
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": final_text})
    if len(history) > HISTORY_LIMIT:
        del history[: len(history) - HISTORY_LIMIT]

    return final_text


# ===========================================================================
# Markdown -> Telegram HTML
# ===========================================================================
# Telegram supports a very small HTML subset (b, i, u, s, code, pre, a,
# blockquote). Claude tends to emit GitHub-flavored markdown (**bold**,
# ## headers, [text](url)). We convert here so the verdict renders cleanly.
def markdown_to_telegram_html(text: str) -> str:
    # 1. Escape source HTML chars (but keep quotes literal in body text).
    text = html_lib.escape(text, quote=False)

    # 2. Code fences ```...``` (multi-line), then inline `...`. Doing fences
    #    first prevents the inline rule from chewing on triple backticks.
    text = re.sub(r"```([^`]*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)

    # 3. Headers: strip leading "#"s. If the line is already wrapped in
    #    **bold**, leave the asterisks for the bold rule below; otherwise
    #    wrap the header text in <b> ourselves.
    def header_repl(m: re.Match) -> str:
        line = m.group(1).strip()
        if line.startswith("**") and line.endswith("**"):
            return line
        return f"<b>{line}</b>"

    text = re.sub(r"^#{1,6}\s+(.+)$", header_repl, text, flags=re.MULTILINE)

    # 4. Bold **...**, then italic *...*. Bold runs first so the italic rule
    #    (which is single-asterisk) doesn't grab halves of a bold pair.
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)

    # 5. Links [label](url).
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)

    return text


# ===========================================================================
# Telegram glue
# ===========================================================================
WELCOME = (
    "🎭 <b>The Truth Committee is in session.</b>\n\n"
    "Throw me a controversial claim, a suspicious statistic, or a question "
    "with numbers in it. My Investigator will dig, my Analyst will sneer, "
    "and my Calculator will keep us honest. Then I'll deliver a verdict.\n\n"
    "<i>Speak.</i>"
)


async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML)


async def _safe_reply(update: Update, html: str) -> None:
    """Send HTML; fall back to plaintext if Telegram rejects the markup."""
    try:
        await update.message.reply_text(
            html, parse_mode=ParseMode.HTML, disable_web_page_preview=True
        )
    except Exception:
        log.exception("HTML reply failed; falling back to plain text")
        # Strip tags crudely for the fallback so the user still gets the content.
        plain = re.sub(r"<[^>]+>", "", html)
        await update.message.reply_text(plain)


async def handle_message(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Top-level Telegram handler. One per incoming user message."""
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # Per-chat lock: prevents two rapid messages from the same user from
    # interleaving their history mutations. Different chats run in parallel.
    lock = CHAT_LOCKS[chat_id]
    async with lock:
        await update.message.chat.send_action(ChatAction.TYPING)

        async def send_status(html: str) -> None:
            await _safe_reply(update, html)

        try:
            verdict = await run_committee(chat_id, user_text, send_status)
        except Exception:
            log.exception("run_committee crashed")
            await update.message.reply_text(
                "💥 The committee descended into chaos. Try again."
            )
            return

        # Convert Claude's markdown verdict to Telegram HTML, then send in
        # 4000-char chunks (Telegram's per-message cap is 4096).
        formatted = markdown_to_telegram_html(verdict)
        for chunk in _chunk(formatted, 4000):
            await _safe_reply(update, chunk)


def _chunk(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


def main() -> None:
    # concurrent_updates=True is essential: without it, PTB processes updates
    # one-at-a-time across the whole bot, so user B has to wait for user A's
    # entire ReAct loop to finish before their first byte gets handled. With
    # it on, each update is dispatched as its own asyncio task and different
    # chats run in true parallel. Per-chat history is still safe because each
    # chat_id has its own asyncio.Lock in CHAT_LOCKS.
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Truth Committee bot starting (polling mode, concurrent updates)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
