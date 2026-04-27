"""
The Truth Committee — Hierarchical Multi-Agent Crew

Investigates the real-world impact of Humanoid Robotics in manufacturing by 2027.
Three agents collaborate under a managing Chief Editor:
  - Lead Investigator: finds breakthroughs and adoption stats.
  - Skeptical Analyst: surfaces economic risks, labor concerns, failure modes.
  - Chief Editor: delegates, critiques, and iterates until satisfied.

Workflow is hierarchical (Process.hierarchical), so the Chief Editor decides
who does what next and can loop back to the Investigator for revisions.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Loads ANTHROPIC_API_KEY from the .env file in this folder.
# override=True ensures the .env value beats any (possibly empty) shell var.
load_dotenv(override=True)

if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Add it to the .env file in this folder."
    )

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Latest Haiku via the native Anthropic provider in CrewAI.
# A single LLM instance is shared by all agents, including the manager.
llm = LLM(
    model="anthropic/claude-haiku-4-5",
    temperature=0.3,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# Wrap LangChain's DuckDuckGo search in a CrewAI BaseTool so it's usable by
# CrewAI agents. Both worker agents get this — the Editor does not search.
class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = (
        "Searches the web using DuckDuckGo. "
        "Input should be a search query string. Returns text snippets."
    )

    def _run(self, query: str) -> str:
        return DuckDuckGoSearchRun().run(query)


search_tool = DuckDuckGoSearchTool()

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
# Lead Investigator — pro-side researcher. allow_delegation=False because the
# investigator is a worker, not a delegator; delegation is the Editor's job.
lead_investigator = Agent(
    role="Lead Investigator",
    goal=(
        "Uncover concrete technical breakthroughs, deployment milestones, "
        "and adoption statistics for humanoid robotics in manufacturing "
        "through 2027. Cite sources for every claim."
    ),
    backstory=(
        "You are a meticulous industry analyst with deep contacts in robotics "
        "labs and OEMs. You favor primary sources, vendor disclosures, and "
        "quantitative data over hype. You attach a source URL or publication "
        "to every fact and flag any claim you cannot independently verify."
    ),
    tools=[search_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Skeptical Analyst — adversarial reviewer. Also a worker, no delegation.
skeptical_analyst = Agent(
    role="Skeptical Analyst",
    goal=(
        "Stress-test the Investigator's findings. Surface economic risks, "
        "labor union concerns, regulatory headwinds, and technical failure "
        "points. Flag unverified, cherry-picked, or vendor-marketing claims."
    ),
    backstory=(
        "You are a former auditor turned tech-skeptic. Your craft is finding "
        "what's missing: the costs hidden behind glossy demos, the failure "
        "modes that don't make press releases, and the labor-displacement "
        "stories that vendors prefer not to discuss. You assign a Confidence "
        "Score (0.0-1.0) to each claim based on source quality and corroboration."
    ),
    tools=[search_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Chief Editor — manager agent. allow_delegation=True so it can route work
# to the Investigator and Analyst, and loop back for revisions when the
# Analyst flags gaps. The manager has no tools of its own — it directs.
chief_editor = Agent(
    role="Chief Editor",
    goal=(
        "Produce a balanced Executive Risk-Benefit Report on humanoid "
        "robotics in manufacturing by 2027. Delegate research to the "
        "Lead Investigator, then a critique to the Skeptical Analyst, "
        "and iterate until every claim is well-sourced and balanced."
    ),
    backstory=(
        "You are a veteran investigative editor who runs newsroom-style "
        "review cycles. You will not sign off on a report that contains "
        "unverified claims or one-sided framing. When the Analyst flags "
        "gaps, you send the work back to the Investigator with specific "
        "instructions, and you keep iterating until the report is balanced "
        "and every claim carries a Confidence Score."
    ),
    allow_delegation=True,
    verbose=True,
    llm=llm,
)

# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
# In a hierarchical crew, you typically define the high-level deliverable as
# one task assigned to (or owned by) the manager. The manager then plans the
# sub-tasks and delegates to the worker agents using CrewAI's built-in
# delegation tools. We do NOT pre-assign sub-tasks to specific agents; the
# Chief Editor decides the routing dynamically.
master_task = Task(
    description=(
        "Investigate the real-world impact of HUMANOID ROBOTICS IN "
        "MANUFACTURING BY 2027. Produce a balanced Executive Risk-Benefit "
        "Report.\n\n"
        "Required workflow (you, the Chief Editor, must orchestrate this):\n"
        "  1. Delegate to the Lead Investigator: gather technical "
        "     breakthroughs, deployment milestones, and adoption statistics, "
        "     each with a source.\n"
        "  2. Delegate to the Skeptical Analyst: critique the findings — "
        "     identify economic risks, labor union concerns, regulatory "
        "     issues, and technical failure modes; flag any unverified or "
        "     cherry-picked claims; assign a Confidence Score (0.0-1.0) to "
        "     each claim.\n"
        "  3. If the Analyst flags significant gaps or unverified claims, "
        "     send the task BACK to the Investigator with specific "
        "     instructions for what to verify or add, then re-run the "
        "     Analyst's critique. Iterate until you are satisfied.\n"
        "  4. Synthesize the final report yourself.\n\n"
        "Do not accept the first draft if it is one-sided, vendor-sourced, "
        "or missing Confidence Scores. Push back at least once."
    ),
    expected_output=(
        "An 'Executive Risk-Benefit Report' in Markdown with these sections:\n"
        "  1. Executive Summary (3-5 sentences, balanced).\n"
        "  2. Key Benefits & Breakthroughs — bulleted; each bullet ends with "
        "     `[Source: <url or publication>] [Confidence: 0.0-1.0]`.\n"
        "  3. Key Risks & Failure Modes — bulleted; same citation+confidence "
        "     format.\n"
        "  4. Net Outlook for 2027 — a balanced 1-paragraph judgment.\n"
        "  5. Methodology Note — one short paragraph describing how many "
        "     review iterations occurred and what was revised."
    ),
    # NOTE: Do NOT assign `agent=` here in hierarchical mode. CrewAI 1.14.3
    # has a quirk where, if a task has an explicit agent, the manager's
    # delegation tool is populated with only that agent as a "coworker" —
    # meaning the manager can only delegate to itself. Leaving `agent` unset
    # lets CrewAI route the task to the manager and expose ALL workers
    # (lead_investigator, skeptical_analyst) as valid delegation targets.
)

# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------
# Hierarchical process. Note the manager agent is passed via `manager_agent`
# and is NOT included in the `agents` list — CrewAI enforces this separation.
# The manager is automatically given delegation tools to route work to the
# worker agents, so it can loop back to the Investigator for revisions.
crew = Crew(
    agents=[lead_investigator, skeptical_analyst],
    tasks=[master_task],
    manager_agent=chief_editor,
    process=Process.hierarchical,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n" + "=" * 70)
    print("EXECUTIVE RISK-BENEFIT REPORT")
    print("=" * 70)
    print(result)
