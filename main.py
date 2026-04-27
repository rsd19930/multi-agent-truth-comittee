"""
Tech News Pipeline - Multi-Agent System using CrewAI

A two-agent crew where a Researcher gathers facts via web search,
and a Writer turns those facts into a short blog post.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Loads ANTHROPIC_API_KEY from a .env file in this folder.
# >>> PUT YOUR ANTHROPIC API KEY in the .env file next to this script <<<
load_dotenv(override=True)

if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Add it to the .env file in this folder."
    )

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# CrewAI uses LiteLLM under the hood. The "anthropic/" prefix routes to the
# Anthropic API. claude-haiku-4-5 is the latest Haiku model.
llm = LLM(
    model="anthropic/claude-haiku-4-5",
    temperature=0.3,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# CrewAI expects tools that subclass crewai.tools.BaseTool. We wrap the
# LangChain DuckDuckGoSearchRun tool so the Researcher agent can use it.
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
# Agent 1: Researcher — has the search tool, does not delegate.
researcher = Agent(
    role="Senior Tech Researcher",
    goal="Find the latest, most accurate news about AI agents.",
    backstory=(
        "You are a seasoned technology researcher with a knack for cutting "
        "through hype and surfacing concrete, well-sourced facts about "
        "emerging AI systems. You favor primary sources and recent reporting."
    ),
    tools=[search_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Agent 2: Writer — no tools, turns research notes into a polished post.
writer = Agent(
    role="Tech Content Strategist",
    goal=(
        "Turn raw research notes into engaging, accurate blog posts that "
        "non-experts can enjoy and learn from."
    ),
    backstory=(
        "You are a tech content strategist who has written for major "
        "publications. You translate dense technical material into crisp, "
        "engaging prose without losing fidelity."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
# Task 1: Research Anthropic's Claude Managed Agents and surface 3 key facts.
research_task = Task(
    description=(
        "Research Anthropic's 'Claude Managed Agents' offering. "
        "Use the DuckDuckGo search tool to find recent, reliable information. "
        "Identify exactly 3 key facts: what it is, how it works, and why it "
        "matters to developers building agentic systems."
    ),
    expected_output=(
        "A bulleted list of 3 concise, well-sourced facts about Claude "
        "Managed Agents, each 1-3 sentences, with the source URL where "
        "possible."
    ),
    agent=researcher,
)

# Task 2: Use the researcher's notes to draft a 2-paragraph blog post.
writing_task = Task(
    description=(
        "Using the research notes from the previous task, write an engaging "
        "blog post about Anthropic's Claude Managed Agents. "
        "The post must be exactly 2 paragraphs, accessible to a technical "
        "but non-specialist reader, and faithful to the source facts."
    ),
    expected_output=(
        "A 2-paragraph blog post (no headings, no bullet lists) with a "
        "compelling opening line and a clear takeaway in the closing line."
    ),
    agent=writer,
    context=[research_task],  # explicitly chain the researcher's output in
)

# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------
# Sequential process: research_task runs first, then writing_task receives
# its output via the `context` link above.
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n" + "=" * 60)
    print("FINAL BLOG POST")
    print("=" * 60)
    print(result)
