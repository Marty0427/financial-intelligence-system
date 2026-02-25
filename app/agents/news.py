"""News agent node — NewsAPI + Tavily web search."""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agents.state import AgentState
from app.config import settings
from app.tools.news_tools import NEWS_TOOLS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a specialist News & Research agent in a financial intelligence system.
Your task: find and summarise the most relevant recent news and web content for the query.

Strategy:
1. Use search_financial_news with a targeted query (include company name + topic).
2. Use tavily_web_search for any real-time data or analyst commentary.
3. If the query is about general market conditions, also call get_top_financial_headlines.

Return a structured list of key findings — title, source, date, and a 1-2 sentence summary of why it's relevant.
Be objective; do not editorialize or provide investment advice.
"""

_llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=0,
    api_key=settings.openai_api_key,
)

_agent = create_react_agent(
    model=_llm,
    tools=NEWS_TOOLS,
    state_modifier=SystemMessage(content=_SYSTEM_PROMPT),
)


async def news_node(state: AgentState) -> dict:
    """
    LangGraph node: invokes the News ReAct agent and appends results
    to state["news_results"].
    """
    company = state.get("company") or state.get("ticker") or ""
    query = state["query"]

    task = (
        f"Query: {query}\n"
        f"Company/Ticker: {company or 'general market query'}\n\n"
        "Find the most relevant recent news and web content. "
        "Search for both the specific company and the topic mentioned in the query."
    )

    try:
        result = await _agent.ainvoke({"messages": [("human", task)]})
        last_message = result["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        logger.info("News agent completed (company=%s)", company)
        return {"news_results": [{"source": "news_agent", "content": content}]}
    except Exception as exc:
        logger.error("News agent failed: %s", exc)
        return {
            "news_results": [{"error": str(exc)}],
            "agent_errors": [f"news_agent: {exc}"],
        }
