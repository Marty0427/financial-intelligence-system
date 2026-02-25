"""Sentiment agent node — FinBERT sentiment + Fear & Greed proxy."""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agents.state import AgentState
from app.config import settings
from app.tools.sentiment_tools import SENTIMENT_TOOLS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a specialist Sentiment Analysis agent in a financial intelligence system.
Your task: assess the market sentiment and investor mood for the given company/query.

Strategy:
1. If a ticker is provided, call get_fear_and_greed_proxy to get a quantitative sentiment score.
2. If news headlines or article snippets are available in the query context, call analyze_news_sentiment
   to classify them as positive / neutral / negative using FinBERT.
3. If FinBERT is unavailable, assess sentiment yourself based on the language in the query and any
   available context (use qualitative descriptors: bullish, neutral, bearish).

Return:
- Overall sentiment label (bullish / neutral / bearish)
- Quantitative scores where available
- Key sentiment drivers (what is pushing sentiment positive or negative)
- Any contrarian signals worth noting

Be objective. Acknowledge uncertainty where it exists.
"""

_llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=0,
    api_key=settings.openai_api_key,
)

_agent = create_react_agent(
    model=_llm,
    tools=SENTIMENT_TOOLS,
    state_modifier=SystemMessage(content=_SYSTEM_PROMPT),
)


async def sentiment_node(state: AgentState) -> dict:
    """
    LangGraph node: invokes the Sentiment ReAct agent and stores results
    in state["sentiment"].
    """
    ticker = state.get("ticker")
    company = state.get("company")
    query = state["query"]

    # Pass any already-retrieved news headlines as context
    news_snippets = []
    for item in state.get("news_results", []):
        content = item.get("content", "")
        if content and not item.get("error"):
            news_snippets.append(content[:300])

    task = (
        f"Query: {query}\n"
        f"Ticker: {ticker or 'not specified'}\n"
        f"Company: {company or 'not specified'}\n"
    )
    if news_snippets:
        task += f"\nRecent news snippets for context:\n" + "\n---\n".join(news_snippets[:5])

    task += "\n\nAssess market sentiment for this company/query."

    try:
        result = await _agent.ainvoke({"messages": [("human", task)]})
        last_message = result["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        logger.info("Sentiment agent completed (ticker=%s)", ticker)
        return {"sentiment": {"analysis": content, "ticker": ticker}}
    except Exception as exc:
        logger.error("Sentiment agent failed: %s", exc)
        return {
            "sentiment": {"error": str(exc)},
            "agent_errors": [f"sentiment_agent: {exc}"],
        }
