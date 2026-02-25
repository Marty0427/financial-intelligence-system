"""Supervisor agent node — query decomposition and agent routing."""

from __future__ import annotations

import logging
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.config import settings

logger = logging.getLogger(__name__)


# ── Structured output schema ───────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    """Structured routing decision produced by the supervisor."""

    ticker: Optional[str] = Field(
        default=None,
        description="Stock ticker symbol extracted from the query (e.g. 'AAPL'). None if no specific company.",
    )
    company: Optional[str] = Field(
        default=None,
        description="Company name extracted from the query (e.g. 'Apple Inc.'). None if no specific company.",
    )
    selected_agents: list[Literal["market_data", "news", "rag", "sentiment"]] = Field(
        description=(
            "Which specialist agents to invoke. Rules:\n"
            "- Always include 'market_data' when a specific ticker/company is mentioned.\n"
            "- Always include 'news' for questions about recent events, earnings, announcements.\n"
            "- Always include 'rag' when the query asks about filings, annual reports, risk factors, management commentary, or financial statements from SEC documents.\n"
            "- Always include 'sentiment' when asking about investor sentiment, market mood, or fear/greed.\n"
            "- Include all four agents for comprehensive company analysis questions."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining which agents were selected and why."
    )


# ── LLM with structured output ─────────────────────────────────────────────────

_llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=0,
    api_key=settings.openai_api_key,
)

_supervisor_llm = _llm.with_structured_output(SupervisorDecision)


# ── System prompt ──────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """\
You are the supervisor of a financial intelligence multi-agent system.
Your job is to analyse the user's query and decide which specialist agents to invoke.

Available agents:
- market_data  → live stock prices, financials (income/balance/cash-flow), technical indicators (RSI, MACD, SMA)
- news         → recent news articles (NewsAPI) and real-time web search (Tavily)
- rag          → semantic search over SEC 10-K/10-Q annual/quarterly reports stored in a vector DB
- sentiment    → FinBERT sentiment analysis on news + Fear & Greed proxy indicator

Selection rules:
1. Specific company/stock query → always select market_data.
2. "Recent", "latest", "current", "today", "this week" → always include news.
3. "Annual report", "10-K", "risk factors", "management discussion", "revenue segments", "filings" → include rag.
4. "Sentiment", "investor mood", "bullish/bearish", "fear/greed" → include sentiment.
5. Broad company analysis ("Tell me everything about X") → include all four.
6. Macro/index queries (e.g. "S&P 500") → news + market_data only.

Extract ticker and company name if present. Ticker should be the official NYSE/NASDAQ symbol.
"""


# ── Node function ─────────────────────────────────────────────────────────────

async def supervisor_node(state: AgentState) -> dict:
    """
    LangGraph node: analyses the query and populates `selected_agents`,
    `ticker`, and `company` in the shared state.
    """
    query = state["query"]
    logger.info("Supervisor analysing query: %r", query[:120])

    try:
        decision: SupervisorDecision = await _supervisor_llm.ainvoke(
            [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
        )

        logger.info(
            "Supervisor decision — agents=%s ticker=%s company=%s reason=%r",
            decision.selected_agents,
            decision.ticker,
            decision.company,
            decision.reasoning,
        )

        return {
            "ticker": decision.ticker,
            "company": decision.company,
            "selected_agents": decision.selected_agents,
            "messages": [HumanMessage(content=query)],
        }

    except Exception as exc:
        logger.error("Supervisor node failed: %s", exc)
        # Safe fallback: invoke all agents
        return {
            "ticker": None,
            "company": None,
            "selected_agents": ["market_data", "news", "rag", "sentiment"],
            "agent_errors": [f"supervisor_error: {exc}"],
            "messages": [HumanMessage(content=query)],
        }
