"""LangGraph state definition for the financial intelligence multi-agent system."""

from __future__ import annotations

from operator import add
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── Custom reducers ────────────────────────────────────────────────────────────

def merge_optional_dict(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    """Merge two optional dicts; later values from b overwrite a."""
    if a is None:
        return b
    if b is None:
        return a
    return {**a, **b}


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(dict):
    """
    Shared state propagated through the LangGraph graph.

    The graph fans out to multiple specialist agents in parallel using Send().
    Each agent writes only to its own slice of state; the reducers merge results
    produced concurrently by different agents.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    """Original user query."""

    ticker: Optional[str]
    """Extracted stock ticker symbol (e.g. 'AAPL')."""

    company: Optional[str]
    """Extracted company name (e.g. 'Apple')."""

    # ── Routing ───────────────────────────────────────────────────────────────
    selected_agents: list[str]
    """
    List of agent node names the supervisor chose to invoke.
    Values: 'market_data' | 'news' | 'rag' | 'sentiment'
    """

    # ── Agent outputs (parallel, merged by reducers) ───────────────────────────
    market_data: Annotated[Optional[dict], merge_optional_dict]
    """Stock quote, financials, technical indicators from Market Data agent."""

    news_results: Annotated[list[dict], add]
    """News articles and web-search snippets from News agent."""

    rag_context: Annotated[list[str], add]
    """Text chunks retrieved from the SEC filings vector store."""

    sentiment: Annotated[Optional[dict], merge_optional_dict]
    """FinBERT / LLM sentiment scores from Sentiment agent."""

    # ── Error tracking ────────────────────────────────────────────────────────
    agent_errors: Annotated[list[str], add]
    """Non-fatal errors from individual agents (used in synthesis prompt)."""

    # ── Final output ──────────────────────────────────────────────────────────
    final_answer: Optional[str]
    """Synthesised plain-text answer written by the Synthesize node."""

    # ── Message history ───────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    """Full conversation messages (supports multi-turn via checkpointer)."""


def initial_state(query: str) -> AgentState:
    """Return a minimal AgentState for a fresh query."""
    return AgentState(  # type: ignore[call-arg]
        query=query,
        ticker=None,
        company=None,
        selected_agents=[],
        market_data=None,
        news_results=[],
        rag_context=[],
        sentiment=None,
        agent_errors=[],
        final_answer=None,
        messages=[],
    )
