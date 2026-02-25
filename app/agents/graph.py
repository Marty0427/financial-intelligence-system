"""LangGraph StateGraph — wires supervisor + parallel specialist agents + synthesizer."""

from __future__ import annotations

import logging
from typing import Union

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from app.agents.market_data import market_data_node
from app.agents.news import news_node
from app.agents.rag import rag_node
from app.agents.sentiment import sentiment_node
from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.config import settings

logger = logging.getLogger(__name__)

# ── Synthesizer ───────────────────────────────────────────────────────────────

_SYNTHESIZER_SYSTEM_PROMPT = """\
You are the final synthesiser in a financial intelligence multi-agent system.
You have been provided with data from multiple specialist agents:
- Market Data agent: live stock data, financials, technical indicators
- News agent: recent news articles and web search results
- RAG agent: extracts from SEC 10-K/10-Q filings
- Sentiment agent: FinBERT sentiment scores and Fear & Greed indicators

Your task: write a comprehensive, well-structured answer to the user's original query.

Guidelines:
- Integrate ALL available data — do not ignore any agent's output.
- Organise the response with clear sections (use markdown headers).
- For each data point, briefly indicate its source (market data / SEC filing / news / sentiment).
- If an agent reported an error or had no data, note it briefly but keep it concise.
- Include a brief "Key Takeaways" bullet list at the end.
- Do NOT provide investment advice or recommendations to buy/sell.
- Be precise about dates, figures, and sources.
- Response length: comprehensive but concise — aim for 400–800 words.
"""

_synthesizer_llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=0.1,
    api_key=settings.openai_api_key,
)


async def synthesize_node(state: AgentState) -> dict:
    """
    Final node: combines all agent outputs into a single coherent answer.
    """
    query = state["query"]
    ticker = state.get("ticker")
    company = state.get("company")
    market_data = state.get("market_data")
    news_results = state.get("news_results", [])
    rag_context = state.get("rag_context", [])
    sentiment = state.get("sentiment")
    agent_errors = state.get("agent_errors", [])

    # Build the synthesis prompt
    parts = [f"## User Query\n{query}\n"]

    if ticker or company:
        parts.append(f"**Company:** {company or ''} **Ticker:** {ticker or ''}\n")

    if market_data:
        parts.append(f"## Market Data Agent Output\n{market_data.get('raw', market_data)}\n")
    else:
        parts.append("## Market Data Agent Output\n_No data available._\n")

    if news_results:
        news_text = "\n\n".join(
            item.get("content", str(item)) for item in news_results if item
        )
        parts.append(f"## News Agent Output\n{news_text[:3000]}\n")
    else:
        parts.append("## News Agent Output\n_No news data available._\n")

    if rag_context:
        rag_text = "\n\n---\n\n".join(rag_context)
        parts.append(f"## RAG Agent Output (SEC Filings)\n{rag_text[:3000]}\n")
    else:
        parts.append("## RAG Agent Output\n_No SEC filing data available._\n")

    if sentiment:
        parts.append(f"## Sentiment Agent Output\n{sentiment.get('analysis', sentiment)}\n")
    else:
        parts.append("## Sentiment Agent Output\n_No sentiment data available._\n")

    if agent_errors:
        parts.append(f"## Agent Errors (non-fatal)\n" + "\n".join(f"- {e}" for e in agent_errors) + "\n")

    synthesis_prompt = "\n".join(parts)

    try:
        response = await _synthesizer_llm.ainvoke(
            [
                SystemMessage(content=_SYNTHESIZER_SYSTEM_PROMPT),
                ("human", synthesis_prompt),
            ]
        )
        final_answer = response.content
        logger.info("Synthesizer completed — %d chars", len(final_answer))
        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
        }
    except Exception as exc:
        logger.error("Synthesizer failed: %s", exc)
        fallback = (
            f"I encountered an error while synthesising the final answer: {exc}\n\n"
            f"Raw data collected:\n{synthesis_prompt[:2000]}"
        )
        return {
            "final_answer": fallback,
            "messages": [AIMessage(content=fallback)],
        }


# ── Routing function ──────────────────────────────────────────────────────────

_AGENT_NODES = {"market_data", "news", "rag", "sentiment"}


def route_to_agents(state: AgentState) -> list[Union[Send, str]]:
    """
    Fan-out: send execution to each selected agent in parallel using LangGraph Send.
    Falls back to all agents if selection is empty.
    """
    selected = state.get("selected_agents") or list(_AGENT_NODES)
    valid = [a for a in selected if a in _AGENT_NODES]

    if not valid:
        logger.warning("No valid agents selected, defaulting to all")
        valid = list(_AGENT_NODES)

    logger.info("Routing to agents: %s", valid)
    return [Send(agent, state) for agent in valid]


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the financial intelligence LangGraph."""
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("market_data", market_data_node)
    builder.add_node("news", news_node)
    builder.add_node("rag", rag_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("synthesize", synthesize_node)

    # Edges
    builder.add_edge(START, "supervisor")

    # Supervisor fans out to selected agents in parallel
    builder.add_conditional_edges(
        "supervisor",
        route_to_agents,
        ["market_data", "news", "rag", "sentiment"],
    )

    # All specialist agents converge at the synthesizer
    for agent_node in ["market_data", "news", "rag", "sentiment"]:
        builder.add_edge(agent_node, "synthesize")

    builder.add_edge("synthesize", END)

    return builder.compile()


# Module-level compiled graph (lazy singleton — initialised in app startup)
_graph = None


def get_graph():
    """Return (or lazily build) the compiled graph singleton."""
    global _graph  # noqa: PLW0603
    if _graph is None:
        _graph = build_graph()
        logger.info("Financial intelligence graph compiled successfully")
    return _graph
