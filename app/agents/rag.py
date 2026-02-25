"""RAG agent node — semantic retrieval from SEC 10-K/10-Q filings."""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agents.state import AgentState
from app.config import settings
from app.tools.rag_tools import RAG_TOOLS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a specialist SEC Filings RAG agent in a financial intelligence system.
You have access to a vector database containing SEC 10-K (annual) and 10-Q (quarterly) filings.

Your task: retrieve and synthesise the most relevant passages from SEC filings to answer the query.

Strategy:
1. Call retrieve_sec_context with a precise semantic query. Include the company ticker if known.
2. If the first query returns insufficient context, refine and call again with a more specific query.
3. Synthesise the retrieved passages into a structured answer covering:
   - Revenue trends and segment breakdown
   - Profitability (gross/operating/net margin)
   - Risk factors relevant to the query
   - Forward guidance / management outlook
   - Capital structure (debt, cash)

Always cite the filing (ticker, form type, date) when referencing specific data.
If no relevant filings are indexed, say so clearly.
"""

_llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=0,
    api_key=settings.openai_api_key,
)

_agent = create_react_agent(
    model=_llm,
    tools=RAG_TOOLS,
    state_modifier=SystemMessage(content=_SYSTEM_PROMPT),
)


async def rag_node(state: AgentState) -> dict:
    """
    LangGraph node: invokes the RAG ReAct agent and appends retrieved
    context to state["rag_context"].
    """
    ticker = state.get("ticker")
    query = state["query"]

    task = (
        f"Query: {query}\n"
        f"Ticker: {ticker or 'not specified'}\n\n"
        "Retrieve and summarise the most relevant SEC filing information for this query."
    )

    try:
        result = await _agent.ainvoke({"messages": [("human", task)]})
        last_message = result["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        logger.info("RAG agent completed (ticker=%s)", ticker)
        return {"rag_context": [content]}
    except Exception as exc:
        logger.error("RAG agent failed: %s", exc)
        return {
            "rag_context": [f"RAG agent error: {exc}"],
            "agent_errors": [f"rag_agent: {exc}"],
        }
