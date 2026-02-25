"""Market Data agent node — wraps market tools in a ReAct agent."""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agents.state import AgentState
from app.config import settings
from app.tools.market_tools import MARKET_TOOLS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a specialist Market Data agent in a financial intelligence system.
Your task: gather comprehensive market data for the company/ticker requested.

Use the available tools in this order when a ticker is provided:
1. get_stock_quote → current price, market cap, P/E, analyst targets
2. get_financial_statements → income statement, balance sheet, cash flow (annual)
3. get_technical_indicators → RSI, MACD, SMA-50/200

Return a JSON-serialisable summary of ALL data you gathered.
If a tool fails, note the error and continue with the remaining tools.
Be factual. Do not provide investment advice.
"""

_llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=0,
    api_key=settings.openai_api_key,
)

_agent = create_react_agent(
    model=_llm,
    tools=MARKET_TOOLS,
    state_modifier=SystemMessage(content=_SYSTEM_PROMPT),
)


async def market_data_node(state: AgentState) -> dict:
    """
    LangGraph node: invokes the market data ReAct agent and stores results
    in state["market_data"].
    """
    ticker = state.get("ticker")
    company = state.get("company")
    query = state["query"]

    if not ticker and not company:
        logger.info("Market Data agent: no ticker/company in state, using query")

    task = (
        f"Query: {query}\n"
        f"Ticker: {ticker or 'unknown — infer from query if possible'}\n"
        f"Company: {company or 'unknown'}\n\n"
        "Fetch all available market data for this company/ticker."
    )

    try:
        result = await _agent.ainvoke({"messages": [("human", task)]})
        last_message = result["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        logger.info("Market Data agent completed (ticker=%s)", ticker)
        return {"market_data": {"raw": content, "ticker": ticker, "company": company}}
    except Exception as exc:
        logger.error("Market Data agent failed: %s", exc)
        return {
            "market_data": {"error": str(exc)},
            "agent_errors": [f"market_data_agent: {exc}"],
        }
