"""Query endpoints — POST /query and POST /query/stream (SSE)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agents.state import initial_state
from app.dependencies import get_chroma_retriever, get_financial_graph

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Financial question or research query")
    thread_id: Optional[str] = Field(default=None, description="Optional conversation thread ID for multi-turn")


class AgentDataSummary(BaseModel):
    market_data_available: bool
    news_articles_found: int
    rag_chunks_retrieved: int
    sentiment_available: bool
    agent_errors: list[str]


class QueryResponse(BaseModel):
    query: str
    answer: str
    ticker: Optional[str] = None
    company: Optional[str] = None
    agents_used: list[str]
    data_summary: AgentDataSummary


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", response_model=QueryResponse, summary="Run financial intelligence query")
async def run_query(
    request: QueryRequest,
    graph=Depends(get_financial_graph),
    _retriever=Depends(get_chroma_retriever),  # ensures retriever is injected
) -> QueryResponse:
    """
    Submit a financial question to the multi-agent system.

    The supervisor analyses the query and fans out to relevant specialist agents
    (market data, news, RAG over SEC filings, sentiment) whose results are
    combined into a single synthesised answer.
    """
    logger.info("Query received: %r", request.query[:100])

    state = initial_state(request.query)
    config = {"configurable": {"thread_id": request.thread_id or "default"}}

    try:
        result = await graph.ainvoke(state, config=config)
    except Exception as exc:
        logger.error("Graph invocation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {exc}") from exc

    return QueryResponse(
        query=request.query,
        answer=result.get("final_answer") or "No answer generated.",
        ticker=result.get("ticker"),
        company=result.get("company"),
        agents_used=result.get("selected_agents", []),
        data_summary=AgentDataSummary(
            market_data_available=result.get("market_data") is not None,
            news_articles_found=len(result.get("news_results", [])),
            rag_chunks_retrieved=len(result.get("rag_context", [])),
            sentiment_available=result.get("sentiment") is not None,
            agent_errors=result.get("agent_errors", []),
        ),
    )


@router.post("/stream", summary="Stream financial intelligence query (SSE)")
async def stream_query(
    request: QueryRequest,
    graph=Depends(get_financial_graph),
    _retriever=Depends(get_chroma_retriever),
):
    """
    Stream the final answer token-by-token via Server-Sent Events.

    Emits two event types:
    - `status`: agent progress updates (e.g. "Running market_data agent…")
    - `token`: individual text tokens of the final answer
    - `done`: final metadata JSON
    """
    state = initial_state(request.query)
    config = {"configurable": {"thread_id": request.thread_id or "stream"}}

    async def event_generator():
        try:
            async for event in graph.astream_events(state, config=config, version="v2"):
                kind = event.get("event")
                name = event.get("name", "")

                # Emit status when a node starts
                if kind == "on_chain_start" and name in ("supervisor", "market_data", "news", "rag", "sentiment", "synthesize"):
                    yield {
                        "event": "status",
                        "data": json.dumps({"agent": name, "status": "running"}),
                    }

                # Stream tokens from the synthesizer
                if kind == "on_chat_model_stream" and "synthesize" in event.get("tags", []):
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield {
                            "event": "token",
                            "data": json.dumps({"text": chunk.content}),
                        }

                # Node completion
                if kind == "on_chain_end" and name in ("market_data", "news", "rag", "sentiment"):
                    yield {
                        "event": "status",
                        "data": json.dumps({"agent": name, "status": "done"}),
                    }

            yield {
                "event": "done",
                "data": json.dumps({"message": "Stream complete"}),
            }
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc)}),
            }

    return EventSourceResponse(event_generator())
