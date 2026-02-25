"""Health and metadata endpoints — GET /health, GET /companies."""

from __future__ import annotations

import logging

import yfinance as yf
from fastapi import APIRouter

from app.config import settings
from app.ingestion.vectorstore import get_collection_stats

router = APIRouter(tags=["meta"])
logger = logging.getLogger(__name__)


@router.get("/health", summary="System health check")
async def health_check() -> dict:
    """
    Check connectivity to all external services:
    - OpenAI API
    - Chroma vector store
    - yfinance (market data)
    - NewsAPI (if configured)
    - Tavily (if configured)
    """
    checks: dict[str, dict] = {}

    # ── OpenAI ────────────────────────────────────────────────────────────────
    try:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
        await llm.ainvoke("ping")
        checks["openai"] = {"status": "ok", "model": settings.openai_model}
    except Exception as exc:
        checks["openai"] = {"status": "error", "detail": str(exc)[:200]}

    # ── Chroma ────────────────────────────────────────────────────────────────
    try:
        stats = get_collection_stats()
        checks["chroma"] = {"status": "ok", **stats}
    except Exception as exc:
        checks["chroma"] = {"status": "error", "detail": str(exc)[:200]}

    # ── yfinance ──────────────────────────────────────────────────────────────
    try:
        price = yf.Ticker("AAPL").info.get("currentPrice")
        checks["yfinance"] = {"status": "ok", "test_ticker": "AAPL", "price": price}
    except Exception as exc:
        checks["yfinance"] = {"status": "error", "detail": str(exc)[:200]}

    # ── NewsAPI ───────────────────────────────────────────────────────────────
    if settings.news_api_key:
        try:
            from newsapi import NewsApiClient  # noqa: PLC0415

            client = NewsApiClient(api_key=settings.news_api_key)
            client.get_top_headlines(language="en", page_size=1)
            checks["newsapi"] = {"status": "ok"}
        except Exception as exc:
            checks["newsapi"] = {"status": "error", "detail": str(exc)[:200]}
    else:
        checks["newsapi"] = {"status": "not_configured"}

    # ── Tavily ────────────────────────────────────────────────────────────────
    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient  # noqa: PLC0415

            client = TavilyClient(api_key=settings.tavily_api_key)
            client.search("financial markets", max_results=1)
            checks["tavily"] = {"status": "ok"}
        except Exception as exc:
            checks["tavily"] = {"status": "error", "detail": str(exc)[:200]}
    else:
        checks["tavily"] = {"status": "not_configured"}

    # ── Alpha Vantage ─────────────────────────────────────────────────────────
    checks["alpha_vantage"] = (
        {"status": "configured"} if settings.alpha_vantage_api_key else {"status": "not_configured"}
    )

    overall = "healthy" if all(
        v.get("status") in ("ok", "not_configured", "configured")
        for v in checks.values()
    ) else "degraded"

    return {"status": overall, "services": checks}


@router.get("/companies", summary="List indexed companies in SEC vector store")
async def list_companies() -> dict:
    """Return all companies/tickers whose SEC filings have been indexed in Chroma."""
    stats = get_collection_stats()
    return {
        "total_documents": stats.get("total_documents", 0),
        "indexed_tickers": stats.get("tickers", []),
        "form_types_available": stats.get("form_types", []),
    }
