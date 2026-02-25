"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.middleware import configure_logging, register_middleware
from app.api.routes import health, ingest, query
from app.config import settings
from app.dependencies import get_chroma_retriever, get_financial_graph

logger = structlog.get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load graph and retriever. Shutdown: nothing special."""
    configure_logging()

    logger.info("Starting Financial Intelligence System…")

    # Pre-build graph (compiles LangGraph StateGraph)
    graph = get_financial_graph()
    logger.info("LangGraph compiled — nodes: %s", list(graph.nodes.keys()))

    # Pre-connect Chroma and inject retriever into RAG tools
    retriever = get_chroma_retriever()
    logger.info("Chroma retriever ready")

    logger.info(
        "Server ready — model=%s chroma_dir=%s",
        settings.openai_model,
        settings.chroma_persist_dir,
    )

    yield  # Application is running

    logger.info("Financial Intelligence System shutting down")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial Intelligence Multi-Agent System",
        description=(
            "Production-grade multi-agent financial intelligence platform powered by "
            "LangGraph + LangChain. Combines live market data (Yahoo Finance, Alpha Vantage), "
            "real-time news (NewsAPI, Tavily), SEC filing RAG (Chroma), and sentiment analysis "
            "(FinBERT) into coherent answers via a Supervisor + Specialist Agents architecture."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    register_middleware(app)

    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(health.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    return app


app = create_app()
