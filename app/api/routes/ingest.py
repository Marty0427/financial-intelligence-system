"""Ingest endpoint — POST /ingest (trigger SEC filing ingestion)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.ingestion.sec_pipeline import ingest_multiple_tickers, ingest_ticker
from app.ingestion.vectorstore import get_collection_stats

router = APIRouter(prefix="/ingest", tags=["ingestion"])
logger = logging.getLogger(__name__)

# Track running ingestion jobs (in-memory; production would use Redis / DB)
_running_jobs: dict[str, str] = {}  # ticker -> status


# ── Models ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    tickers: list[str] = Field(
        ...,
        min_length=1,
        description="List of ticker symbols to ingest (e.g. ['AAPL', 'MSFT'])",
    )
    forms: list[str] = Field(
        default=["10-K", "10-Q"],
        description="SEC form types to download",
    )
    num_filings: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of filings per form type per ticker",
    )


class IngestResponse(BaseModel):
    message: str
    tickers: list[str]
    job_id: str


class IngestStatusResponse(BaseModel):
    results: list[dict]
    vectorstore_stats: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", response_model=IngestResponse, summary="Trigger SEC filing ingestion")
async def start_ingestion(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Download SEC 10-K/10-Q filings for the specified tickers, parse them,
    and index them in the Chroma vector store.

    This runs as a background task. Check vectorstore stats at GET /companies.
    """
    tickers = [t.upper().strip() for t in request.tickers if t.strip()]
    if not tickers:
        raise HTTPException(status_code=422, detail="No valid tickers provided")

    job_id = f"ingest-{'_'.join(tickers[:3])}"

    for ticker in tickers:
        _running_jobs[ticker] = "queued"

    async def run_ingestion():
        for ticker in tickers:
            _running_jobs[ticker] = "running"
            try:
                result = await ingest_ticker(
                    ticker=ticker,
                    forms=request.forms,
                    num_filings=request.num_filings,
                )
                _running_jobs[ticker] = f"done:{result.get('total_chunks_indexed', 0)} chunks"
                logger.info("Ingestion complete for %s: %s", ticker, result)
            except Exception as exc:
                _running_jobs[ticker] = f"error:{exc}"
                logger.error("Ingestion failed for %s: %s", ticker, exc)

    background_tasks.add_task(run_ingestion)

    return IngestResponse(
        message=f"Ingestion started for {len(tickers)} ticker(s) in background.",
        tickers=tickers,
        job_id=job_id,
    )


@router.get("/status", summary="Check ingestion job status")
async def get_ingestion_status() -> dict:
    """Return the status of all ingestion jobs since server start."""
    return {"jobs": dict(_running_jobs)}


@router.post("/sync", summary="Synchronous ingestion (blocks until complete)")
async def ingest_sync(request: IngestRequest) -> IngestStatusResponse:
    """
    Run ingestion synchronously and return results immediately.
    Use for small batches (1–2 tickers) or testing purposes.
    """
    tickers = [t.upper().strip() for t in request.tickers if t.strip()]
    if not tickers:
        raise HTTPException(status_code=422, detail="No valid tickers provided")
    if len(tickers) > 5:
        raise HTTPException(status_code=422, detail="Synchronous ingestion limited to 5 tickers. Use POST /ingest for larger batches.")

    results = await ingest_multiple_tickers(
        tickers=tickers,
        forms=request.forms,
        num_filings=request.num_filings,
    )
    stats = get_collection_stats()

    return IngestStatusResponse(results=results, vectorstore_stats=stats)
