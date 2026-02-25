"""RAG retrieval tools — Chroma vector store over SEC 10-K/10-Q filings."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# The Chroma retriever is injected at graph-build time via a module-level variable
# so that we don't initialise Chroma on every tool call.
_retriever = None


def set_retriever(retriever) -> None:
    """Inject the Chroma retriever (called once during app startup)."""
    global _retriever  # noqa: PLW0603
    _retriever = retriever


@tool
def retrieve_sec_context(
    query: str,
    ticker: Optional[str] = None,
    top_k: int = 6,
) -> list[str]:
    """
    Retrieve relevant passages from SEC 10-K / 10-Q filings stored in Chroma.

    Use this tool to answer questions about a company's financial position,
    risk factors, management discussion, revenue segments, and forward guidance
    as stated in their official regulatory filings.

    Args:
        query: Semantic search query, e.g. 'revenue growth drivers and segment breakdown'.
        ticker: Optional ticker to filter results to a specific company (e.g. 'AAPL').
        top_k: Number of document chunks to retrieve (default 6).
    """
    if _retriever is None:
        logger.warning("RAG retriever not initialised — SEC context unavailable")
        return ["RAG retriever not initialised. Run the SEC ingestion pipeline first."]

    try:
        search_filter = {"ticker": ticker.upper()} if ticker else None

        if search_filter:
            docs = _retriever.vectorstore.similarity_search(
                query, k=top_k, filter=search_filter
            )
        else:
            docs = _retriever.invoke(query)[:top_k]

        passages = []
        for doc in docs:
            meta = doc.metadata
            header = f"[{meta.get('ticker', 'UNKNOWN')} | {meta.get('form_type', '')} | {meta.get('filing_date', '')}]"
            passages.append(f"{header}\n{doc.page_content}")

        return passages if passages else ["No relevant SEC filing passages found."]
    except Exception as exc:
        logger.warning("retrieve_sec_context failed: %s", exc)
        return [f"RAG retrieval error: {exc}"]


@tool
def list_indexed_companies() -> list[dict]:
    """
    List all company tickers that have been indexed in the SEC filings vector store.
    Useful for checking what data is available before querying.
    """
    if _retriever is None:
        return [{"warning": "RAG retriever not initialised"}]

    try:
        collection = _retriever.vectorstore._collection  # type: ignore[attr-defined]
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        seen: dict[str, dict] = {}
        for meta in all_meta:
            ticker = meta.get("ticker", "UNKNOWN")
            if ticker not in seen:
                seen[ticker] = {
                    "ticker": ticker,
                    "company": meta.get("company"),
                    "forms": set(),
                }
            seen[ticker]["forms"].add(meta.get("form_type", ""))
        return [
            {**v, "forms": sorted(v["forms"])}
            for v in seen.values()
        ]
    except Exception as exc:
        logger.warning("list_indexed_companies failed: %s", exc)
        return [{"error": str(exc)}]


# ── Exported list ─────────────────────────────────────────────────────────────

RAG_TOOLS = [retrieve_sec_context, list_indexed_companies]
