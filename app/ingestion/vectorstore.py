"""Chroma vector store factory and retriever builder."""

from __future__ import annotations

import logging

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


def get_embeddings() -> OpenAIEmbeddings:
    """Return the OpenAI embedding model used for indexing and retrieval."""
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vectorstore() -> Chroma:
    """
    Return a persistent Chroma vector store instance.

    The store is persisted at `settings.chroma_persist_dir` so that
    indexed documents survive server restarts.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    count = vectorstore._collection.count()
    logger.info(
        "Chroma vectorstore loaded — collection=%r docs=%d persist_dir=%r",
        settings.chroma_collection_name,
        count,
        settings.chroma_persist_dir,
    )
    return vectorstore


def get_retriever(vectorstore: Chroma | None = None, search_type: str = "mmr", k: int = 6):
    """
    Build a LangChain retriever from the Chroma vector store.

    Args:
        vectorstore: Existing vectorstore instance (created if None).
        search_type: 'similarity' or 'mmr' (max marginal relevance for diversity).
        k: Number of documents to retrieve.
    """
    vs = vectorstore or get_vectorstore()
    retriever = vs.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
    logger.info("Retriever built (search_type=%s, k=%d)", search_type, k)
    return retriever


def get_collection_stats() -> dict:
    """Return metadata about the Chroma collection (doc count, tickers indexed)."""
    try:
        vs = get_vectorstore()
        collection = vs._collection
        total = collection.count()
        if total == 0:
            return {"total_documents": 0, "tickers": [], "form_types": []}

        all_meta = collection.get(include=["metadatas"])["metadatas"]
        tickers = sorted({m.get("ticker", "UNKNOWN") for m in all_meta})
        form_types = sorted({m.get("form_type", "UNKNOWN") for m in all_meta})
        return {
            "total_documents": total,
            "tickers": tickers,
            "form_types": form_types,
        }
    except Exception as exc:
        logger.warning("get_collection_stats failed: %s", exc)
        return {"error": str(exc)}
