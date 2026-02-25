"""FastAPI dependency injection — graph singleton, retriever, vectorstore."""

from __future__ import annotations

import logging

from fastapi import Depends

from app.agents.graph import get_graph
from app.ingestion.vectorstore import get_retriever, get_vectorstore
from app.tools.rag_tools import set_retriever

logger = logging.getLogger(__name__)

_retriever_initialised = False


def get_financial_graph():
    """Return the compiled LangGraph financial intelligence graph."""
    return get_graph()


def get_chroma_retriever():
    """Return the Chroma MMR retriever (singleton)."""
    global _retriever_initialised  # noqa: PLW0603
    vs = get_vectorstore()
    retriever = get_retriever(vectorstore=vs)
    if not _retriever_initialised:
        # Inject into the RAG tool module so tools can use it
        set_retriever(retriever)
        _retriever_initialised = True
        logger.info("Chroma retriever injected into RAG tools")
    return retriever
