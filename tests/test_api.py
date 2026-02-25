"""API integration tests using FastAPI TestClient."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma")


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_graph():
    """Return a mock graph that returns a plausible state."""
    graph = MagicMock()
    graph.nodes = {"supervisor": None, "market_data": None, "news": None, "rag": None, "sentiment": None, "synthesize": None}
    graph.ainvoke = AsyncMock(
        return_value={
            "query": "What is Apple's stock price?",
            "final_answer": "Apple (AAPL) is trading at $182.50. Recent news indicates strong iPhone sales.",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "selected_agents": ["market_data", "news"],
            "market_data": {"raw": "price: 182.50"},
            "news_results": [{"content": "Apple beats earnings estimates"}],
            "rag_context": [],
            "sentiment": None,
            "agent_errors": [],
        }
    )
    return graph


def _mock_retriever():
    return MagicMock()


def _mock_chroma_stats():
    return {"total_documents": 150, "tickers": ["AAPL", "MSFT"], "form_types": ["10-K", "10-Q"]}


@pytest.fixture
def app_with_mocks():
    """Create the FastAPI app with external services mocked."""
    with (
        patch("app.dependencies.get_financial_graph", return_value=_mock_graph()),
        patch("app.dependencies.get_chroma_retriever", return_value=_mock_retriever()),
        patch("app.ingestion.vectorstore.get_collection_stats", return_value=_mock_chroma_stats()),
        patch("langchain_openai.ChatOpenAI"),
        patch("langchain_openai.OpenAIEmbeddings"),
        patch("chromadb.PersistentClient"),
    ):
        from app.main import create_app
        return create_app()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    @pytest.mark.asyncio
    async def test_query_returns_200(self, app_with_mocks):
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={"query": "What is Apple's current stock price?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "agents_used" in data
        assert "data_summary" in data

    @pytest.mark.asyncio
    async def test_query_validation_too_short(self, app_with_mocks):
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            response = await client.post("/query", json={"query": "Hi"})
        assert response.status_code == 422  # validation error

    @pytest.mark.asyncio
    async def test_query_response_structure(self, app_with_mocks):
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={"query": "Analyse Apple's financial performance"},
            )
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["company"] == "Apple Inc."
        assert data["data_summary"]["market_data_available"] is True
        assert data["data_summary"]["news_articles_found"] == 1


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_companies_endpoint(self, app_with_mocks):
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            with patch("app.api.routes.health.get_collection_stats", return_value=_mock_chroma_stats()):
                response = await client.get("/companies")
        assert response.status_code == 200
        data = response.json()
        assert "indexed_tickers" in data
        assert "AAPL" in data["indexed_tickers"]


class TestIngestEndpoint:
    @pytest.mark.asyncio
    async def test_ingest_returns_202_accepted(self, app_with_mocks):
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            response = await client.post(
                "/ingest",
                json={"tickers": ["AAPL"], "forms": ["10-K"], "num_filings": 1},
            )
        assert response.status_code == 200
        data = response.json()
        assert "AAPL" in data["tickers"]
        assert "job_id" in data

    @pytest.mark.asyncio
    async def test_ingest_empty_tickers(self, app_with_mocks):
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            response = await client.post(
                "/ingest",
                json={"tickers": [], "forms": ["10-K"], "num_filings": 1},
            )
        # pydantic min_length=1 on the list
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_status(self, app_with_mocks):
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            response = await client.get("/ingest/status")
        assert response.status_code == 200
        assert "jobs" in response.json()
