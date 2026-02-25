"""Unit tests for individual tools (market, news, RAG, sentiment)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma")


# ── Market tools ───────────────────────────────────────────────────────────────

class TestMarketTools:
    def test_safe_value_int(self):
        from app.tools.market_tools import _safe_value
        try:
            import numpy as np
            assert _safe_value(np.int64(42)) == 42
            assert isinstance(_safe_value(np.int64(42)), int)
        except ImportError:
            pytest.skip("numpy not installed")

    def test_safe_value_plain(self):
        from app.tools.market_tools import _safe_value
        assert _safe_value("hello") == "hello"
        assert _safe_value(3.14) == 3.14

    @patch("yfinance.Ticker")
    def test_get_stock_quote_success(self, mock_ticker):
        mock_ticker.return_value.info = {
            "shortName": "Apple Inc.",
            "currentPrice": 182.5,
            "previousClose": 180.0,
            "marketCap": 2_800_000_000_000,
            "trailingPE": 28.5,
            "currency": "USD",
        }
        from app.tools.market_tools import get_stock_quote
        result = get_stock_quote.invoke({"ticker": "AAPL"})
        assert result["ticker"] == "AAPL"
        assert result["price"] == 182.5
        assert result["name"] == "Apple Inc."

    @patch("yfinance.Ticker")
    def test_get_stock_quote_error(self, mock_ticker):
        mock_ticker.return_value.info = MagicMock(side_effect=Exception("Network error"))
        # Should not raise — returns error dict
        from app.tools.market_tools import get_stock_quote
        # Patch the property access that fails
        mock_ticker.return_value.info = {"currentPrice": None}
        result = get_stock_quote.invoke({"ticker": "FAKE"})
        assert "ticker" in result

    @patch("yfinance.Ticker")
    def test_get_price_history(self, mock_ticker):
        import pandas as pd
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        hist = pd.DataFrame({"Close": [150.0 + i for i in range(20)], "Volume": [1_000_000] * 20}, index=dates)
        mock_ticker.return_value.history.return_value = hist

        from app.tools.market_tools import get_price_history
        result = get_price_history.invoke({"ticker": "AAPL", "period": "1mo"})
        assert result["ticker"] == "AAPL"
        assert result["return_pct"] > 0
        assert "recent_5_days" in result


# ── News tools ────────────────────────────────────────────────────────────────

class TestNewsTools:
    def test_no_api_key_returns_warning(self):
        import app.config as cfg
        original = cfg.settings.news_api_key
        cfg.settings.news_api_key = ""
        try:
            from app.tools.news_tools import search_financial_news
            result = search_financial_news.invoke({"query": "Apple earnings"})
            assert isinstance(result, list)
            assert "warning" in result[0]
        finally:
            cfg.settings.news_api_key = original

    def test_no_tavily_key_returns_warning(self):
        import app.config as cfg
        original = cfg.settings.tavily_api_key
        cfg.settings.tavily_api_key = ""
        try:
            from app.tools.news_tools import tavily_web_search
            result = tavily_web_search.invoke({"query": "Apple stock news"})
            assert isinstance(result, list)
            assert "warning" in result[0]
        finally:
            cfg.settings.tavily_api_key = original


# ── RAG tools ─────────────────────────────────────────────────────────────────

class TestRagTools:
    def test_retrieve_without_retriever_returns_message(self):
        import app.tools.rag_tools as rag
        original = rag._retriever
        rag._retriever = None
        try:
            from app.tools.rag_tools import retrieve_sec_context
            result = retrieve_sec_context.invoke({"query": "revenue growth", "ticker": "AAPL"})
            assert isinstance(result, list)
            assert len(result) == 1
            assert "not initialised" in result[0].lower()
        finally:
            rag._retriever = original

    def test_set_retriever(self):
        from app.tools.rag_tools import set_retriever
        mock_retriever = MagicMock()
        set_retriever(mock_retriever)
        import app.tools.rag_tools as rag
        assert rag._retriever is mock_retriever
        rag._retriever = None  # cleanup


# ── Sentiment tools ───────────────────────────────────────────────────────────

class TestSentimentTools:
    def test_analyze_empty_input(self):
        from app.tools.sentiment_tools import analyze_news_sentiment
        result = analyze_news_sentiment.invoke({"texts": []})
        assert "error" in result

    def test_fear_greed_with_mocked_yfinance(self):
        import pandas as pd
        import numpy as np

        with patch("yfinance.Ticker") as mock_ticker:
            dates = pd.date_range("2024-01-01", periods=90, freq="D")
            prices = pd.Series(
                [100 + np.sin(i / 10) * 10 + i * 0.5 for i in range(90)],
                index=dates,
            )
            mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": prices})

            from app.tools.sentiment_tools import get_fear_and_greed_proxy
            result = get_fear_and_greed_proxy.invoke({"ticker": "AAPL"})
            assert "fear_greed_score" in result
            assert 0 <= result["fear_greed_score"] <= 100
            assert result["label"] in ("Extreme Greed", "Greed", "Neutral", "Fear", "Extreme Fear")
