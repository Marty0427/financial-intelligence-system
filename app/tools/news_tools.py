"""News and web-search tools — NewsAPI + Tavily."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)


# ── NewsAPI ───────────────────────────────────────────────────────────────────

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def search_financial_news(
    query: str,
    days_back: int = 7,
    max_results: int = 10,
) -> list[dict]:
    """
    Search recent financial news articles via NewsAPI.

    Args:
        query: Search keywords, e.g. 'Apple earnings Q4 2024'.
        days_back: How many days back to search (default 7, max 30 for free tier).
        max_results: Maximum number of articles to return (max 10).
    """
    if not settings.news_api_key:
        logger.warning("NEWS_API_KEY not configured — skipping NewsAPI search")
        return [{"warning": "NEWS_API_KEY not configured"}]

    try:
        from newsapi import NewsApiClient  # noqa: PLC0415

        client = NewsApiClient(api_key=settings.news_api_key)
        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        response = client.get_everything(
            q=query,
            from_param=from_date,
            language="en",
            sort_by="relevancy",
            page_size=min(max_results, 10),
        )
        articles = []
        for article in response.get("articles", [])[:max_results]:
            articles.append(
                {
                    "title": article.get("title"),
                    "source": article.get("source", {}).get("name"),
                    "published_at": article.get("publishedAt"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "content_snippet": (article.get("content") or "")[:500],
                }
            )
        return articles
    except Exception as exc:
        logger.warning("search_financial_news failed: %s", exc)
        return [{"error": str(exc)}]


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_top_financial_headlines(category: str = "business", max_results: int = 10) -> list[dict]:
    """
    Fetch top headlines from the business / finance category.

    Args:
        category: 'business' or 'general' (NewsAPI category).
        max_results: Maximum number of headlines to return.
    """
    if not settings.news_api_key:
        return [{"warning": "NEWS_API_KEY not configured"}]

    try:
        from newsapi import NewsApiClient  # noqa: PLC0415

        client = NewsApiClient(api_key=settings.news_api_key)
        response = client.get_top_headlines(
            category=category,
            language="en",
            country="us",
            page_size=min(max_results, 10),
        )
        return [
            {
                "title": a.get("title"),
                "source": a.get("source", {}).get("name"),
                "published_at": a.get("publishedAt"),
                "description": a.get("description"),
                "url": a.get("url"),
            }
            for a in response.get("articles", [])[:max_results]
        ]
    except Exception as exc:
        logger.warning("get_top_financial_headlines failed: %s", exc)
        return [{"error": str(exc)}]


# ── Tavily Web Search ─────────────────────────────────────────────────────────

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def tavily_web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Perform a real-time web search using Tavily — ideal for very recent events,
    analyst commentary, and data not yet in NewsAPI.

    Args:
        query: Search query string.
        max_results: Number of results to return (1–10).
    """
    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY not configured — skipping Tavily search")
        return [{"warning": "TAVILY_API_KEY not configured"}]

    try:
        from tavily import TavilyClient  # noqa: PLC0415

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=min(max_results, 10),
            include_answer=True,
        )
        results = []
        if response.get("answer"):
            results.append({"type": "direct_answer", "content": response["answer"]})
        for r in response.get("results", [])[:max_results]:
            results.append(
                {
                    "type": "web_result",
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content", "")[:600],
                    "score": r.get("score"),
                }
            )
        return results
    except Exception as exc:
        logger.warning("tavily_web_search failed: %s", exc)
        return [{"error": str(exc)}]


# ── Exported list ─────────────────────────────────────────────────────────────

NEWS_TOOLS = [search_financial_news, get_top_financial_headlines, tavily_web_search]
