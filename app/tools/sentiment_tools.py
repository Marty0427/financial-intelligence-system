"""Sentiment analysis tools — FinBERT (local) with LLM fallback."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# FinBERT pipeline is loaded lazily to avoid startup delay
_finbert_pipeline = None


def _load_finbert():
    """Lazily load the FinBERT pipeline (ProsusAI/finbert)."""
    global _finbert_pipeline  # noqa: PLW0603
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline  # noqa: PLC0415

            logger.info("Loading FinBERT model (first call may take a few seconds)…")
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,  # return all labels with scores
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as exc:
            logger.warning("Could not load FinBERT: %s — will use LLM fallback", exc)
    return _finbert_pipeline


def _score_with_finbert(texts: list[str]) -> list[dict]:
    """Run FinBERT on a list of text snippets; return per-text label/score."""
    pipe = _load_finbert()
    if pipe is None:
        return []

    results = []
    for text in texts:
        try:
            raw = pipe(text[:512])  # FinBERT max tokens
            # raw is list[list[dict]] when top_k=None
            scores = raw[0] if isinstance(raw[0], list) else raw
            label = max(scores, key=lambda x: x["score"])["label"].lower()
            label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            sentiment_map = {s["label"].lower(): s["score"] for s in scores}
            results.append(
                {
                    "label": label,
                    "score": label_map.get(label, 0.0),
                    "positive": round(sentiment_map.get("positive", 0), 4),
                    "neutral": round(sentiment_map.get("neutral", 0), 4),
                    "negative": round(sentiment_map.get("negative", 0), 4),
                }
            )
        except Exception as exc:
            logger.warning("FinBERT scoring error: %s", exc)
            results.append({"label": "neutral", "score": 0.0, "error": str(exc)})
    return results


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def analyze_news_sentiment(texts: list[str]) -> dict:
    """
    Run financial sentiment analysis (FinBERT) on a list of news headlines
    or article snippets and return aggregate sentiment metrics.

    Args:
        texts: List of text strings (headlines, descriptions, or short passages).
               Maximum 20 items; each truncated to 512 characters.
    """
    if not texts:
        return {"error": "No texts provided"}

    texts = [t[:512] for t in texts[:20]]

    finbert_results = _score_with_finbert(texts)

    if not finbert_results:
        # LLM fallback handled by the calling agent (prompt-based)
        return {
            "source": "finbert_unavailable",
            "message": "FinBERT model not loaded. Sentiment analysis delegated to LLM.",
        }

    positive = sum(1 for r in finbert_results if r["label"] == "positive")
    negative = sum(1 for r in finbert_results if r["label"] == "negative")
    neutral = sum(1 for r in finbert_results if r["label"] == "neutral")
    avg_score = sum(r["score"] for r in finbert_results) / len(finbert_results)

    return {
        "source": "finbert",
        "num_texts_analyzed": len(finbert_results),
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "average_sentiment_score": round(avg_score, 4),
        "overall_label": "positive" if avg_score > 0.15 else ("negative" if avg_score < -0.15 else "neutral"),
        "per_text": finbert_results,
    }


@tool
def get_fear_and_greed_proxy(ticker: str) -> dict:
    """
    Compute a proxy Fear & Greed indicator from market data:
    - RSI momentum (overbought / oversold)
    - Price vs 50-day SMA (trend)
    - 30-day volatility (market stress)

    Args:
        ticker: Stock ticker symbol to analyse.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        hist = yf.Ticker(ticker).history(period="3mo")["Close"]
        if len(hist) < 30:
            return {"error": "Insufficient history for fear & greed calculation", "ticker": ticker}

        # RSI-14
        delta = hist.diff()
        gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] or 1e-9)))

        # Price vs SMA-50
        sma50 = float(hist.rolling(50).mean().iloc[-1]) if len(hist) >= 50 else float(hist.mean())
        price = float(hist.iloc[-1])
        price_vs_sma_pct = (price / sma50 - 1) * 100

        # 30-day volatility (annualised)
        returns = hist.pct_change().dropna()
        volatility_30d = float(np.std(returns.tail(30)) * np.sqrt(252) * 100)

        # Score 0–100
        rsi_score = rsi  # already 0–100
        trend_score = min(max(price_vs_sma_pct * 5 + 50, 0), 100)
        vol_score = max(100 - volatility_30d * 2, 0)  # high vol = low score
        composite = round((rsi_score + trend_score + vol_score) / 3, 1)

        label = (
            "Extreme Greed" if composite >= 75
            else "Greed" if composite >= 55
            else "Neutral" if composite >= 45
            else "Fear" if composite >= 25
            else "Extreme Fear"
        )

        return {
            "ticker": ticker.upper(),
            "fear_greed_score": composite,
            "label": label,
            "components": {
                "rsi_14": round(rsi, 2),
                "price_vs_sma50_pct": round(price_vs_sma_pct, 2),
                "volatility_30d_annualised_pct": round(volatility_30d, 2),
            },
        }
    except Exception as exc:
        logger.warning("get_fear_and_greed_proxy failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker}


# ── Exported list ─────────────────────────────────────────────────────────────

SENTIMENT_TOOLS = [analyze_news_sentiment, get_fear_and_greed_proxy]
