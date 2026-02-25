"""Market data tools — yfinance (free) + Alpha Vantage (API key optional)."""

from __future__ import annotations

import logging
from typing import Optional

import yfinance as yf
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)


# ── yfinance helpers ──────────────────────────────────────────────────────────

def _safe_value(val):
    """Convert numpy / pandas types to plain Python for JSON serialisation."""
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.ndarray,)):
            return val.tolist()
    except ImportError:
        pass
    return val


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_stock_quote(ticker: str) -> dict:
    """
    Fetch the current stock quote for a given ticker symbol.

    Returns price, market cap, P/E ratio, 52-week high/low, volume, and
    analyst target price.

    Args:
        ticker: NYSE/NASDAQ ticker symbol, e.g. 'AAPL', 'MSFT', 'TSLA'.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker.upper(),
            "name": info.get("shortName") or info.get("longName"),
            "price": _safe_value(info.get("currentPrice") or info.get("regularMarketPrice")),
            "previous_close": _safe_value(info.get("previousClose")),
            "day_change_pct": _safe_value(info.get("regularMarketChangePercent")),
            "market_cap": _safe_value(info.get("marketCap")),
            "pe_ratio": _safe_value(info.get("trailingPE")),
            "forward_pe": _safe_value(info.get("forwardPE")),
            "52w_high": _safe_value(info.get("fiftyTwoWeekHigh")),
            "52w_low": _safe_value(info.get("fiftyTwoWeekLow")),
            "volume": _safe_value(info.get("volume")),
            "avg_volume": _safe_value(info.get("averageVolume")),
            "analyst_target": _safe_value(info.get("targetMeanPrice")),
            "recommendation": info.get("recommendationKey"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange"),
        }
    except Exception as exc:
        logger.warning("get_stock_quote failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker}


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_financial_statements(ticker: str, period: str = "annual") -> dict:
    """
    Retrieve income statement, balance sheet, and cash-flow statement.

    Args:
        ticker: Stock ticker symbol.
        period: 'annual' (default) or 'quarterly'.
    """
    try:
        stock = yf.Ticker(ticker)
        freq = "yearly" if period == "annual" else "quarterly"

        income = stock.get_income_stmt(freq=freq)
        balance = stock.get_balance_sheet(freq=freq)
        cashflow = stock.get_cash_flow(freq=freq)

        def df_to_dict(df):
            if df is None or df.empty:
                return {}
            # Take the two most recent periods only to keep payload small
            df = df.iloc[:, :2]
            return {
                str(col.date()): {
                    k: _safe_value(v)
                    for k, v in col_vals.items()
                    if v is not None
                }
                for col, col_vals in df.items()
            }

        return {
            "ticker": ticker.upper(),
            "period": period,
            "income_statement": df_to_dict(income),
            "balance_sheet": df_to_dict(balance),
            "cash_flow": df_to_dict(cashflow),
        }
    except Exception as exc:
        logger.warning("get_financial_statements failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker}


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_price_history(ticker: str, period: str = "6mo") -> dict:
    """
    Fetch historical OHLCV data and compute basic momentum metrics.

    Args:
        ticker: Stock ticker symbol.
        period: Lookback period — '1mo', '3mo', '6mo', '1y', '2y'.
    """
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return {"error": "No price data returned", "ticker": ticker}

        recent = hist.tail(5)[["Close", "Volume"]].to_dict(orient="index")
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        return {
            "ticker": ticker.upper(),
            "period": period,
            "start_price": round(start_price, 4),
            "end_price": round(end_price, 4),
            "return_pct": round((end_price / start_price - 1) * 100, 2),
            "avg_volume": int(hist["Volume"].mean()),
            "recent_5_days": {
                str(k.date()): {"close": round(float(v["Close"]), 4), "volume": int(v["Volume"])}
                for k, v in recent.items()
            },
        }
    except Exception as exc:
        logger.warning("get_price_history failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker}


# ── Alpha Vantage (optional — only used if API key is configured) ─────────────

@tool
def get_technical_indicators(ticker: str) -> dict:
    """
    Fetch RSI-14, MACD, and 50/200-day SMA for the given ticker.
    Uses Alpha Vantage API if the key is configured, otherwise falls back
    to computing them from yfinance data.

    Args:
        ticker: Stock ticker symbol.
    """
    if settings.alpha_vantage_api_key:
        return _av_technical_indicators(ticker)
    return _yf_technical_indicators(ticker)


def _av_technical_indicators(ticker: str) -> dict:
    """Alpha Vantage technical indicators."""
    try:
        from alpha_vantage.techindicators import TechIndicators  # noqa: PLC0415

        ti = TechIndicators(key=settings.alpha_vantage_api_key, output_format="pandas")
        rsi_data, _ = ti.get_rsi(symbol=ticker, interval="daily", time_period=14)
        macd_data, _ = ti.get_macd(symbol=ticker, interval="daily")
        sma50, _ = ti.get_sma(symbol=ticker, interval="daily", time_period=50)
        sma200, _ = ti.get_sma(symbol=ticker, interval="daily", time_period=200)

        return {
            "ticker": ticker.upper(),
            "source": "alpha_vantage",
            "rsi_14": float(rsi_data["RSI"].iloc[-1]),
            "macd": float(macd_data["MACD"].iloc[-1]),
            "macd_signal": float(macd_data["MACD_Signal"].iloc[-1]),
            "sma_50": float(sma50["SMA"].iloc[-1]),
            "sma_200": float(sma200["SMA"].iloc[-1]),
        }
    except Exception as exc:
        logger.warning("Alpha Vantage indicators failed for %s: %s", ticker, exc)
        return _yf_technical_indicators(ticker)


def _yf_technical_indicators(ticker: str) -> dict:
    """Compute RSI-14, SMA-50, SMA-200 from yfinance data (no API key needed)."""
    try:
        import pandas as pd  # noqa: PLC0415

        hist = yf.Ticker(ticker).history(period="1y")["Close"]
        if len(hist) < 50:
            return {"error": "Insufficient data for indicators", "ticker": ticker}

        # RSI
        delta = hist.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - 100 / (1 + rs)

        # MACD
        ema12 = hist.ewm(span=12, adjust=False).mean()
        ema26 = hist.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        return {
            "ticker": ticker.upper(),
            "source": "yfinance_computed",
            "rsi_14": round(float(rsi.iloc[-1]), 2),
            "macd": round(float(macd_line.iloc[-1]), 4),
            "macd_signal": round(float(signal_line.iloc[-1]), 4),
            "sma_50": round(float(hist.rolling(50).mean().iloc[-1]), 4),
            "sma_200": round(float(hist.rolling(200).mean().iloc[-1]), 4) if len(hist) >= 200 else None,
        }
    except Exception as exc:
        logger.warning("yf_technical_indicators failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker}


# ── Exported list ─────────────────────────────────────────────────────────────

MARKET_TOOLS = [get_stock_quote, get_financial_statements, get_price_history, get_technical_indicators]
