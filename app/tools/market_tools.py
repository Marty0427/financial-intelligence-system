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


# ── Peer comparison ───────────────────────────────────────────────────────────

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def compare_stocks(tickers: str) -> dict:
    """
    Compare 2–5 stocks side-by-side on key valuation, growth, and profitability
    metrics. Useful for competitive analysis or selecting among similar companies.

    Args:
        tickers: Comma-separated ticker symbols, e.g. 'AAPL,MSFT,GOOGL'.
    """
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()][:5]
    if len(symbols) < 2:
        return {"error": "Provide at least 2 comma-separated tickers", "tickers": tickers}

    rows = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            rows[sym] = {
                "name": info.get("shortName") or info.get("longName"),
                "price": _safe_value(info.get("currentPrice") or info.get("regularMarketPrice")),
                "market_cap_B": round(_safe_value(info.get("marketCap") or 0) / 1e9, 2),
                "pe_trailing": _safe_value(info.get("trailingPE")),
                "pe_forward": _safe_value(info.get("forwardPE")),
                "ev_ebitda": _safe_value(info.get("enterpriseToEbitda")),
                "price_to_book": _safe_value(info.get("priceToBook")),
                "revenue_growth_yoy": _safe_value(info.get("revenueGrowth")),
                "earnings_growth_yoy": _safe_value(info.get("earningsGrowth")),
                "gross_margin": _safe_value(info.get("grossMargins")),
                "operating_margin": _safe_value(info.get("operatingMargins")),
                "net_margin": _safe_value(info.get("profitMargins")),
                "roe": _safe_value(info.get("returnOnEquity")),
                "debt_to_equity": _safe_value(info.get("debtToEquity")),
                "dividend_yield": _safe_value(info.get("dividendYield")),
                "analyst_target": _safe_value(info.get("targetMeanPrice")),
                "recommendation": info.get("recommendationKey"),
            }
        except Exception as exc:
            logger.warning("compare_stocks failed for %s: %s", sym, exc)
            rows[sym] = {"error": str(exc)}

    return {"tickers": symbols, "comparison": rows}


# ── Earnings calendar & surprise history ──────────────────────────────────────

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_earnings_info(ticker: str) -> dict:
    """
    Return the next scheduled earnings date plus the last 4 quarters of EPS
    actuals vs. estimates (surprise history). Useful for event-driven analysis.

    Args:
        ticker: Stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)

        # Next earnings date
        cal = stock.calendar
        next_date = None
        if cal is not None and not cal.empty:
            # calendar is a DataFrame; earnings date is usually row 0
            try:
                val = cal.loc["Earnings Date"].iloc[0] if "Earnings Date" in cal.index else None
                if val is not None:
                    next_date = str(val.date()) if hasattr(val, "date") else str(val)
            except Exception:
                pass

        # EPS surprise history (last 4 quarters)
        history = stock.earnings_history
        surprises = []
        if history is not None and not history.empty:
            for _, row in history.tail(4).iterrows():
                surprises.append({
                    "period": str(row.get("period") or row.name),
                    "eps_estimate": _safe_value(row.get("epsEstimate")),
                    "eps_actual": _safe_value(row.get("epsActual")),
                    "surprise_pct": _safe_value(row.get("epsDifference")),
                })

        # Analyst EPS estimates for upcoming quarters
        forecasts = []
        try:
            eps_trend = stock.eps_trend
            if eps_trend is not None and not eps_trend.empty:
                for col in eps_trend.columns[:4]:
                    current = eps_trend[col].get("current")
                    forecasts.append({
                        "period": str(col),
                        "eps_estimate": _safe_value(current) if current is not None else None,
                    })
        except Exception:
            pass

        return {
            "ticker": ticker.upper(),
            "next_earnings_date": next_date,
            "eps_surprise_history": surprises,
            "eps_forecasts": forecasts,
        }
    except Exception as exc:
        logger.warning("get_earnings_info failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker}


# ── Insider transactions ───────────────────────────────────────────────────────

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_insider_transactions(ticker: str) -> dict:
    """
    Fetch the most recent insider buy and sell transactions (Form 4 filings).
    Insider activity can signal management conviction or distribution pressure.

    Args:
        ticker: Stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)
        txns = stock.insider_transactions

        if txns is None or txns.empty:
            return {"ticker": ticker.upper(), "transactions": [], "note": "No insider transaction data available"}

        records = []
        for _, row in txns.head(20).iterrows():
            records.append({
                "date": str(row.get("startDate") or row.get("date") or ""),
                "insider": str(row.get("filerName") or row.get("insider") or ""),
                "title": str(row.get("filerRelation") or row.get("title") or ""),
                "transaction": str(row.get("transactionText") or row.get("transaction") or ""),
                "shares": _safe_value(row.get("shares")),
                "value_usd": _safe_value(row.get("value")),
            })

        # Summarise buys vs sells
        buys = [r for r in records if "purchase" in r["transaction"].lower() or "buy" in r["transaction"].lower()]
        sells = [r for r in records if "sale" in r["transaction"].lower() or "sell" in r["transaction"].lower()]

        return {
            "ticker": ticker.upper(),
            "recent_transactions": records,
            "summary": {
                "total_shown": len(records),
                "buy_transactions": len(buys),
                "sell_transactions": len(sells),
                "net_signal": "bullish" if len(buys) > len(sells) else ("bearish" if len(sells) > len(buys) else "neutral"),
            },
        }
    except Exception as exc:
        logger.warning("get_insider_transactions failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker}


# ── Dividend analysis ─────────────────────────────────────────────────────────

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_dividend_analysis(ticker: str) -> dict:
    """
    Return dividend yield, payout ratio, 5-year dividend CAGR, and recent
    dividend history. Useful for income-focused or dividend-growth investing.

    Args:
        ticker: Stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        divs = stock.dividends  # pandas Series indexed by date

        annual_dividend = _safe_value(info.get("dividendRate"))
        yield_val = _safe_value(info.get("dividendYield"))
        payout_ratio = _safe_value(info.get("payoutRatio"))
        ex_date = str(info.get("exDividendDate") or "")
        five_yr_avg_yield = _safe_value(info.get("fiveYearAvgDividendYield"))

        # 5-year CAGR from dividend history
        cagr_5yr = None
        recent_history = []
        if divs is not None and not divs.empty:
            # Annual totals for last 6 years
            annual = divs.resample("YE").sum()
            recent_history = [
                {"year": str(idx.year), "total_dividends": round(float(val), 4)}
                for idx, val in annual.tail(5).items()
                if val > 0
            ]
            if len(annual) >= 5:
                old = float(annual.iloc[-5]) if float(annual.iloc[-5]) > 0 else None
                new = float(annual.iloc[-1]) if float(annual.iloc[-1]) > 0 else None
                if old and new:
                    cagr_5yr = round(((new / old) ** (1 / 4) - 1) * 100, 2)

        if not annual_dividend and (divs is None or divs.empty):
            return {
                "ticker": ticker.upper(),
                "pays_dividend": False,
                "note": "This company does not appear to pay a regular dividend.",
            }

        return {
            "ticker": ticker.upper(),
            "pays_dividend": True,
            "annual_dividend_per_share": annual_dividend,
            "dividend_yield_pct": round(yield_val * 100, 3) if yield_val else None,
            "five_year_avg_yield_pct": five_yr_avg_yield,
            "payout_ratio_pct": round(payout_ratio * 100, 1) if payout_ratio else None,
            "ex_dividend_date": ex_date,
            "dividend_cagr_5yr_pct": cagr_5yr,
            "annual_history": recent_history,
        }
    except Exception as exc:
        logger.warning("get_dividend_analysis failed for %s: %s", ticker, exc)
        return {"error": str(exc), "ticker": ticker}


# ── Exported list ─────────────────────────────────────────────────────────────

MARKET_TOOLS = [
    get_stock_quote,
    get_financial_statements,
    get_price_history,
    get_technical_indicators,
    compare_stocks,
    get_earnings_info,
    get_insider_transactions,
    get_dividend_analysis,
]
