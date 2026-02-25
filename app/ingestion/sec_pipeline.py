"""SEC EDGAR ingestion pipeline — download, parse, chunk, embed, and store."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings
from app.ingestion.vectorstore import get_embeddings, get_vectorstore

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

SEC_FILINGS_DIR = Path(settings.chroma_persist_dir).parent / "sec_filings"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
SUPPORTED_FORMS = ["10-K", "10-Q"]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_txt_file(filepath: Path) -> str:
    """Read a plain-text or HTML SEC filing and return clean text."""
    try:
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        # Strip HTML tags if present
        if "<html" in raw[:500].lower() or "<!doctype" in raw[:500].lower():
            try:
                from bs4 import BeautifulSoup  # noqa: PLC0415

                soup = BeautifulSoup(raw, "html.parser")
                # Remove script/style elements
                for tag in soup(["script", "style", "header", "footer", "nav"]):
                    tag.decompose()
                raw = soup.get_text(separator="\n", strip=True)
            except ImportError:
                import re  # noqa: PLC0415

                raw = re.sub(r"<[^>]+>", " ", raw)
        return raw
    except Exception as exc:
        logger.warning("Could not parse %s: %s", filepath, exc)
        return ""


def _parse_pdf_file(filepath: Path) -> str:
    """Extract text from a PDF SEC filing."""
    try:
        from pypdf import PdfReader  # noqa: PLC0415

        reader = PdfReader(str(filepath))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as exc:
        logger.warning("Could not parse PDF %s: %s", filepath, exc)
        return ""


def _extract_text(filepath: Path) -> str:
    """Route to the appropriate parser based on file extension."""
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf_file(filepath)
    return _parse_txt_file(filepath)  # .txt, .htm, .html


def _build_documents(
    text: str,
    ticker: str,
    company: str,
    form_type: str,
    filing_date: str,
    accession_number: str,
) -> list[Document]:
    """Split text into chunks and wrap each in a LangChain Document with metadata."""
    if not text.strip():
        logger.warning("Empty text for %s %s %s — skipping", ticker, form_type, filing_date)
        return []

    chunks = text_splitter.split_text(text)
    logger.info(
        "Split %s %s %s into %d chunks", ticker, form_type, filing_date, len(chunks)
    )

    return [
        Document(
            page_content=chunk,
            metadata={
                "ticker": ticker.upper(),
                "company": company,
                "form_type": form_type,
                "filing_date": filing_date,
                "accession_number": accession_number,
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
    ]


# ── Core ingestion ────────────────────────────────────────────────────────────

async def ingest_ticker(
    ticker: str,
    forms: list[str] | None = None,
    num_filings: int = 3,
    after_date: str = "2021-01-01",
) -> dict:
    """
    Download SEC filings for `ticker`, extract text, chunk, embed, and store in Chroma.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        forms: Filing form types to download (default: ['10-K', '10-Q']).
        num_filings: Maximum number of filings per form type.
        after_date: Only download filings after this date (YYYY-MM-DD).

    Returns:
        Summary dict with counts of documents indexed.
    """
    if forms is None:
        forms = SUPPORTED_FORMS

    ticker = ticker.upper()
    logger.info("Starting ingestion for ticker=%s forms=%s", ticker, forms)

    try:
        from sec_edgar_downloader import Downloader  # noqa: PLC0415
    except ImportError:
        return {
            "error": "sec-edgar-downloader not installed. Run: pip install sec-edgar-downloader",
            "ticker": ticker,
        }

    dl_dir = SEC_FILINGS_DIR / ticker
    dl_dir.mkdir(parents=True, exist_ok=True)

    downloader = Downloader(
        company_name="FinancialIntelligenceSystem",
        email_address="research@example.com",
        save_path=str(dl_dir),
    )

    total_docs = 0
    vectorstore = get_vectorstore()
    all_documents: list[Document] = []

    for form_type in forms:
        try:
            logger.info("Downloading %s filings for %s (limit=%d)", form_type, ticker, num_filings)
            downloader.get(form_type, ticker, limit=num_filings, after=after_date)

            form_dir = dl_dir / "sec-edgar-filings" / ticker / form_type
            if not form_dir.exists():
                logger.warning("No %s filings found for %s", form_type, ticker)
                continue

            for filing_dir in sorted(form_dir.iterdir(), reverse=True)[:num_filings]:
                if not filing_dir.is_dir():
                    continue

                accession = filing_dir.name

                # Find the primary filing file
                filing_file = None
                for pattern in ["*.htm", "*.html", "*.txt", "*.pdf"]:
                    candidates = list(filing_dir.glob(pattern))
                    # Prefer the largest file (usually the main document)
                    if candidates:
                        filing_file = max(candidates, key=lambda p: p.stat().st_size)
                        break

                if filing_file is None:
                    logger.warning("No filing file found in %s", filing_dir)
                    continue

                # Derive filing date from accession number (YYYYMMDD format embedded)
                try:
                    date_part = accession.split("-")
                    filing_date = f"{date_part[1][:4]}-{date_part[1][4:6]}-{date_part[1][6:8]}" if len(date_part) > 1 else "unknown"
                except Exception:
                    filing_date = "unknown"

                text = _extract_text(filing_file)
                if not text:
                    continue

                # Get company name from yfinance
                company_name = ticker
                try:
                    import yfinance as yf  # noqa: PLC0415
                    info = yf.Ticker(ticker).info
                    company_name = info.get("shortName") or info.get("longName") or ticker
                except Exception:
                    pass

                docs = _build_documents(
                    text=text,
                    ticker=ticker,
                    company=company_name,
                    form_type=form_type,
                    filing_date=filing_date,
                    accession_number=accession,
                )
                all_documents.extend(docs)
                logger.info(
                    "Processed %s %s (%s) → %d chunks",
                    ticker, form_type, filing_date, len(docs)
                )

        except Exception as exc:
            logger.error("Failed to download/process %s %s: %s", ticker, form_type, exc)

    if all_documents:
        logger.info("Adding %d documents to Chroma for %s…", len(all_documents), ticker)
        vectorstore.add_documents(all_documents)
        logger.info("Chroma ingestion complete for %s", ticker)
        total_docs = len(all_documents)
    else:
        logger.warning("No documents to index for %s", ticker)

    return {
        "ticker": ticker,
        "total_chunks_indexed": total_docs,
        "forms_attempted": forms,
    }


async def ingest_multiple_tickers(
    tickers: list[str],
    forms: list[str] | None = None,
    num_filings: int = 3,
) -> list[dict]:
    """Ingest SEC filings for multiple tickers sequentially."""
    results = []
    for ticker in tickers:
        result = await ingest_ticker(ticker, forms=forms, num_filings=num_filings)
        results.append(result)
    return results
