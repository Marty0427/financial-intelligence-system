"""
CLI script to ingest SEC 10-K/10-Q filings into the Chroma vector store.

Usage:
    python scripts/ingest_sec.py AAPL MSFT GOOGL
    python scripts/ingest_sec.py --forms 10-K --num 2 TSLA NVDA
    python scripts/ingest_sec.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_sec")


async def main(tickers: list[str], forms: list[str], num_filings: int) -> None:
    from app.ingestion.sec_pipeline import ingest_ticker  # noqa: PLC0415
    from app.ingestion.vectorstore import get_collection_stats  # noqa: PLC0415

    print(f"\n{'='*60}")
    print(f"  SEC Filing Ingestion Pipeline")
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  Forms   : {', '.join(forms)}")
    print(f"  Max     : {num_filings} filing(s) per form")
    print(f"{'='*60}\n")

    for ticker in tickers:
        print(f"[{ticker}] Starting ingestion…")
        try:
            result = await ingest_ticker(
                ticker=ticker,
                forms=forms,
                num_filings=num_filings,
            )
            chunks = result.get("total_chunks_indexed", 0)
            if chunks:
                print(f"[{ticker}] ✓ Indexed {chunks} document chunks")
            else:
                print(f"[{ticker}] ⚠ No documents indexed (check filing availability)")
        except Exception as exc:
            print(f"[{ticker}] ✗ Error: {exc}")
            logger.exception("Ingestion error for %s", ticker)

    print("\n── Vectorstore Summary ──")
    stats = get_collection_stats()
    if "error" in stats:
        print(f"Could not retrieve stats: {stats['error']}")
    else:
        print(f"Total document chunks : {stats.get('total_documents', 0)}")
        print(f"Indexed tickers       : {', '.join(stats.get('tickers', []))}")
        print(f"Form types            : {', '.join(stats.get('form_types', []))}")

    print("\nIngestion complete. Start the server with:")
    print("  uvicorn app.main:app --reload\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest SEC 10-K/10-Q filings into the Chroma vector store"
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        help="One or more stock ticker symbols (e.g. AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-Q"],
        choices=["10-K", "10-Q"],
        help="SEC form types to download (default: 10-K 10-Q)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of filings per form type (default: 3)",
    )

    args = parser.parse_args()
    asyncio.run(main(args.tickers, args.forms, args.num))
