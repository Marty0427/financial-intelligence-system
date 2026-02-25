# Financial Intelligence Multi-Agent System

A production-grade multi-agent financial intelligence platform built with **LangGraph** + **LangChain**. A supervisor agent routes queries to specialist sub-agents that each pull from a different live data source, combining results into one coherent answer.

## Architecture

```
User Query (FastAPI)
       │
  [Supervisor]  ← GPT-4o: analyses query, selects agents via Send()
       │
   ┌───┴───────┬───────────┬────────────┐
   │           │           │            │
[Market]   [News]       [RAG]      [Sentiment]
   │           │           │            │
 Yahoo      NewsAPI     Chroma       FinBERT
 Finance  + Tavily    (SEC 10-K/Q)  + F&G proxy
   │           │           │            │
   └───────────┴───────────┴────────────┘
                      │
               [Synthesize]  ← GPT-4o merges all data
                      │
                 Final Answer
```

## Data Sources

| Agent | Sources |
|-------|---------|
| **Market Data** | Yahoo Finance (prices, financials, P/E, technicals), Alpha Vantage (RSI, MACD, SMA) |
| **News** | NewsAPI (articles), Tavily (real-time web search) |
| **RAG** | SEC EDGAR 10-K / 10-Q filings → Chroma vector store |
| **Sentiment** | FinBERT (ProsusAI/finbert), Fear & Greed proxy from price data |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch (smaller install):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required:
- `OPENAI_API_KEY` — [platform.openai.com](https://platform.openai.com)

Optional (system degrades gracefully without them):
- `NEWS_API_KEY` — [newsapi.org](https://newsapi.org) (free tier)
- `TAVILY_API_KEY` — [tavily.com](https://tavily.com) (free tier)
- `ALPHA_VANTAGE_API_KEY` — [alphavantage.co](https://www.alphavantage.co) (free tier)

### 3. Ingest SEC filings (RAG knowledge base)

```bash
# Index 3 most recent 10-K + 10-Q filings for Apple and Microsoft
python scripts/ingest_sec.py AAPL MSFT

# Customise: only 10-K, 2 filings
python scripts/ingest_sec.py --forms 10-K --num 2 AAPL MSFT GOOGL TSLA NVDA
```

### 4. Start the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints

### `POST /query` — Main query endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Apple'\''s revenue trend, current stock price, and investor sentiment?"}'
```

Response:
```json
{
  "query": "...",
  "answer": "## Apple Inc. (AAPL) Financial Intelligence Report\n\n### Market Data\n...",
  "ticker": "AAPL",
  "company": "Apple Inc.",
  "agents_used": ["market_data", "news", "rag", "sentiment"],
  "data_summary": {
    "market_data_available": true,
    "news_articles_found": 5,
    "rag_chunks_retrieved": 6,
    "sentiment_available": true,
    "agent_errors": []
  }
}
```

### `POST /query/stream` — SSE streaming

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyse Microsoft'\''s competitive position"}'
```

Emits `status`, `token`, and `done` events as the pipeline progresses.

### `POST /ingest` — Index new company (background)

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["NVDA", "AMD"], "forms": ["10-K", "10-Q"], "num_filings": 3}'
```

### `POST /ingest/sync` — Synchronous ingestion (blocks)

```bash
curl -X POST http://localhost:8000/ingest/sync \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["TSLA"], "forms": ["10-K"], "num_filings": 2}'
```

### `GET /ingest/status` — Ingestion job status

```bash
curl http://localhost:8000/ingest/status
```

### `GET /health` — System health check

```bash
curl http://localhost:8000/health
```

### `GET /companies` — List indexed tickers

```bash
curl http://localhost:8000/companies
```

## Example Queries

```
"What is Apple's current stock price and how does it compare to analyst targets?"

"Summarise Microsoft's revenue segments and growth drivers from their most recent 10-K."

"What are the main risk factors Tesla mentioned in their annual report, and what's the current investor sentiment?"

"Compare NVIDIA's financial performance over the last two years and explain recent news catalysts."

"What does Apple's management say about AI strategy in their SEC filings, and how is the market reacting?"
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
financial-intelligence-system/
├── app/
│   ├── main.py               # FastAPI app + lifespan
│   ├── config.py             # pydantic-settings
│   ├── dependencies.py       # FastAPI DI
│   ├── agents/
│   │   ├── state.py          # AgentState TypedDict + reducers
│   │   ├── graph.py          # LangGraph StateGraph
│   │   ├── supervisor.py     # Supervisor (routing) node
│   │   ├── market_data.py    # Market Data agent
│   │   ├── news.py           # News agent
│   │   ├── rag.py            # RAG agent
│   │   └── sentiment.py      # Sentiment agent
│   ├── tools/
│   │   ├── market_tools.py   # yfinance + Alpha Vantage
│   │   ├── news_tools.py     # NewsAPI + Tavily
│   │   ├── rag_tools.py      # Chroma retrieval
│   │   └── sentiment_tools.py # FinBERT + F&G proxy
│   ├── api/
│   │   ├── middleware.py     # CORS, rate limit, logging
│   │   └── routes/
│   │       ├── query.py      # POST /query, /query/stream
│   │       ├── ingest.py     # POST /ingest
│   │       └── health.py     # GET /health, /companies
│   └── ingestion/
│       ├── vectorstore.py    # Chroma factory
│       └── sec_pipeline.py   # SEC download + chunking + embedding
├── scripts/
│   └── ingest_sec.py         # CLI ingestion script
├── tests/
│   ├── conftest.py
│   ├── test_tools.py
│   ├── test_graph.py
│   └── test_api.py
├── data/
│   ├── chroma_db/            # Chroma persistence (git-ignored)
│   └── sec_filings/          # Downloaded SEC filings (git-ignored)
├── .env.example
└── requirements.txt
```

## Key Design Decisions

### Supervisor + Parallel Agents (LangGraph Send API)
The supervisor uses GPT-4o with structured output (Pydantic) to select which agents to invoke. Selected agents run in parallel via `langgraph.constants.Send`, dramatically reducing latency compared to sequential execution.

### Graceful Degradation
Each agent node is wrapped in try/except. Errors are collected in `agent_errors` and the synthesiser is instructed to acknowledge missing data rather than failing the entire request.

### Lazy FinBERT Loading
The FinBERT model is loaded on first call to avoid startup latency. If the model can't load (memory constraints, etc.), the sentiment agent falls back to LLM-based sentiment analysis automatically.

### Chroma MMR Retrieval
The RAG agent uses Maximum Marginal Relevance search to retrieve diverse, relevant SEC filing passages rather than potentially redundant top-k nearest neighbours.

### Rate Limiting
An in-memory per-IP rate limiter (30 req/min by default, configurable via `.env`) protects against abuse. For production, replace with Redis-backed rate limiting (e.g. `slowapi`).
