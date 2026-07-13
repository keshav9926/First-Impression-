# First Impression

> **AI system that analyzes a startup's public product experience and produces a grounded, citation-backed report on the new-user journey.**

Given a company's public site or docs URL, First Impression crawls public content, builds a hybrid semantic + keyword index, and runs a ReAct analysis agent that produces a structured report — every claim tied to a source page.

Full spec: [project.md](project.md)

---

## Features

- **Respectful crawling** — robots.txt enforced at the API boundary; rate-limited, no login scraping
- **Hybrid retrieval** — dense vector search (Voyage AI) + BM25 keyword search fused via Reciprocal Rank Fusion (RRF), reranked by a cross-encoder
- **Relevance gate** — refuses to answer if no retrieved chunk clears a calibrated relevance threshold (fail-closed, not fail-open)
- **ReAct analysis agent** — explores the ingested knowledge base with tools (`list_pages`, `read_page`, `search_content`) before synthesizing a report
- **Structured, grounded output** — `FirstImpressionReport` Pydantic schema enforces citations structurally (not just via prompt); every `Observation` requires a `source_url`
- **Multi-provider LLM support** — Gemini (default, free tier) or Groq for the agent; Gemini or Anthropic Claude for Q&A
- **Eval harness** — retrieval precision/recall evals over a curated dataset with configurable relevance-threshold tuning

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns app name and environment |
| `POST` | `/ingest` | Crawl a public URL, chunk, embed, and store content |
| `POST` | `/ask` | Answer a question from ingested content with citations |
| `POST` | `/report` | Run the ReAct agent and produce a full First Impression report |

Interactive docs at **http://127.0.0.1:8000/docs** once the server is running.

---

## Architecture

```
POST /ingest
  └── robots.is_allowed()          # hard rule #1: public data only
      └── fetcher.crawl()          # trafilatura-powered HTML extraction
          └── chunker.chunk_text() # hand-written semantic chunker
              └── embeddings.embed_documents()  # Voyage AI
                  └── store.replace_all()       # ChromaDB

POST /ask
  └── pipeline.retrieve()          # embed → vector + BM25 → RRF → rerank
      └── relevance gate           # score < min_relevance → honest refusal
          └── qa.answer()          # LLM (Gemini / Anthropic) answers from chunks only

POST /report
  └── agent/react.py               # ReAct loop (list_pages / read_page / search_content)
      └── agent/report.py          # synthesis → FirstImpressionReport (structured output)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI + Uvicorn |
| Embeddings | Voyage AI (`voyage-3.5`) |
| Vector store | ChromaDB |
| Keyword search | BM25 (`rank-bm25`) |
| Reranking | Voyage AI reranker |
| LLM — Q&A | Gemini 2.5 Flash / Claude (Anthropic) |
| LLM — Agent | Gemini 2.5 Flash / Groq (`llama-3.3-70b-versatile`) |
| HTML extraction | `trafilatura` |
| Config | `pydantic-settings` + `.env` |
| Containerization | Docker + Docker Compose |

---

## Quickstart

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- API keys — see [Configuration](#configuration) below

```sh
# 1. Clone and install
git clone https://github.com/keshav9926/First-Impression-.git
cd First-Impression-

# 2. Install dependencies
uv sync

# 3. Configure environment
copy .env.example .env
# → open .env and fill in at minimum: VOYAGE_API_KEY + GEMINI_API_KEY

# 4. Start the dev server
uv run uvicorn app.main:app --reload
```

Then open:
- **http://127.0.0.1:8000/docs** — interactive Swagger UI
- **http://127.0.0.1:8000/health** — liveness check

### Typical workflow

```sh
# 1. Ingest a public site (crawls up to 15 pages by default)
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.example.com", "max_pages": 15}'

# 2. Ask a question with citations
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this product do?"}'

# 3. Generate a full First Impression report
curl -X POST http://127.0.0.1:8000/report
```

---

## Configuration

Copy `.env.example` to `.env` and set the values:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VOYAGE_API_KEY` | ✅ | — | Voyage AI key — free tier at [dashboard.voyageai.com](https://dashboard.voyageai.com) |
| `GEMINI_API_KEY` | ✅ (default) | — | Google AI Studio key — free tier at [aistudio.google.com](https://aistudio.google.com) |
| `ANTHROPIC_API_KEY` | When `LLM_PROVIDER=anthropic` | — | Anthropic key |
| `GROQ_API_KEY` | When `AGENT_PROVIDER=groq` | — | Groq key — free tier at [console.groq.com](https://console.groq.com) |
| `LLM_PROVIDER` | — | `gemini` | `gemini` or `anthropic` for `/ask` |
| `AGENT_PROVIDER` | — | `gemini` | `gemini` or `groq` for `/report` agent |
| `GEMINI_MODEL` | — | `gemini-2.5-flash` | Model for `/ask` |
| `GEMINI_AGENT_MODEL` | — | `gemini-2.5-flash` | Model for `/report` agent |
| `EMBEDDING_MODEL` | — | `voyage-3.5` | Voyage embedding model |
| `MIN_RELEVANCE` | — | `0.45` | Reranker score threshold below which answers are refused |

> **Free-tier tip:** Groq is recommended for `AGENT_PROVIDER` on free tier — its rate limits handle the agent's many calls far better than Gemini's ~20 RPD free quota.

---

## Run Tests

```sh
uv run pytest
```

---

## Run in Docker

```sh
docker compose up --build
```

The compose file mounts `.env` automatically — no extra config needed.

---

## Run Retrieval Evals

```sh
# Score the retrieval pipeline against the eval dataset
uv run python evals/run_retrieval_eval.py

# Debug individual retrieval queries
uv run python evals/debug_retrieval.py
```

---

## Design Decisions (Build vs. Buy)

- **Chunking is hand-written** ([app/ingestion/chunker.py](app/ingestion/chunker.py)) as a deliberate learning artifact — ~60 testable lines whose strategy we fully own. LangChain's splitters (`RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`) are a drop-in alternative and the reasonable "buy" choice in a production rush; speed is identical and quality is comparable on clean docs text.

- **HTML extraction is bought** (`trafilatura`) — the inverse call: stripping boilerplate from arbitrary HTML is genuinely hard, so a battle-tested library wins.

- **Retrieval is hybrid** — pure vector search misses exact-match queries (product names, error codes); pure BM25 misses semantic paraphrases. RRF fusion + cross-encoder reranking gives the best of both without a training dataset.

- **Fail-closed relevance gate** — if the reranker scores all retrieved chunks below `min_relevance`, the system refuses to answer rather than hallucinate. Wrong-but-confident is the worst outcome when output is shown to third parties.

- **Structured output as a grounding mechanism** — the `FirstImpressionReport` Pydantic schema is passed directly as a Gemini `response_schema`. An `Observation` without a `source_url` is structurally impossible to produce — hard rule #2 (grounded output only) is enforced by the schema, not just the prompt.

---

## Build Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ | Repo skeleton, FastAPI, Docker, env config |
| 1 | ✅ | Public content ingestion, chunking, embeddings, ChromaDB, plain RAG Q&A |
| 2 | ✅ | Hybrid search (BM25 + vectors + RRF), reranking, relevance gate, retrieval evals |
| 3 | ✅ | ReAct analysis agent → structured `FirstImpressionReport` with citations |
| 4 | 🔜 | Multi-agent crew (researcher / user-sim / evaluator / skeptic) via LangGraph |
| 5 | 🔜 | Evals + guardrails: groundedness check (LLM-as-judge + RAGAS), prompt-injection filter |
| 6 | 🔜 | FastAPI streaming endpoint + live agent-step dashboard |
| 7 | 🔜 | Custom MCP server exposing the analyzer as a tool |
| 8 | 🔜 | Observability (Langfuse traces), finalize Docker deployment |

---

## License

MIT
