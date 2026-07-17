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
- **Persona panel** — three distinct visitors (technical evaluator / business buyer / first-time user) judge the same evidence in parallel (LangGraph fan-out), so the report shows *who* bounces *where*
- **Structured, grounded output** — `FirstImpressionReport` Pydantic schema enforces citations structurally (not just via prompt); every `Observation` requires a `source_url`
- **Refuses on empty evidence** — a report is never fabricated from an empty/near-empty store (robots-blocked or dead crawl → HTTP 409, not a hallucinated report)
- **JS-site rendering** — static crawl escalates to a headless Playwright render when extraction is thin, so Framer/Webflow/SPA sites are readable
- **Guardrails** — prompt-injection sanitizer pre-chunking + a groundedness judge that drops claims the cited page doesn't support
- **Multi-provider LLM chain** — one NVIDIA NIM key drives a quality-first fallback chain (GLM-5.2 → DeepSeek-V4-Pro → Nemotron-3-Ultra → Mistral-Medium-3.5), with Gemini/Groq on separate keys as deep rate-limit insurance; automatic failover + circuit breaker + usage accounting
- **MCP server** — the analyzer is exposed over the Model Context Protocol (stdio), so any MCP client (Claude Desktop, Claude Code, an IDE) can call `analyze_first_impression` / `ask_ingested` as native tools — the same pipeline the HTTP API serves, no drift
- **Observability** — optional Langfuse tracing: each report is one span tree with every LLM call nested as a generation (model, token usage, latency). A hard no-op without keys, so nothing is paid for until you opt in
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

POST /report                       # ?panel=true → persona panel
  └── report.generate_report()     # refuses (409) if the store is empty/thin
      └── explore ONCE (tools)      # llm_pool: GLM → Nemotron → Mistral (DeepSeek-Pro skipped: no tools)
          └── panel (LangGraph)     # 3 personas judge the same evidence in parallel
              └── synthesize        # GLM → … → Gemini → FirstImpressionReport
                  └── apply_guards  # citation verify + groundedness judge + thin-crawl caveat

POST /analyze/stream               # SSE: full ingest → report as live agent-step events
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
| LLM — Agent (explore / persona / synthesis) | NVIDIA NIM chain: GLM-5.2 → DeepSeek-V4-Pro → Nemotron-3-Ultra → Mistral-Medium-3.5 |
| LLM — deep fallback | Gemini (`gemini-3.1-flash-lite` / `gemini-3-flash-preview`), Groq |
| LLM — Q&A (`/ask`) | Groq (default) / Gemini / Claude |
| JS rendering | Playwright (headless Chromium), lazy fallback |
| HTML extraction | `trafilatura` |
| Config | `pydantic-settings` + `.env` |
| Agent tool interface | MCP server (`mcp`, stdio) |
| Observability | Langfuse traces (optional) |
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
# → open .env and fill in at minimum: VOYAGE_API_KEY + NVIDIA_API_KEY

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
| `VOYAGE_API_KEY` | ✅ | — | Voyage AI embeddings — free tier at [dashboard.voyageai.com](https://dashboard.voyageai.com) |
| `NVIDIA_API_KEY` | ✅ | — | NVIDIA NIM — drives the whole agent chain (GLM-5.2 → DeepSeek → Nemotron → Mistral); free endpoints at [build.nvidia.com](https://build.nvidia.com) |
| `GEMINI_API_KEY` | Recommended | — | Deep fallback + native synthesis — [aistudio.google.com](https://aistudio.google.com) |
| `GEMINI_SECONDACC_API_KEY` | Optional | — | 2nd Google account → 2× Gemini fallback headroom |
| `GROQ_API_KEY` | Optional | — | Deep fallback — [console.groq.com](https://console.groq.com) |
| `ANTHROPIC_API_KEY` | When `LLM_PROVIDER=anthropic` | — | Anthropic key (for `/ask` only) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Optional | — | Both set → report runs are traced to Langfuse ([cloud.langfuse.com](https://cloud.langfuse.com)); absent → tracing is a no-op |
| `LANGFUSE_HOST` | — | `https://cloud.langfuse.com` | Set to `https://us.cloud.langfuse.com` for the US region |
| `EMBEDDING_MODEL` | — | `voyage-3.5` | Voyage embedding model |
| `MIN_RELEVANCE` | — | `0.45` | Reranker score threshold below which answers are refused |

> **Why one NVIDIA key for four models:** they're all free NVIDIA NIM endpoints on `integrate.api.nvidia.com`, so a single key fails over across GLM → DeepSeek-Pro → Nemotron → Mistral. Because they share one account quota, Gemini/Groq (different keys) stay as the real rate-limit insurance.

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

The compose file mounts `.env` automatically — no extra config needed. The image installs headless Chromium so JS-rendered sites work in-container, and the ChromaDB vector store lives on a named volume (`chroma_data`) so ingested content survives restarts. A `/health` healthcheck gates the container.

---

## Observability (optional)

Set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` + `LANGFUSE_BASE_URL` (free project at [cloud.langfuse.com](https://cloud.langfuse.com)) and every report run is traced following Langfuse's [instrumentation best practices](https://github.com/langfuse/skills):

- **LLM calls** are captured by the **Langfuse OpenAI drop-in** — model, token usage, cost, and latency are recorded automatically (no manual logging) for the whole NVIDIA/Gemini chain.
- **Correct observation types**: the explore loop and each persona are `agent` observations (so they show as distinct nodes in Langfuse's Agent Graph), retrieval is a `retriever`, and the root span's input/output is the ingested pages → the finished report.

So "which model actually answered", "how many tokens", "where did the 3 minutes go", and "which persona bounced" are answerable at a glance. Without the keys, tracing is a hard no-op ([app/observability.py](app/observability.py)) — no account or config required to run the app.

---

## Run as an MCP Server

The analyzer is also a [Model Context Protocol](https://modelcontextprotocol.io) server, so an MCP client (Claude Desktop, Claude Code, an IDE) can call it as a tool instead of over HTTP.

```sh
# Serve the tools over stdio (the transport local MCP clients speak)
uv run python -m app.mcp_server
```

Register it in your MCP client's config (paths are examples):

```json
{
  "mcpServers": {
    "first-impression": {
      "command": "uv",
      "args": ["run", "python", "-m", "app.mcp_server"],
      "cwd": "/absolute/path/to/First-Impression-"
    }
  }
}
```

**Tools exposed**

| Tool | What it does |
|------|--------------|
| `analyze_first_impression(url, max_pages=15, panel=True)` | Crawl → ReAct report → structured `FirstImpressionReport` (every claim cited) |
| `ask_ingested(question, top_k=5)` | Grounded Q&A over the most recently analyzed site |
| `ingestion_status()` | Whether a site is currently ingested (chunk count + source pages) |

Each tool delegates to the same pipeline functions the HTTP endpoints call, and returns a structured `{status, ...}` result — a robots-blocked or thin crawl refuses rather than fabricating a report, exactly as `/report` does.

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
| 4 | ✅ | Persona panel (technical / business / first-time) via LangGraph fan-out |
| 5 | ✅ | Guardrails: groundedness judge (LLM-as-judge) + prompt-injection sanitizer |
| 6 | ✅ | Playwright JS rendering, streaming `/analyze/stream` dashboard, multi-provider pool (circuit breaker + usage accounting), evidence-floor guard |
| 7 | ✅ | Custom MCP server (`app/mcp_server.py`) exposing the analyzer as stdio tools |
| 8 | ✅ | Observability (optional Langfuse traces), finalized Docker deployment (Chromium in-image + persisted vector store) |

---

## License

MIT
