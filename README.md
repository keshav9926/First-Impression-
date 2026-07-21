# FIE — First Impression Engine

> **Nature doesn't guess. A first impression is always there — FIE makes it visible.**

FIE is a complete, end-to-end **agentic AI system** that reads a startup's public website the way a first-time visitor would, then produces a grounded, citation-backed report on **what the product is, what lands, what confuses, and what's missing** — reasoning about the *product itself*, not just the signup funnel. Every claim cites the exact public page it came from; if the evidence is too thin to be fair, FIE refuses rather than invents.

One autonomous pipeline — **plan → crawl → render → sanitize → index → hybrid-retrieve → multi-persona reasoning → schema-constrained synthesis → self-verification** — with no human in the loop.

### ▸ Live demos

| Company | Report |
|---|---|
| **Vortexify** (`vortexify.ai`) | **[firstimpressione.netlify.app/vortexify](https://firstimpressione.netlify.app/vortexify)** |
| **KAINest** (`kainest.com`) | **[firstimpressione.netlify.app/kainest](https://firstimpressione.netlify.app/kainest)** |

---

## What it does

- **Crawls** any public site (robots-compliant), **headless-renders** JS/SPA pages, and **reads product screenshots** with a vision model — so nothing on the page is invisible to it.
- **Explores** the site as an autonomous **ReAct agent** (decides what to read and search), then judges the same evidence through a **3-persona panel** (technical evaluator · business buyer · first-time user).
- **Synthesizes** a structured `FirstImpressionReport` — product identity, likely new-user journey, friction, standout strengths, open questions, and forward-looking improvement ideas.
- **Self-verifies** every claim against its cited page and **drops anything unsupported** — the failure mode is a shorter report, never a wronger one.
- **Delivers** one static, shareable page per company (engineering-datasheet design), plus a paste-ready outreach draft.

---

## Why it's trustworthy

FIE is designed around one rule: **never say anything about a company that its public pages don't support.**

| Guard | What it does |
|---|---|
| **robots.txt gate** | Checked *before* any request. Disallowed → no fetch, ever. Public pages only — no login areas, no scraping behind auth |
| **Prompt-injection sanitizer** | Page text is scrubbed of instruction-like content before it ever reaches an LLM |
| **Structural citations** | The `FirstImpressionReport` schema *requires* a `source_url` on every observation — uncited claims cannot exist |
| **Citation verification** | Any claim citing a page that was never ingested is dropped in code, not by prompt |
| **Groundedness judge** | A second adversarial LLM pass reads each claim next to its cited page's actual text and drops unsupported ones |
| **Contradiction check** | Uncited statements (persona impressions, open questions) are checked against *all* page text — "X is not mentioned" is dropped when the site does mention X (caught live: a claimed "no SOC 2" vs a site's "SOC 2 audit in progress") |
| **Visual-evidence metadata** | Image alt-text/filenames and vision captions are captured as metadata, preventing false "no product screenshots" claims about pages full of dashboard shots |
| **Empty-evidence refusal** | A robots-blocked or dead crawl produces HTTP 409, never a fabricated report |
| **Relevance gate** | Retrieval refuses to answer when nothing clears a calibrated relevance floor (fail-closed) |
| **Judge determinism** | The fact-check pass runs at temperature 0 — same evidence, same verdicts |

---

## Architecture

```
URL ──► robots.txt gate ──► Crawl (httpx, BFS, same-domain)
                              │  always headless-render (Playwright) so JS/SPA
                              ▼  nav + product screenshots are actually captured
                     Vision (VLM captions product screenshots, 3-model failover)
                              ▼
                     Sanitize (prompt-injection scrub)
                              ▼
                     Chunk (~1600 chars, overlap; heading/CTA/image metadata)
                              ▼
                     Embed (NVIDIA nemotron-3-embed-1b) ──► Chroma (local, persistent)
                              ▼
        ┌─────────────── Hybrid retrieval ───────────────┐
        │  dense vectors + BM25 ──► RRF fusion ──► rerank │
        │  (Voyage cross-encoder) ──► relevance gate      │
        └─────────────────────────────────────────────────┘
                              ▼
                     ReAct explore agent  (up to 40 steps)
                     (list_pages / read_page / search_content)
                              ▼
                     Persona panel (LangGraph fan-out / fan-in)
                     technical evaluator · business buyer · first-time user
                              ▼
                     Synthesis (schema-constrained JSON, per-model quality failover)
                              ▼
                     Guards: citations ─► groundedness judge ─► contradiction check
                              ▼
                     FirstImpressionReport ──► static web page / MCP / API / outreach
```

### Failover pipelines

`→` means *"if this model fails or produces no valid report, run the next one."* Every LLM call in a run (explore, personas, synthesis, judge) inherits the selected chain.

| Mode | Chain | Character |
|---|---|---|
| **normal** (default) | Nemotron-3-Ultra → DeepSeek-V4-Pro → V4-Flash | Fast, reliable |
| **deep** | **DeepSeek-V4-Pro** → V4-Flash → Nemotron-3-Ultra | Accuracy-first, no time budget |

All models run on the NVIDIA API (one key); GLM-5.2 and Mistral-Medium are configured as additional fallbacks. Failover includes **quality-failover** — a synthesis whose JSON doesn't validate against the report schema falls through to the next model.

### Reliability engineering (the LLM pool)

Free-tier LLM endpoints are flaky; a single report fires dozens of calls, so one bad response must never kill a run. The pool (`app/agent/llm_pool.py`) fails over — or retries in place — on every failure mode observed live:

- **`DEGRADED` deployment (400)** — provider took a model offline → trip it and fail over (not a hard error).
- **Rate limits** — per-**minute** 429 sleeps the server's `Retry-After`; per-**day** 429 switches provider immediately (waiting won't help today).
- **5xx / empty completions / connection blips / intermittent 404** (NIM cold-scale) — retry with backoff, then fail over.
- **Malformed tool-calls (`tool_use_failed`)** — re-ask a few times before moving on.
- **Circuit breaker** — a giving-up provider is benched (15 min for a daily cap, 1 min for a transient throttle) so the loop stops re-probing a dead model.
- **Adaptive timeouts** — 300s for slow reasoning models / deep mode (no time budget), 60s for fast paths.
- **Tolerant JSON parsing** — reasoning models wrap output in `<think>` blocks, fences, or a single-key object; the parser unwraps all three before validating.

### Model roles

| Role | Model |
|---|---|
| Explore / personas / synthesis / judge | Chain above (mode-selected) |
| Vision (screenshot captions) | `nemotron-3-nano-omni-30b` → 2-model VLM failover |
| Embeddings | `nvidia/nemotron-3-embed-1b` (2048-dim) |
| Rerank | Voyage `rerank-2.5-lite` cross-encoder (calibrated gate) |
| Observability | Langfuse (optional; hard no-op without keys) |

---

## Quickstart

```bash
# 1. deps (Python ≥3.12)
uv sync

# 2. keys — .env
NVIDIA_API_KEY=nvapi-...     # the whole LLM chain + embeddings + vision
VOYAGE_API_KEY=pa-...        # reranker
# optional: LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_BASE_URL
# optional (publishing): NETLIFY_TOKEN / NETLIFY_PROJECT_ID

# 3. run
uv run uvicorn app.main:app --reload
# live dashboard at http://127.0.0.1:8000  ·  API docs at /docs
```

Docker:

```bash
docker compose up --build
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/ingest` | Crawl a public URL → chunk → embed → store |
| `POST` | `/ask` | Grounded Q&A over the ingested site, with citations |
| `POST` | `/report?panel=true&deep=false` | Full report; `deep=true` selects the accuracy-first chain |
| `GET` | `/analyze/stream?url=...&deep=false` | One-call crawl+report with live SSE progress events |

## MCP server

The same pipeline, exposed over the Model Context Protocol (stdio) for Claude Desktop / Claude Code / IDEs:

```bash
uv run python -m app.mcp_server
```

Tools: `analyze_first_impression(url, max_pages, panel, deep)` · `ask_ingested(question)` · `ingestion_status()`.

---

## The deliverable: one page per company

Each analyzed company gets a single, static, shareable report page (engineering-datasheet design — monochrome, mono-forward, print-like):

```
pipeline (deep run)  ──►  reports/<company>.json      # verified report + run meta
web/report.html      ──►  the design template          # reads everything from `var REPORT`
web/render_report.py ──►  web/dist/<company>.html      # real data injected into the template
web/deploy.py        ──►  Netlify static host          # clean link per founder (--slug = unguessable)
```

> Report JSONs contain third-party company analysis and are **never committed**
> (`reports/`, `web/dist/`, and `outreach.xlsx` are git-ignored). Rendered pages are hosted as
> static, unindexed links delivered to that founder only (or exported to a static PDF).

```bash
python -m web.render_report            # render every reports/*.json
python -m web.render_report vortexify  # just one
python -m web.deploy                   # publish web/dist/ to Netlify → prints each link
```

Scores on the page are **derived from real signals** (persona verdicts, strength/friction balance, crawl coverage) — never invented. Founders receive a link, not a file; viewing costs zero backend.

`evals/build_outreach_xlsx.py` builds `outreach.xlsx` — company, founder, contact, and a paste-ready, credit-first email draft distilled from each verified report.

---

## Design decisions

- **Reads the product, not just the funnel.** Prompts push the agent to form a genuine view on the product itself — its core idea, what's distinctive, the philosophy the site reveals — and to make improvement ideas about *product/positioning/narrative*, not only "add a pricing table." Still fully grounded: sharper interpretation of real evidence, never invention.
- **Explore-then-synthesize.** Free-form ReAct exploration first (the agent decides what to read/search), then a separate schema-constrained synthesis pass. Creativity where it helps, structure where it matters.
- **One store, one company.** Chroma holds the company being analyzed; each ingest starts clean. Reports are frozen to JSON + static HTML at generation time, so nothing depends on the store afterward.
- **Custom chunker over LangChain.** Chunking is ~60 lines: paragraph-aware packing to ~1600 chars with tail overlap. LangChain's `RecursiveCharacterTextSplitter` is the standard alternative and would slot in directly — the custom version was chosen to keep the ingestion path dependency-light and fully inspectable, not because the alternative wouldn't work.
- **Judge can only drop, never add.** The verification layer removes unsupported or contradicted content; it cannot introduce new claims. Failure mode is a shorter report, not a wronger one.
- **Fail-open judge, surfaced.** If the judge model is unavailable the report still ships — but with an explicit scope-note caveat that the automated fact-check didn't run.
- **Kind but honest.** Reports credit what works first, never manufacture positivity, and phrase friction observationally ("a first-time visitor may hesitate here") — they're sent to the founders themselves.

## Evals

`evals/` contains the harnesses that drove the model and threshold choices:

- `model_bakeoff*.py` — multi-model bake-off (real companies, LLM-referee scoring) that produced the chains above
- `embed_rerank_bakeoff.py` — embedding/rerank provider comparison (kept Voyage rerank for its calibrated score scale; moved embeddings to NVIDIA for speed)
- `vision_bakeoff.py` — VLM comparison on real product dashboards (picked omni-30b on speed + accuracy)
- `run_retrieval_eval.py` — hit@5 / MRR over a labeled retrieval set
- `run_report.py` / `run_deep_reports.py` / `rejudge_reports.py` — production runs on real companies + guard-pass re-application
- `build_outreach_xlsx.py` — the outreach workbook

## Tests

```bash
uv run python -m pytest tests/ -q     # 85 tests, no network
```

Covers: crawling/robots, sanitizer, chunking, retrieval fusion + gate, agent failover chains, panel merging, judge (support + contradiction + truncation salvage + fail-open), API endpoints, SSE streaming, MCP wrappers.

## Project structure

```
app/
  main.py            FastAPI app: ingest / ask / report / SSE stream
  mcp_server.py      MCP front door (stdio) — same pipeline, no drift
  config.py          All knobs (models, chains, thresholds) — env-overridable
  schemas.py         FirstImpressionReport & friends (citations required by type)
  observability.py   Langfuse tracing (no-op without keys)
  ingestion/         fetcher (crawl+render+metadata) · vision · sanitize · chunker · robots
  rag/               store (Chroma) · embeddings · keyword (BM25) · pipeline (RRF+rerank+gate) · qa
  agent/             llm_pool (chains/failover) · groq_driver (ReAct+synthesis) ·
                     panel (LangGraph personas) · judge · grounding · prompts · tools · report
web/
  report.html        The shareable report page template (single REPORT object)
  render_report.py   report JSON → static per-company page
  deploy.py          publish web/dist/ to Netlify
evals/               bake-offs, retrieval evals, production runs, outreach builder
reports/             verified report JSONs per company (git-ignored)
tests/               85 offline tests
```
