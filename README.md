# FIE — First Impression Engine

> **Nature doesn't guess. A first impression is always there — FIE makes it visible.**

FIE is a complete, end-to-end **agentic AI system** that reads a startup's public website the way a first-time visitor would, then produces a grounded, citation-backed report on **what the product is, what lands, what confuses, and what's missing** — reasoning about the *product itself*, not just the signup funnel. Every claim cites the exact public page it came from; if the evidence is too thin to be fair, FIE refuses rather than invents.

One autonomous pipeline — **plan → crawl → render → sanitize → index → hybrid-retrieve → multi-persona reasoning → schema-constrained synthesis → self-verification** — with no human in the loop.

This is **not** a chatbot, **not** single-shot RAG, and **not** a LangChain demo. It is an agentic system built from the ground up, where reliability, grounding, and evaluation got as much engineering as generation.

### ▸ Live demos

| Company | Report |
|---|---|
| **Vortexify** (`vortexify.ai`) | **[firstimpressione.netlify.app/vortexify](https://firstimpressione.netlify.app/vortexify)** |
| **KAINest** (`kainest.com`) | **[firstimpressione.netlify.app/kainest](https://firstimpressione.netlify.app/kainest)** |

---

## Contents

1. [What it does](#1-what-it-does)
2. [How it works — the pipeline](#2-how-it-works--the-pipeline)
3. [Built phase by phase](#3-built-phase-by-phase)
4. [Retrieval, in depth](#4-retrieval-in-depth)
5. [Trust & grounding](#5-trust--grounding)
6. [Reliability engineering](#6-reliability-engineering)
7. [Evaluation](#7-evaluation)
8. [Design decisions & trade-offs](#8-design-decisions--trade-offs)
9. [Run it](#9-run-it)
10. [The deliverable](#10-the-deliverable-one-page-per-company)
11. [Project structure](#11-project-structure)

---

## 1. What it does

- **Crawls** any public site (robots-compliant), **headless-renders** JS/SPA pages, and **reads product screenshots** with a vision model — so nothing on the page is invisible to it.
- **Explores** the site as an autonomous **ReAct agent** (it decides what to read and search, up to 40 steps), then judges the same evidence through a **3-persona panel** — technical evaluator · business buyer · first-time user.
- **Synthesizes** a structured `FirstImpressionReport` — product identity, likely new-user journey, friction, standout strengths, open questions, and forward-looking improvement ideas.
- **Self-verifies** every claim against its cited page and **drops anything unsupported** — the failure mode is a shorter report, never a wronger one.
- **Delivers** one static, shareable page per company (engineering-datasheet design), plus a paste-ready outreach draft.

**The problem it solves:** a product's first impression gates every downstream metric (sign-ups, trust, retention), yet the signal is scattered across a landing page, docs, pricing, security, and screenshots — and the founder is too close (curse of knowledge) to judge it objectively. Single-shot RAG can't help: it can answer a question, but it can't *explore* a site, hold *multiple perspectives*, or *verify* its own output. FIE is architected as a direct response to each of those gaps.

---

## 2. How it works — the pipeline

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
                     Embed (NVIDIA nemotron-3-embed-1b, 2048-dim) ──► Chroma (local, persistent)
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

**Design signature:** the stages where the system *decides* (explore, personas) are separate from the stages where it *verifies* (citation check, judge, contradiction check). The model that writes is never the model that fact-checks. Untrusted page text is sanitized *before* any LLM sees it, and every claim is audited *after* it's produced — everything risky is sandwiched between a scrub and a judge.

### Model roles

| Role | Model |
|---|---|
| Explore / personas / synthesis | NVIDIA failover chain (mode-selected — see [§6](#6-reliability-engineering)) |
| Groundedness judge | Gemini `3.1-flash-lite`, temperature 0, fail-open |
| Vision (screenshot captions) | `nemotron-3-nano-omni-30b` → 2-model VLM failover |
| Embeddings | `nvidia/nemotron-3-embed-1b` (2048-dim) |
| Rerank | Voyage `rerank-2.5-lite` cross-encoder (calibrated 0–1 gate) |
| Observability | Langfuse (optional; hard no-op without keys) |

---

## 3. Built phase by phase

Built one phase at a time, with a review gate between each — RAG foundations first, then agents, then the guardrails and delivery that make it trustworthy and shippable. Each phase below states **what shipped**, **why it exists**, and **the key trade-off**.

### Phase 0 · Skeleton
- **Shipped:** FastAPI app, Docker, typed env config (`pydantic-settings`), `uv` dependency management.
- **Why:** a typed config surface (`app/config.py`) makes every model, chain, and threshold env-overridable and inspectable from day one.
- **Tech:** FastAPI · Docker · pydantic-settings · uv · Python ≥3.12.

### Phase 1 · Ingestion & RAG
- **Shipped:** robots-compliant crawler → chunk → embed → Chroma, plus grounded Q&A over one company's site.
- **Why:** if the crawler can't see it, the whole system is blind to it — ingestion quality caps everything downstream.
- **Key decisions:** BFS crawl (mirrors how a visitor fans out; O(V+E), hard-capped at 300 pages) · **custom paragraph-aware chunker** at ~1600 chars with tail overlap (`RecursiveCharacterTextSplitter` from LangChain is the standard drop-in alternative — the custom version keeps the path light and fully inspectable) · robots.txt checked **first**, fail-closed.
- **Tech:** httpx · ChromaDB · embeddings.

### Phase 2 · Retrieval quality
- **Shipped:** hybrid **dense + BM25** → **RRF** fusion → **cross-encoder rerank** → calibrated relevance gate; a hit@5 / MRR eval harness.
- **Why:** vector search misses exact terminology (`SOC 2`, a SKU code); BM25 misses meaning (a query with no shared keywords for a page titled *Inventory Risk*). Their blind spots are largely disjoint, so fusing both raises recall; a cross-encoder then buys precision on the final ordering.
- **Key trade-off:** the cross-encoder is O(candidates), so RRF narrows ~40 raw hits → 10 *before* the expensive rerank — a ~4× cost cut on the priciest step. See [§4](#4-retrieval-in-depth).
- **Tech:** BM25Okapi · Reciprocal Rank Fusion · Voyage rerank.

### Phase 3 · Analysis agent
- **Shipped:** a single **ReAct** agent (`list_pages` / `read_page` / `search_content`) that explores autonomously, then a schema-constrained, fully-cited report.
- **Why:** the pages that matter differ per site (pricing here, security there); interleaving reason + act lets the agent adapt its path where a fixed pipeline can't. Bounded at 40 steps so exploration always terminates.
- **Key decisions:** typed function-calling (valid actions, not regex-parsed strings) · explore-then-synthesize (creative exploration, then a *separate* strict JSON writer) · context trimming to bound token cost across 40 steps.
- **Tech:** ReAct · tool-calling · Pydantic.

### Phase 4 · Persona panel
- **Shipped:** **LangGraph** fan-out/fan-in — three personas judge one shared exploration in parallel.
- **Why:** one perspective averages away real friction. **Explore once, judge thrice** — exploration is the expensive stage, so running it per-persona would triple cost *and* let the three disagree on facts. Instead they share one frozen evidence set; only cheap synthesis-only judgment fans out.
- **Tech:** LangGraph.

### Phase 5 · Guardrails
- **Shipped:** **groundedness judge** (LLM-as-judge), contradiction check, **prompt-injection sanitizer**, empty-evidence refusal (HTTP 409).
- **Why:** a citation only proves the model *pointed* at a page, not that the page *supports* the claim — the most dangerous error is a cited-but-unsupported one, because it looks trustworthy. The judge can only **drop**, never add. See [§5](#5-trust--grounding).
- **Tech:** LLM-as-judge (Gemini, temperature 0).

### Phase 6 · Streaming + JS rendering
- **Shipped:** **SSE** live dashboard (crawl → report streamed step-by-step); **Playwright** headless render.
- **Why:** static crawls silently mis-read Framer/Webflow/SPA sites (an empty `<div id="root">`) — always rendering was the single biggest accuracy fix. SSE is the minimal correct transport because progress is strictly one-way (server → browser); WebSockets would be duplex overkill.
- **Tech:** Server-Sent Events · Playwright.

### Phase 7 · MCP server
- **Shipped:** the same pipeline exposed as **MCP tools** (stdio) for Claude Desktop / Claude Code / IDEs.
- **Why:** the MCP tools delegate to the *same functions* as the HTTP routes, so HTTP and MCP **can't drift** — one implementation, two front doors.
- **Tech:** FastMCP.

### Phase 8 · Observability & Docker
- **Shipped:** **Langfuse** tracing (one span tree per run, generations auto-captured); a production container with Chromium + a persistent vector volume.
- **Why:** a 40-step agent making dozens of model calls is undebuggable without tracing — you need to replay *why*, not guess. Made a hard no-op without keys so it never becomes a dependency that can break a run.
- **Tech:** Langfuse · Docker.

### Phase 9 · Vision
- **Shipped:** a **VLM** captions product screenshots the text extractor is blind to, with a multi-model failover chain.
- **Why:** sites communicate through images; without captions FIE would emit "no product screenshots" on a page full of them — a false negative that destroys trust. Captions ride into chunk metadata so the agent, personas, and judge all *see* what a screenshot shows.
- **Tech:** `nemotron-3-nano-omni-30b` VLM (→ 2-model failover).

### Phase 10 · Reliability & delivery
- **Shipped:** a multi-model **NVIDIA failover pool** (circuit breaker, per-minute vs per-day 429 intelligence, DEGRADED-deployment recovery, adaptive timeouts) · a product-substance prompt reframe · **Netlify** publishing.
- **Why:** free-tier endpoints are flaky and one report fires dozens of calls — no single bad response may kill a run. See [§6](#6-reliability-engineering).
- **Tech:** multi-LLM orchestration · Netlify.

---

## 4. Retrieval, in depth

A single retrieval strategy fails under real-world website evaluation, so FIE runs a **funnel**: cheap-and-wide recall first, expensive-and-narrow precision last.

```
 query
   ├── dense vectors (top 20)     semantic · synonyms · intent
   └── BM25 keyword  (top 20)     exact tokens: SOC2 · OEE · SAML · model names
          │
          ▼   Reciprocal Rank Fusion (k=60, guaranteed seats = 3 per list)
      10 candidates
          │
          ▼   cross-encoder rerank (Voyage rerank-2.5-lite → calibrated 0–1)
      top 5
          │
          ▼   relevance gate:  score ≥ 0.30  else  REFUSE (fail-closed)
     grounded answer, with a source URL on every sentence
```

- **Why RRF:** dense cosine scores and BM25 scores live on incomparable scales, so you can't just add them. RRF combines *ranks*, not scores — robust and near parameter-free. **Guaranteed seats** ensure the top hits from *either* retriever always get a candidate slot, so a strong #1 from one arm can't be buried by consensus with the other.
- **Why a cross-encoder:** it scores the query and a chunk *together* (not as two independent vectors), which is far more precise on the final ordering — and the LLM reads better-ranked chunks first. It's applied to ~10 candidates, not 100, to keep it cheap.
- **Why the gate:** `min_relevance = 0.30` is a **conservative safety floor** (fail-closed): if nothing clears it, `/ask` answers "no relevant content" *without* calling the LLM. In both eval runs, obvious junk scored well below 0.30 — the floor is validated against data, not vibes.

---

## 5. Trust & grounding

FIE is designed around one rule: **never say anything about a company that its public pages don't support.**

**Grounding vs groundedness** — the distinction that runs the whole system. *Grounding* (input side) means giving the model real evidence: retrieval, citations, a `source_url` on every chunk. *Groundedness* (output side) means **verifying it actually used that evidence** — a separate adversarial pass reads each finished claim next to its cited page and drops anything unsupported. A citation alone only proves the model *pointed* at a page.

| Guard | What it does |
|---|---|
| **robots.txt gate** | Checked *before* any request. Disallowed → no fetch, ever. Public pages only — no login areas, no scraping behind auth |
| **Prompt-injection sanitizer** | Page text is scrubbed of instruction-like content before it ever reaches an LLM |
| **Structural citations** | The `FirstImpressionReport` schema *requires* a `source_url` on every observation — uncited claims cannot exist (grounding enforced by the type system) |
| **Citation verification** | Any claim citing a page that was never ingested is dropped in code, not by prompt |
| **Groundedness judge** | A second adversarial LLM pass (Gemini, temp 0) reads each claim next to its cited page's actual text and drops unsupported ones |
| **Contradiction check** | Uncited statements (persona impressions, open questions) are checked against *all* page text — "X is not mentioned" is dropped when the site does mention X (caught live: a claimed "no SOC 2" vs a site's "SOC 2 audit in progress") |
| **Visual-evidence metadata** | Image alt-text/filenames and vision captions are captured as metadata, preventing false "no product screenshots" claims about pages full of dashboard shots |
| **Empty-evidence refusal** | A robots-blocked or dead crawl (store < 200 chars) produces HTTP 409, never a fabricated report |
| **Relevance gate** | Retrieval refuses to answer when nothing clears the calibrated floor (fail-closed) |
| **Judge determinism** | The fact-check pass runs at temperature 0 — same evidence, same verdicts |

**The invariant:** the verification layer can only *remove* content, never introduce it. So the worst case of an over-zealous judge is a *shorter* report, never a *wronger* one. If the judge model itself is unavailable the report still ships — but with an explicit scope-note caveat that the automated fact-check didn't run (fail-open, surfaced).

---

## 6. Reliability engineering

### Failover pipelines

`→` means *"if this model fails or produces no valid report, run the next one."* Every LLM call in a run (explore, personas, synthesis) inherits the selected chain. Values below are the finalized chains from `app/config.py` (2026-07-19 bake-off) — the authoritative source.

| Mode | Chain | Character |
|---|---|---|
| **normal** (default) | DeepSeek-V4-Pro → V4-Flash → Nemotron-3-Ultra | Fast, reliable |
| **deep** | **GLM-5.2** → DeepSeek-V4-Pro → V4-Flash → Nemotron-3-Ultra | Accuracy-first, no time budget |

All models run on the NVIDIA API (one `nvapi-` key); Mistral-Medium is configured as an additional extraction fallback, and Gemini/Groq (separate keys) remain the *deep* rate-limit insurance. Two nuances worth knowing:

- **`pool_prefer = "dspro"`** — DeepSeek-V4-Pro leads the normal chain (2026-07-18 bake-off: cleanest full-pipeline run, rich synthesis, ~8 min vs GLM's ~30).
- **DeepSeek-V4-Pro cannot explore** (its tool-calling fails), so the pool *skips it* whenever tools are requested and uses GLM/Nemotron for the ReAct loop. It still leads persona/synthesis (no-tools) calls.
- **Quality-failover** — a synthesis whose JSON doesn't validate against the report schema falls through to the next model. "Responded" and "responded usefully" are different bars.

### The LLM pool

Free-tier LLM endpoints are flaky; a single report fires dozens of calls, so one bad response must never kill a run. The pool (`app/agent/llm_pool.py`) fails over — or retries in place — on every failure mode observed live:

- **`DEGRADED` deployment (400)** — provider took a model offline → trip it and fail over (not a hard error).
- **Rate limits** — a per-**minute** 429 sleeps the server's `Retry-After`; a per-**day** 429 switches provider immediately (waiting won't help today).
- **5xx / empty completions / connection blips / intermittent 404** (NIM cold-scale) — retry with backoff, then fail over.
- **Malformed tool-calls (`tool_use_failed`)** — re-ask a few times before moving on.
- **Circuit breaker** — a giving-up provider is benched (15 min for a daily cap, 1 min for a transient throttle) so the loop stops re-probing a dead model.
- **Adaptive timeouts** — 300s for slow reasoning models / deep mode, 60s for fast paths.
- **Tolerant JSON parsing** — reasoning models wrap output in `<think>` blocks, fences, or a single-key object; the parser unwraps all three before validating.

**Vision has its own mirror of this:** three VLMs tried in order per image (`omni-30b` → `llama-3.2-90b-vision` → `nemotron-nano-vl-8b`), first successful caption wins — because the NVIDIA per-worker concurrency 503 usually only saturates one deployment at a time.

---

## 7. Evaluation

`evals/` holds the harnesses that drove the model and threshold choices. The point of this section is honesty: **every model and knob was chosen by a measured bake-off, not by hype** — and no headline metric is quoted that isn't backed by a committed run.

- `model_bakeoff.py` — multi-model bake-off (real companies, LLM-referee scoring) that produced the chains above. The *ranking* becomes the failover order — the same eval that picks the primary engineers the reliability chain behind it.
- `embed_rerank_bakeoff.py` — embedding/rerank provider comparison. Kept Voyage rerank for its calibrated 0–1 score (what makes the gate meaningful); moved embeddings to NVIDIA for speed at a bake-off MRR tie.
- `vision_bakeoff.py` — VLM comparison on real product dashboards (picked `omni-30b` on speed + accuracy over llama-90b-vision, gemma, and the nemotron VL 8b/12b).
- `run_retrieval_eval.py` — measures **hit@5 / MRR / false-answer rate** over a labeled set (`retrieval_eval.json`: 20 answerable questions — 2 per major page, rephrased/synonym/angle — plus 10 unanswerable, over 10 pages / 39 chunks). It runs the vector-only baseline against the hybrid pipeline on the same set, and the unanswerable set stress-tests the false-answer rate (target 0%). Exact hit@5/MRR are re-run per ingested site rather than frozen as a headline number.
- `run_report.py` / `run_deep_reports.py` / `rejudge_reports.py` — production runs on real companies + guard-pass re-application.

```bash
uv run python -m pytest tests/ -q     # 85 tests, no network
```

Tests cover: crawling/robots, sanitizer, chunking, retrieval fusion + gate, agent failover chains, panel merging, judge (support + contradiction + truncation salvage + fail-open), API endpoints, SSE streaming, and MCP wrappers.

---

## 8. Design decisions & trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **Playwright over Selenium** | Async API, auto-waiting, bundled Chromium, reliable screenshots | Slow + RAM-heavy → bounded by page/depth caps |
| **Always render** | Static crawls silently mis-read SPA sites | Every page pays render latency (fine for a batch analyst) |
| **Custom chunker over LangChain** | ~60 lines, paragraph-aware, fully inspectable, zero dep weight | Re-implements what `RecursiveCharacterTextSplitter` offers |
| **ChromaDB over FAISS** | First-class metadata + filtering + persistence (needed for citations) | Not distributed, not for billions of vectors — fine at one-site scale |
| **Hybrid retrieval** | Dense and lexical search fail on opposite queries | Higher latency → mitigated by the RRF→rerank funnel |
| **Cross-encoder rerank** | Highest-precision ranking; reads query+chunk together | O(candidates) → rerank 10, not 100 |
| **ReAct agent** | Adapts the exploration path per site | Needs hard constraints (step cap, typed tools) to stay reliable |
| **Persona panel (LangGraph)** | Multiple viewpoints surface friction one prompt averages away | More orchestration complexity |
| **Groundedness judge** | Citations don't guarantee support | One extra LLM call per report (fail-open) |
| **SSE over WebSockets** | Progress is one-way; SSE is the minimal correct fit | No client→server channel (not needed) |
| **FastAPI over Flask** | Native async for an I/O-bound pipeline; typed I/O + free OpenAPI | — |

**Product-substance reframe (Phase 10):** prompts push the agent to form a genuine view on the *product itself* — its core idea, what's distinctive, the philosophy the site reveals — and to make improvement ideas about product/positioning/narrative, not only "add a pricing table." Still fully grounded: sharper interpretation of real evidence, never invention. Reports credit what works first and phrase friction observationally ("a first-time visitor may hesitate here") — they're sent to the founders themselves.

---

## 9. Run it

```bash
# 1. deps (Python ≥3.12)
uv sync

# 2. keys — .env
NVIDIA_API_KEY=nvapi-...     # the whole LLM chain + embeddings + vision
VOYAGE_API_KEY=pa-...        # reranker
# optional: GEMINI_SECONDACC_API_KEY (groundedness judge / deep fallback)
# optional: LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_BASE_URL
# optional (publishing): NETLIFY_TOKEN / NETLIFY_PROJECT_ID

# 3. run
uv run uvicorn app.main:app --reload
# live dashboard at http://127.0.0.1:8000  ·  API docs at /docs
```

Docker: `docker compose up --build`

### HTTP API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/ingest` | Crawl a public URL → chunk → embed → store |
| `POST` | `/ask` | Grounded Q&A over the ingested site, with citations |
| `POST` | `/report?panel=true&deep=false` | Full report; `deep=true` selects the accuracy-first chain |
| `GET` | `/analyze/stream?url=...&deep=false` | One-call crawl+report with live SSE progress events |
| `GET` | `/` | Live streaming dashboard |

### MCP server

The same pipeline over the Model Context Protocol (stdio) for Claude Desktop / Claude Code / IDEs:

```bash
uv run python -m app.mcp_server
```

Tools: `analyze_first_impression(url, max_pages, panel, deep)` · `ask_ingested(question)` · `ingestion_status()`.

---

## 10. The deliverable: one page per company

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

Scores on the page are **derived from real signals** (persona verdicts, strength/friction balance, crawl coverage) — never invented. Founders receive a link, not a file; viewing costs zero backend. `evals/build_outreach_xlsx.py` builds `outreach.xlsx` — a paste-ready, credit-first email draft distilled from each verified report.

---

## 11. Project structure

```
presentation.pptx    16:9 Widescreen technical presentation slide deck
CHANGELOG.md         Master architectural decision log & post-mortems
README.md            Comprehensive project documentation & guide
build_pptx.py        PowerPoint generation script from markdown
app/
  main.py            FastAPI app: ingest / ask / report / SSE stream / dashboard
  mcp_server.py      MCP front door (stdio) — same pipeline, no drift
  config.py          All knobs (models, chains, thresholds) — env-overridable
  schemas.py         FirstImpressionReport & friends (citations required by type)
  observability.py   Langfuse tracing (no-op without keys)
  ingestion/         fetcher (crawl) · render (Playwright) · vision (VLM) · sanitize · chunker · robots
  rag/               store (Chroma) · embeddings · keyword (BM25) · fusion (RRF) · rerank · pipeline · qa
  agent/             llm_pool (chains/failover) · react (ReAct loop) · groq_driver (synthesis) ·
                     tools · personas · panel (LangGraph) · judge · grounding · prompts · report
web/
  report.html        The shareable report page template (single REPORT object)
  render_report.py   report JSON → static per-company page
  deploy.py          publish web/dist/ to Netlify
evals/               bake-offs, retrieval evals, production runs, outreach builder
reports/             verified report JSONs per company (git-ignored)
tests/               85 offline tests
```

---

*FIE is an engineering study in reliable AI systems: retrieval, agentic reasoning, evaluation, grounding, safety, multi-agent orchestration, observability, and LLM infrastructure — where the hard part was never calling an LLM, but making dozens of unreliable calls, over untrusted input, produce something a founder can trust, and being able to prove it.*
