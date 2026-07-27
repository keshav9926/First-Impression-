# First Impression Engine (FIE) — Technical Presentation & System Architecture

> **Master Slide Deck & Technical Blueprint**  
> *End-to-End System Evolution from Phase 0 to Phase 9*  
> **Target Audience**: Technical Mentors, System Architects, & Engineering Evaluation Panels

---

## Slide 1: Title & Executive Overview

### **First Impression Engine (FIE)**
*Autonomous Outside-In Product Analysis & Grounded Multi-Agent Evaluation Platform*

- **Presenter**: Engineering Lead / Developer
- **Scope**: Complete System Evolution (Phases 0 through 9)
- **Core Focus**: High-Reliability Agentic Systems, Hybrid RAG, Multi-Provider Failover Pools, & Grounded Verification

#### **Key System Metrics**
- **Architecture**: Hybrid RAG + ReAct Agent + LangGraph Parallel Panel + Multi-LLM Pool
- **Verification Stack**: 5-Layer Trust Pipeline (0 Unbacked Claims)
- **Scale Capability**: Dual-Provider Execution (Groq Explore + Gemini Final Eval Synthesis)
- **Latency / Performance**: 5-Step Cap ReAct Loop, Parallel Persona Fan-Out, Live SSE Streaming

---

## Slide 2: Problem Statement & Operating Philosophy

### **The Problem: Founder Blind Spots**
- Founders and product teams suffer from *curse of knowledge*—they know how their product works and cannot experience their own public site as a stranger.
- Traditional tools audit SEO, page speed, or surface UI elements, but **cannot evaluate product substance, value proposition clarity, or onboarding friction**.

### **Core Operating Principles (Hard Rules)**
1. **Rule #1 (Public Data Only)**: Respects `robots.txt` strictly. Analyzes only pre-signup, publicly accessible content.
2. **Rule #2 (Zero Hallucination Guarantee)**: Every claim must be grounded in ingested page text with verified source URLs. Unsupported claims are dropped by verification layers.
3. **Fail-Closed Retrieval & Fail-Open Verification**: Retrieval refuses to answer if relevance is below calibrated threshold ($<0.30$); verification layers drop invalid claims so failure results in a shorter report, never an inaccurate one.

---

## Slide 3: End-to-End System Architecture

```
[ USER / DASHBOARD / MCP CLIENT ]
               │
               ▼
   ┌──────────────────────┐
   │ FastAPI / MCP Server │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Robots.txt Gate      │  ◄── Rule #1 (Public Data Only)
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ BFS Crawl &          │  ◄── Escalates to Playwright Chromium
   │ Playwright JS Render │      on Thin Text Extraction (<1%)
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Injection Sanitizer  │  ◄── Strips Prompt-Injection Regex Lines
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Custom Paragraph     │  ◄── ~1600 chars, tail-overlap,
   │ Chunker              │      attaches Headings & CTAs
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Persistent Vector    │  ◄── Dense Vectors (Nemotron / Voyage)
   │ Store (ChromaDB)     │
   └──────────┬───────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                 HYBRID RETRIEVAL FUNNEL                     │
│  Dense Vector (Top 20)  │  Sparse BM25 Keyword (Top 20)     │
│             └───────────┴───────────┘                       │
│                         ▼                                   │
│            Reciprocal Rank Fusion (RRF)                     │
│               + Guaranteed Seats (top 3)                    │
│                         ▼                                   │
│            Cross-Encoder Re-ranker                          │
│               + Shifted Sigmoid Scaling                     │
│                         ▼                                   │
│            Calibrated Min Relevance Gate (≥ 0.30)           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
   ┌──────────────────────┐
   │ ReAct Exploration    │  ◄── Groq Llama-3.3 (Max 5 Steps)
   │ Agent                │      Section Maps on Truncation
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ LangGraph Parallel   │  ◄── 3 Parallel Personas: Tech Evaluator,
   │ Persona Panel        │      Business Buyer, First-Time User
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Gemini Synthesis     │  ◄── Native response_schema Enforcement
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ 5-Layer Trust Stack  │  ◄── Schema → Citation URL → Fact Judge
   │                      │      → Contradiction → Scope Caveats
   └──────────┬───────────┘
              │
              ▼
[ VERIFIED REPORT OUTPUT: Web HTML / PDF / SSE / MCP ]
```

---

## Slide 4: Phase 0 & 1 — Pipeline Foundation & Ingestion

### **Fast Same-Domain Crawling**
- **BFS Crawl Loop**: Implemented with `httpx`. Deduplicates URLs by canonicalizing (dropping query params `?ref=` and fragments `#section`).
- **Segment Exclusion**: Blocks authentication/account paths (`/login`, `/signup`) via path-segment matching rather than coarse substring matching.

### **Robots.txt Politeness Gate (`app/ingestion/robots.py`)**
- Uses `urllib.robotparser.RobotFileParser`. Caches parser instances per origin with a 1-hour TTL (`_TTL_SECONDS = 3600.0`).
- **Fail-Closed Design**: If fetching `robots.txt` encounters a network failure, the parser sets `disallow_all = True`. Access must be explicitly verified.

### **Custom Paragraph-Aware Chunker (`app/ingestion/chunker.py`)**
- **Why Custom over LangChain?**: Cuts framework overhead, preserves natural paragraph boundaries (`\n\n`), and binds page metadata (headings, CTAs) directly to chunks.
- **Algorithm**:
  1. Splits text into paragraph units.
  2. Greedily packs paragraphs up to `max_chars = 1600` (~400 tokens).
  3. Overlaps the last paragraph into the next chunk window to prevent edge context loss.

---

## Slide 5: Phase 2 — Hybrid Retrieval Funnel & Math

### **1. Dense Vectors vs. Sparse BM25**
- **Dense Vector Search**: Captures semantic intent using `nvidia/nemotron-3-embed-1b` (2048 dimensions). Uses asymmetric retrieval mode (`input_type="passage"` for documents, `input_type="query"` for questions).
- **Sparse BM25 Search**: Uses `rank_bm25` (`BM25Okapi`) with custom light stemming (`-ing`, `-es`, `-ed`, `-s`). Captures exact acronyms (`SOC2`, `OEE`, `SDK`).

### **2. Reciprocal Rank Fusion (RRF) & Guaranteed Seats**
$$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$
- **The RRF Bug & Fix**: Standard RRF consensus could omit a document ranked **#1 in Vector Search** if BM25 ranked it poorly due to stemming mismatches.
- **Engineering Comeback**: Implemented **Guaranteed Seats (`guaranteed_per_list = 3`)** in `fusion.py`. The top 3 candidates from *each* arm are guaranteed a seat in the candidate pool sent to the cross-encoder.

### **3. Shifted Sigmoid Reranker Normalization**
- NVIDIA Cross-Encoder Reranker outputs raw logits $L \in [-20, +5]$. Standard sigmoid compresses negative logits near 0.0.
- **Shifted Sigmoid Formula**:
  $$S_{\text{norm}} = \frac{1}{1 + e^{-\frac{L - (-10.0)}{3.0}}}$$
  Re-centers the distribution so relevant items map to $>0.30$, and junk falls below the calibrated **0.30 Relevance Gate**.

---

## Slide 6: Phase 3 — ReAct Exploration Loop & History Management

### **ReAct (Reason + Act + Observe) Agent**
- Driven by Groq (Llama-3.3-70B). Executes tool calls: `list_pages`, `read_page`, `search_content`.
- Hard-capped at **5 Steps** (`MAX_STEPS = 5`): Sufficient for `list_pages` (1) $\rightarrow$ `read_page` key pages (2-3) $\rightarrow$ `search_content` (1).

### **Solving the Unknown-Unknown: Section Maps**
- `read_page` outputs are capped at 4,000 characters. When truncated, the agent was blind to deeper page sections.
- **Solution**: Prepend an HTML Heading Section Map ($h_1..h_3$) to truncated `read_page` observations. The agent learns what sections exist deeper on the page and issues targeted `search_content` calls.

### **Bounded Context Window (`_trim_history`)**
- Prevents $O(N^2)$ token explosion during multi-step tool calls by keeping the 2 most recent observations verbatim and trimming older tool responses to 600 characters. Includes a **Repeat-Call Guard** to prevent duplicate argument calls.

---

## Slide 7: Phase 4 — LangGraph Multi-Persona Panel

### **LangGraph State Graph Topology**
- **Explore Once, Judge Thrice**: The ReAct agent explores the site *once*, producing a shared evidence base.
- **Parallel Fan-Out**: Graph fans out to 3 parallel persona nodes (*Technical Evaluator*, *Business Buyer*, *First-Time End User*).
- **Reducer**: Uses `operator.add` on the `impressions` state list to prevent concurrent write race conditions.

```
                  ┌──────────────────────────┐
                  │ ReAct Exploration Node   │
                  └────────────┬─────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │ Technical   │ │ Business    │ │ First-Time  │
        │ Evaluator   │ │ Buyer       │ │ End User    │
        └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
               └───────────────┼───────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │ Fan-In Merge Node        │
                  └──────────────────────────┘
```

### **The CTA Erasure Comeback**
- **Problem**: `favor_precision=True` in `trafilatura` stripped header/footer navigation debris, inadvertently deleting navbar `"Try for free"` CTA buttons. Personas falsely reported "no signup button".
- **Fix**: Built pre-cleaning HTML AST CTA Collectors (`_CtaCollector`) in `fetcher.py`. Extracted CTAs ride as chunk metadata and surface on every `read_page` call.

---

## Slide 8: Phase 5 — Multi-Layer Verification & Guardrails

### **5-Layer Trust Pipeline**
1. **Layer 1 (Schema Enforcement)**: Pydantic `FirstImpressionReport` schema requires `claim`, `evidence`, and `source_url` on every `Observation`.
2. **Layer 2 (Citation URL Matching)**: `enforce_citations` checks every `source_url` against canonical URLs in Chroma. Claims citing un-ingested URLs are dropped in code.
3. **Layer 3 (Adversarial Fact-Check Judge)**: `verify_groundedness` runs a temperature 0.0 LLM pass comparing claims against stored chunk text. Unsupported claims are dropped.
4. **Layer 4 (Contradiction Check)**: Uncited persona claims are checked against all stored text (e.g. claiming "no SOC 2" when text mentions "SOC 2 in progress").
5. **Layer 5 (Fail-Open Scope Caveat)**: If judge call encounters an API outage, the report still ships with an explicit scope caveat banner attached to `scope_note`.

### **Indirect Prompt Injection Defense (`app/ingestion/sanitize.py`)**
- Regex line stripping removes instruction-shaped language (`ignore previous instructions`, `you are now`) before chunking. Count returned in `IngestResponse.injection_lines_removed`.

---

## Slide 9: Phase 6 — Playwright JS Escalation & Live SSE Dashboard

### **The JS SPA "Black Hole" Failure & Comeback**
- **Failure**: Static HTTP crawler extracted only 368 chars from `trynarrative.com` (a Framer SPA). Report hallucinated that the startup had no pricing or product features.
- **Comeback**: Built **Playwright Chromium Headless Escalation** (`render.py`). If aggregate text < 1% or seed page < 1,200 chars, crawler automatically escalates to Playwright with auto-scrolling. Text extracted via `inner_text("body")`. Result: Text extraction jumped **368 chars $\rightarrow$ 1,706 chars**.

### **Live SSE Streaming Dashboard (`/analyze/stream`)**
- Uses Python `contextvars` (`app/events.py`) to create a thread-safe event bus.
- Pipeline runs in a background thread while the main worker streams real-time SSE progress events (`crawl`, `render.escalate`, `tool`, `persona`, `report.done`) to `static/index.html`.

---

## Slide 10: Phase 7 & 8 — MCP Server & Observability

### **Phase 7: Model Context Protocol (MCP) Server (`app/mcp_server.py`)**
- Exposes analyzer tools over stdio via `FastMCP`: `analyze_first_impression`, `ask_ingested`, `ingestion_status`.
- **Zero Code Drift**: Tools delegate directly to core FastAPI functions (`_ingest_site`, `generate_report`, `pipeline.retrieve`, `qa.answer`).

### **Phase 8: Langfuse OpenTelemetry Tracing (`app/observability.py`)**
- Zero-overhead no-op when unconfigured. Auto-instruments OpenAI client calls.
- Creates hierarchical span trees: Root `report_trace` span $\rightarrow$ `agent` subagent spans $\rightarrow$ `retriever` spans $\rightarrow$ `generation` leaf nodes.

### **Docker Container Hardening**
- Base image: `python:3.13-slim`. Pre-installs Playwright Chromium dependencies (`RUN playwright install --with-deps chromium`).
- Mounts named volume `chroma_data` for persistent vector storage.

---

## Slide 11: Phase 9 — Dual-Provider Split & LLM Pool Hardening

### **The 429 Rate-Limit Collapse & Dual-Provider Architecture**
- **Problem**: Gemini free tier (~20 req/day) exhausted on single reports. Groq free tier hit per-minute token limits on long histories.
- **Solution**:
  - **`/ask`**: Uses **Groq (Llama-3.3-70B)** (generous free RPM).
  - **`/report` Phase A (Explore)**: Uses **Groq** for high-burst ReAct tool calls.
  - **`/report` Phase B (Synthesis)**: Uses **Google Gemini (2.5-Flash)** via native `response_schema`.

```
        ┌─────────────────────────────────────────────────────────┐
        │                    WORKLOAD ROUTER                      │
        └───────────┬─────────────────────────────────┬───────────┘
                    │                                 │
                    ▼                                 ▼
   ┌─────────────────────────────────┐   ┌─────────────────────────┐
   │          GROQ DRIVER            │   │      GEMINI DRIVER      │
   │       (Llama-3.3-70B)           │   │     (Gemini 2.5 Flash)  │
   │ • Fast /ask Q&A                 │   │ • Final Report Eval     │
   │ • ReAct Explore Loop (≤5 steps) │   │   Synthesis             │
   │ • Persona Panel Nodes           │   │ • Native Schema JSON    │
   └─────────────────────────────────┘   └─────────────────────────┘
```

### **LLM Resilience Pool (`app/agent/llm_pool.py`)**
- Multi-provider chain across NVIDIA NIM (`GLM-5.2` $\rightarrow$ `DeepSeek-V4-Pro` $\rightarrow$ `Nemotron-3-Ultra`).
- Differentiates per-minute 429s (sleep Retry-After) from per-day quota caps (15-min circuit breaker). Handles HTTP 400 DEGRADED & 404 container cold-scaling.

---

## Slide 12: Technical Trade-Offs & Decision Matrix

| Domain | Option Selected | Alternative Considered | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Chunking** | **Custom Paragraph Chunker** | LangChain `RecursiveCharacterTextSplitter` | Preserves paragraph boundaries (`\n\n`), cuts dependency bloat, binds headings & CTAs. |
| **Retrieval** | **Hybrid RRF + Reranker** | Dense Vector Only | Vectors handle semantic intent; BM25 handles proper nouns/acronyms (`SOC2`). Reranker rescores fused hits. |
| **LLM Execution** | **Dual-Provider Split** | Single LLM Provider | Groq absorbs high-burst tool calls; Gemini provides type-safe final JSON via `response_schema`. |
| **ReAct Loop** | **Capped at 5 Steps** | High/Unlimited Steps (12-20) | Bounds token cost & latency; forces high-signal exploration (`list` $\rightarrow$ `read` $\rightarrow$ `search`). |
| **Rerank Scaling** | **Shifted Sigmoid** | Raw Logits / Linear Min-Max | Re-centers raw NVIDIA logits ($[-20, +5]$) so relevant items map above $0.30$. |
| **Verification** | **Fail-Closed Retrieval, Fail-Open Judge** | Uniform Strategy | Refuses Q&A on low relevance (trust); ships report with caveat if judge fails (availability). |

---

## Slide 13: Presentation Summary & Demo Verification

### **Summary of System Achievements**
1. **Production-Grade Reliability**: Multi-LLM pool surviving rate limits, 400 DEGRADED errors, & container cold-scaling.
2. **Zero-Hallucination Grounding**: 5-layer verification stack guaranteeing backed claims and verified citations.
3. **JS-Hydrated Execution**: Automated escalation to Playwright Chromium for single-page applications.
4. **Multi-Channel Delivery**: FastAPI REST, live SSE event dashboard, Netlify static HTML publishing, PDF generation, and stdio MCP server.

### **Live Presentation Commands**
- **Run API Server**: `uv run uvicorn app.main:app --reload`
- **Test Suite**: `uv run python -m pytest tests/ -q` (All 85 offline tests pass)
- **Run MCP Server**: `uv run python -m app.mcp_server`
- **Render Web Deliverables**: `python -m web.render_report` & `python -m web.deploy`

---
