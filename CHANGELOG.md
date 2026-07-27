# First Impression Engine (FIE) — Technical Changelog & Architectural Decision Log

> **For Mentor Review & Technical Interviews**: Comprehensive record of architectural evolution, critical failures, engineering comebacks, trade-offs, mathematical formulations, and phase-by-phase system design.

---

## 1. Executive Summary: Current System Architecture

The **First Impression Engine (FIE)** is an autonomous, production-grade product analysis platform that evaluates a startup's public web surface from an unauthenticated visitor perspective. 

### Current High-Reliability Technical Stack
- **Multi-LLM Failover Pool (`app/agent/llm_pool.py`)**: Standardized on **NVIDIA NIM API** (`integrate.api.nvidia.com/v1`).
  - **Primary Model Chain**: `z-ai/glm-5.2` (GLM-5.2 — accuracy anchor) → `deepseek-ai/deepseek-v4-pro` (DeepSeek-V4-Pro — preferred synthesis/persona engine) → `deepseek-ai/deepseek-v4-flash` → `nvidia/nemotron-3-ultra-550b-a55b` (Nemotron-3-Ultra — 186 t/s fast safety net).
  - **Pipeline Modes**: `normal` (Fast: DeepSeek-V4-Pro → Flash → Nemotron) & `deep` (Thorough: GLM-5.2 → DeepSeek-V4-Pro → Flash → Nemotron).
  - **Exploration Cap**: **40 ReAct Steps** (`agent_max_steps = 40`) allowing complete multi-page exploration without step exhaustion.
- **VLM Vision Captioning Pipeline (`app/ingestion/vision.py`)**:
  - Primary VLM: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (2.3s/img speed, reads exact shipment IDs, status tables, UI charts).
  - VLM Failover Chain: `omni-30b` → `meta/llama-3.2-90b-vision-instruct` → `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`.
  - Cap: `vision_max_images_per_page = 40`, `vision_max_images_total = 200`. Injects screenshot descriptions directly into chunk metadata (`images`), eliminating text extractor blindness.
- **Hybrid Ingestion & Rendering (`app/ingestion/render.py`, `fetcher.py`)**:
  - Forced Headless Rendering (`force_render = True`): Always executes Playwright Chromium with auto-scrolling to hydrate JS single-page applications (SPAs) and trigger lazy-loaded product dashboards.
  - HTML AST Collectors (`_CtaCollector` & `_HeadingCollector`): Extracts navbar CTAs and heading hierarchies *before* boilerplate removal.
- **Hybrid Retrieval Funnel (`app/rag/`)**:
  - Dense Embeddings: `nvidia/nemotron-3-embed-1b` (2048-dim, `embed_provider = "nvidia"`), eliminating rate-limit throttles.
  - Sparse Keyword Search: `BM25Okapi` (`rank_bm25`) with custom light stemming (`-ing`, `-es`, `-ed`, `-s`).
  - Fusion: Reciprocal Rank Fusion ($K=60$) with **Guaranteed Seats (`guaranteed_per_list = 3`)** ensuring top 3 items from both Dense and Sparse arms reach the cross-encoder.
  - Reranker: Voyage `rerank-2.5-lite` or NVIDIA `rerank-qa-mistral-4b` with **Shifted Sigmoid Normalization** ($S_{\text{norm}} = \frac{1}{1 + e^{-(L - (-10.0))/3.0}}$).
  - Gate: Calibrated **0.30 Min Relevance Floor**.
- **Multi-Persona Panel (`app/agent/panel.py`)**:
  - LangGraph State Graph with parallel fan-out to 3 personas (*Technical Evaluator*, *Business Buyer*, *First-Time End User*), using `operator.add` state reducer.
- **5-Layer Verification & Trust Stack (`app/agent/judge.py`, `grounding.py`)**:
  - Schema Constraints → Citation URL Matching → Adversarial Fact-Check Judge (temp 0.0) → Contradiction Checker → Scope Caveats.
  - Empty-Evidence Refusal: Stores < 200 chars raise `InsufficientEvidenceError` (HTTP 409).
- **Observability & Delivery Interfaces**:
  - Langfuse OpenTelemetry tracing (`app/observability.py`), FastMCP stdio server (`app/mcp_server.py`), live SSE event bus (`app/events.py`), Netlify static publishing (`web/deploy.py`), and Playwright PDF generation (`web/to_pdf.py`).

---

## 2. Top 5 Critical Failures & Engineering Comebacks

### Failure 1: The JS SPA "Black Hole" (0.1% Text Extraction on Framer/Webflow SPAs)
- **What Broke**: On `trynarrative.com` (a Framer SPA), static HTTP crawling extracted only 368 chars from 388KB of HTML. The LLM report hallucinated: *"The startup has no pricing, no getting started guide, and no product features."*
- **Root Cause**: Client-Side Rendering (CSR) populates DOM nodes via JavaScript after `DOMContentLoaded`. Static scrapers see empty root `<div>` tags.
- **Engineering Comeback**: 
  1. Built an **Extraction Ratio Guard** (`_is_thin_extraction`): Detects text-to-HTML ratio < 1% or seed text < 1,200 chars.
  2. Built **Playwright Chromium Headless Escalation** (`render.py`) & set `force_render = True`: Automates Chromium execution with top-to-bottom auto-scrolling to trigger lazy hydration. Extracted via `inner_text("body")`.
  3. Result: Text extraction on `trynarrative.com` jumped **368 chars → 1,706 chars**, restoring full report accuracy.

### Failure 2: Erased CTA Buttons (Over-Aggressive Boilerplate Removal)
- **What Broke**: The *First-Time End User* persona reported: *"The homepage lacks a clear Sign Up or Try Free button."* But inspecting the live homepage revealed a prominent `"Try for free"` CTA in the navbar.
- **Root Cause**: `trafilatura` stripped header/footer navigation debris, inadvertently deleting `<button>` and `<a>` elements containing signup CTAs.
- **Engineering Comeback**:
  1. Built HTML AST Collectors (`_CtaCollector` & `_HeadingCollector`) in `fetcher.py` that parse raw HTML *before* boilerplate stripping.
  2. CTAs matching signup/login patterns (`"try for free"`, `"sign up"`, `"get started"`) ride as structured chunk metadata.
  3. Every `read_page` execution surfaces `"Primary actions available on this page: [Try for free, Book Demo]"` at the top of the observation.
  4. Result: Persona false negatives eliminated while keeping RAG chunks 100% clean of boilerplate.

### Failure 3: The 429 & Availability Cascade (NIM DEGRADED & 404 Cold-Scaling)
- **What Broke**: Long agent runs experienced sudden crashes when an LLM provider returned HTTP 429, HTTP 400 `"DEGRADED function cannot be invoked"`, or transient HTTP 404s during container cold-scaling.
- **Root Cause**: Unhandled single-provider failure modes crashed multi-minute agent loops.
- **Engineering Comeback**:
  1. Built **Multi-LLM Resilience Pool (`app/agent/llm_pool.py`)**: Automatic fallback across NVIDIA NIM endpoints (`GLM-5.2` → `DeepSeek-V4-Pro` → `DeepSeek-V4-Flash` → `Nemotron-3-Ultra`).
  2. **Circuit Breakers (`_trip`, `_live`)**: Differentiates per-minute 429s (sleep Retry-After) from per-day quota caps or 400 DEGRADED errors (15-min circuit breaker cooldown: `_DAILY_COOLDOWN = 900s`).
  3. **404 Cold-Scale Retries**: Retries 404 errors up to 2 times before tripping, handling NIM container spin-ups.
  4. **Tolerant JSON Parser (`_extract_report_json`)**: Strips `<think>` tags emitted by reasoning models and performs balanced-brace JSON parsing.

### Failure 4: The Hallucinated Startup Report (Empty-Evidence Ingestion)
- **What Broke**: When a website blocked crawling via `robots.txt` or returned a 404 error page, the agent received an empty vector store, but the synthesis LLM hallucinated a plausible-sounding report out of thin air.
- **Root Cause**: LLMs default to completion when given open prompts, even over empty context.
- **Engineering Comeback**:
  1. Enforced **Empty-Evidence Refusal**: Added a strict threshold (`_MIN_EVIDENCE_CHARS = 200`). If stored text is below 200 chars, `generate_report()` raises `InsufficientEvidenceError`.
  2. FastAPI maps this to an **HTTP 409 Conflict** ("Nothing ingested yet or crawl disallowed").
  3. Result: The system refuses to generate a report rather than outputting ungrounded claims.

### Failure 5: RRF Consensus Burying #1 Vector Matches
- **What Broke**: During retrieval evaluation, a document ranked **#1 by Dense Vector Search** was omitted from the top-5 candidates sent to the LLM.
- **Root Cause**: Standard Reciprocal Rank Fusion (RRF) score $\sum \frac{1}{60 + r_m(d)}$ penalized vector #1 matches if BM25 ranked them poorly due to acronym/stemming mismatches.
- **Engineering Comeback**:
  1. Implemented **Guaranteed Seats (`guaranteed_per_list = 3`)** in `fusion.py`.
  2. Top 3 items from *each* individual search arm (Dense Vector, BM25) are guaranteed a slot in the candidate pool sent to the cross-encoder reranker.

---

## 3. Key Technical Trade-Offs & Decision Matrix

| Domain | Option Selected | Alternative Considered | Technical Rationale & Impact |
| :--- | :--- | :--- | :--- |
| **Model Execution** | **NVIDIA NIM Failover Pool** (`app/agent/llm_pool.py`) | Single LLM Provider | **Rationale**: High availability across GLM-5.2, DeepSeek-V4-Pro, and Nemotron. Automatic circuit breakers handle 429s, 400 DEGRADED, and 5xx errors.<br>**Trade-off**: Requires tolerant parsing to handle model output format variations (`<think>` tags). |
| **Vision Pipeline** | **Multi-VLM Failover Chain** (`app/ingestion/vision.py`) | Text-Only Ingestion | **Rationale**: `nemotron-3-nano-omni-30b` reads product screenshots, status tables, and charts (2.3s/img). Captions injected into chunk metadata (`images`).<br>**Trade-off**: Increases ingestion duration (~2s per screenshot). |
| **Rendering Strategy** | **Forced Playwright Rendering** (`force_render = True`) | Static HTTP Crawling | **Rationale**: Guarantees full JS SPA hydration (Framer, Webflow, React) and lazy-loaded dashboard image discovery.<br>**Trade-off**: Higher memory usage and execution time per page crawl. |
| **Retrieval Embeddings** | **NVIDIA Nemotron-3 Embed** (`embed_provider = "nvidia"`) | Voyage AI Embeddings | **Rationale**: 2048-dim dense vectors with zero request-per-minute rate-limit throttling (eliminating Voyage 3 req/min wait wall).<br>**Trade-off**: Shares account quota with LLM pool calls. |
| **Reranker Scaling** | **Shifted Sigmoid Normalization** | Raw Logits / Linear Scale | **Rationale**: NVIDIA reranker outputs strongly negative logits ($[-20, +5]$). Shifted Sigmoid $\frac{1}{1 + e^{-(L+10)/3}}$ re-centers scores above calibrated $0.30$ floor.<br>**Trade-off**: Required empirical tuning of center ($-10.0$) and scale ($3.0$) parameters. |
| **Verification Strategy** | **Fail-Closed Retrieval, Fail-Open Fact Judge** | Uniform Strategy | **Rationale**: Refuses Q&A on low relevance score (trust). Ships report with explicit scope caveat banner if verification judge encounters API outage.<br>**Trade-off**: Report remains accessible during judge outages, with warning attached. |

---

## 4. Phase-by-Phase Timeline & Architecture Evolution

- **Phase 9 — Model Pool Hardening & Multi-VLM Vision**: Integrated NVIDIA NIM failover chain (`GLM-5.2` → `DeepSeek-V4-Pro` → `Nemotron-3-Ultra`), circuit breakers, tolerant `<think>` JSON parsing, and 3-model VLM vision pipeline (`omni-30b` → `llama-3.2-90b` → `nemotron-nano-vl`).
- **Phase 8.1 & 8 — Langfuse Observability & Containerization**: OpenTelemetry instrumentation (`app/observability.py`), monkey-patched OpenAI client, hierarchical span trees, Python 3.13 slim Dockerfile with pre-installed Playwright Chromium and persistent Chroma volume.
- **Phase 7 — Model Context Protocol (MCP) Server**: Built FastMCP stdio server (`app/mcp_server.py`) exposing `analyze_first_impression`, `ask_ingested`, and `ingestion_status` with zero code drift.
- **Phase 6 — Live Streaming SSE & Playwright JS Escalation**: Created `contextvars` event bus (`app/events.py`) for live `/analyze/stream` updates and Playwright headless Chromium escalation (`render.py`).
- **Phase 5 — 5-Layer Trust Pipeline & Injection Sanitizer**: Schema constraints, citation URL verification, adversarial fact-check judge (temp 0.0), contradiction check, scope caveats, and regex prompt-injection line stripping (`sanitize.py`).
- **Phase 4 — LangGraph Multi-Persona Panel**: Parallel fan-out to *Technical Evaluator*, *Business Buyer*, and *First-Time End User* nodes, `operator.add` state reducer, and HTML AST CTA collector.
- **Phase 3 — ReAct Analysis Agent**: Autonomous tool loop (`list`, `read`, `search`), Section Maps prepended to truncated pages, and `_trim_history` context bounding.
- **Phase 2 — Hybrid Retrieval Funnel**: BM25 + Nemotron Dense Vectors, Reciprocal Rank Fusion ($K=60$) with Guaranteed Seats (`guaranteed_per_list = 3`), Cross-Encoder Reranker with Shifted Sigmoid, and 0.30 Min Relevance Floor.
- **Phase 0 & 1 — Pipeline Foundation**: FastAPI application skeleton, ChromaDB vector store, custom paragraph chunker, and `robots.txt` politeness gate.

---

## 5. Mentor Presentation Cheat Sheet (Quick Answers)

- **Q: What LLM infrastructure powers the system?**
  - *Answer*: An NVIDIA NIM failover chain (`GLM-5.2` → `DeepSeek-V4-Pro` → `DeepSeek-V4-Flash` → `Nemotron-3-Ultra`). Circuit breakers handle rate limits, 400 DEGRADED deployment errors, and container cold-scaling.
- **Q: How does FIE analyze JS-rendered single page applications?**
  - *Answer*: `force_render = True` executes Playwright Chromium with top-to-bottom auto-scrolling to hydrate JS SPAs and trigger lazy-loaded dashboard images. Text is extracted via `inner_text("body")`.
- **Q: How are product screenshots analyzed?**
  - *Answer*: `app/ingestion/vision.py` passes screenshots to a 3-VLM chain led by `nvidia/nemotron-3-nano-omni-30b` (2.3s/img). Generated captions describe dashboard UI, charts, and metrics, and ride inside chunk metadata (`images`).
- **Q: How are hallucinations prevented?**
  - *Answer*: 5-layer trust pipeline: Schema constraints → Citation URL matching against Chroma → Adversarial temp-0.0 LLM fact check → Contradiction check → Empty-evidence refusal (HTTP 409 if <200 chars).
