# CHANGELOG — what changed, broke, improved

Reverse order (newest first). For learning + interview recall.

## Phase 3 — ReAct report agent

**04d2738 — friendly outside-in framing + grounded suggestions**
- Report prompts reframed: prospective-user voice, actionable improvement notes.

**c5e495c — verify citations + survive live glitches**
- BROKE→FIXED: Llama emits malformed tool-call syntax → Groq 400 `tool_use_failed`. Added retry (stochastic glitch, re-ask). Only that 400 retried; other 400s propagate.
- Added citation verification.

**bc583c6 — Phase 3 hardening**
- Repeat-call guard: same (tool,args) twice → reminder, not re-execute. Saves steps + tokens.
- search_content near-miss margin (0.10): borderline score → "uncertain", not false "not covered".

**2424cb5 — read_page bare-slug fix**
- BROKE: agent passed "pricing"/"home" not exact URL → read_page failed → agent never read pages. FIXED: slug→URL recovery (unambiguous match only).

**03adf5f — Groq explore + Gemini synthesize**
- /ask default → Groq (high free RPM). /report: Groq drives ReAct loop, Gemini does final structured synthesis (response_schema). ReAct cap 5 steps.

**f5ba51f — Phase 3 built**
- ReAct agent: tools (list_pages/read_page/search_content), react.py loop, report.py, FirstImpressionReport schema (citations REQUIRED = rule #2 structural). Shared rag/pipeline.py.
- Rate-limit saga: Voyage 3/min (paced) → Gemini ~20/DAY (too small, retry Retry-After) → Groq 12K tok/min (bounded tool outputs: READ_PAGE_MAX_CHARS 4000, SEARCH_TOP_K 3).
- Verified Gemini + Groq function-calling & structured-output via smoke tests BEFORE building.

## Phase 2 — retrieval quality

**c203d5e — favor_precision extraction** — nav-debris chunk scored 0.490 (above bar); stricter trafilatura cleanup.

**ba01415 — guaranteed reranker seats**
- BROKE: RRF consensus (k=60) buried a chunk ranked #1 by ONE arm; vector-#1 pages missed. FIXED: each arm's top-3 always reach reranker. Fusion nominates, reranker judges.

**f9a3114 — threshold 0.4→0.45** — chosen from eval score gap (data, not vibes).

**929f458 — unanswerable evals + fail-closed** — false-answer-rate metric; non-rate-limit rerank errors → 503 (refuse, don't degrade).

**ad8c5e1 — Phase 2 built** — BM25 (rank-bm25) + vectors → RRF (hand-written) → Voyage rerank → min_relevance gate. Eval harness (hit@5, MRR).

## Phase 0–1 — foundation

**eb9f52d — pace Voyage** — free tier 3 RPM/10K TPM → batched with sleeps; RateLimitError → 429 not 500.

**7aa7a6d — Gemini provider** — Claude wanted $5; swapped to Gemini free in ~3 files (adapter pattern payoff).

**d1dc2ea / 610e6ec — docs** — build-vs-buy note (LangChain chunking alt); call-flow docstrings.

**6ea26e8 — Phase 0+1** — FastAPI skeleton, /health, config, Docker, uv. /ingest (crawl→chunk→embed→store), /ask (RAG, cited). Hand-written chunker + robots + fetcher; Voyage + Chroma.

## Recurring lesson
Every provider has a different free-tier wall. Pattern: cap request size for hard per-request limits; retry honoring Retry-After for per-minute; separate model = separate quota bucket. Operational constraints, not code bugs — most real-world pain lives here.
