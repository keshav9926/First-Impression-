# CHANGELOG — what changed, broke, improved

Reverse order (newest first). For learning + interview recall.

## Phase 3 — ReAct report agent

**(next) — thin-extraction guard (JS-rendered sites)**
- BROKE (found by testing a SECOND site — trynarrative.com, Framer): static crawl extracted 0.1% of the homepage (368 chars from 388K of HTML). Report then confidently claimed "no pricing / no getting-started / no integrations" — OUR blindness reported as the FOUNDER'S gap. Worst possible failure for this product.
- FIX: crawl computes text/HTML extraction ratio; thin if aggregate < 1% OR the SEED page < 1% (aggregate alone was fooled: big static legal pages diluted it to 3.4% while the homepage sat at 0.1%).
- Flag rides chunk metadata → list_pages warns the agent on its FIRST call ("do not report absence as a gap") → scope_note gets a hard caveat appended IN CODE (not trusted to the LLM). IngestResponse exposes extraction_warning.
- LESSON: single-site eval at 100% was blind to an entire failure class. One diverse site surfaced it instantly. Real cure = headless rendering (Playwright) — deferred, documented.

**1634c31 — heading-aware section map (coverage)**
- GAP (found by user pushing on the design): read_page truncates a long page at 4000 chars → agent blind past the cut, can't search for sections it never learned EXIST (unknown-unknown). Docs page = 44K chars.
- FIX: parse h1–h3 from raw HTML (`_HeadingCollector`, same HTMLParser trick as `_LinkCollector`). Headings ride as chunk metadata. read_page prepends "Sections on this page: …" (~150 tokens) ONLY when truncating (short pages pay nothing).
- Additive: trafilatura TEXT pipeline untouched → zero retrieval regression. Old stores keep working (`.get` default ""). Caps: 40 headings × 80 chars, deduped.
- Re-ingested vortexify: 15 pages (crawl found 5 new use-case pages), 44 chunks, docs = 40 headings (hit cap). Bonus: partially fixes the weak-citation open item (agent now knows doc section names).
- "Cheap" here = TOKENS not money — map is ~150 tok vs the 1000-tok prefix; everything stays free-tier.

**04d2738 — friendly outside-in framing + grounded suggestions**
- Prompts reframed: warm colleague voice; lead with the OUTSIDE-IN delta (what a stranger takes away vs what the site intends = the founder's blind spot = the actual value); kind about gaps.
- ADDED improvement_opportunities: 2–4 friendly "you might consider…" notes. Separate `ImprovementOpportunity` type + field so OPINION never contaminates cited FACT (rules #2/#3 preserved). Each cites the page it responds to; defaults `[]` (no invented filler).
- grounding now verifies suggestion source_urls too.

**c5e495c — verify citations + survive live glitches (hardening round 2)**
- ADDED (the diabolical fix): `grounding.enforce_citations` — the synthesis LLM GENERATES source_urls, nothing guaranteed they're real. Now drop any observation/suggestion citing a non-ingested page. Rule #2 made STRUCTURAL, not trusted. URL-normalized (trailing-slash / case tolerant).
- BROKE→FIXED (live): Llama emits malformed tool-call syntax → Groq 400 `tool_use_failed`. Retry the stochastic glitch (3×, re-ask); only that 400 retried, others propagate; persists → 502.
- BROKE→FIXED (live): agent bursts search_content → Voyage 3-RPM 429 killed the whole report. `embed_query` now retries into the next minute-window.
- search_content near-miss margin (0.10): borderline score → "uncertain", not a false "not covered" (min_relevance still tuned on ONE site — real fix = multi-site eval later).
- read_page truncation now LOGGED (docs 44K→4K was silent).

**bc583c6 — Phase 3 hardening (round 1)**
- Step cap unified → `settings.agent_max_steps` (was duplicated in prompts.py + groq_driver → drift risk).
- Repeat-call guard: same (tool,args) twice → reminder, not re-execute. Saves steps + tokens.
- Synthesis (Phase B) now uses `generate_with_retry` — a Gemini 429 on the LAST call no longer discards the whole (already-paid-for) exploration.
- `parsed=None` guard in both drivers → clear ValueError (was AttributeError deep in the stack).
- /report maps Gemini 429 → HTTP 429 (was opaque 500); ValueError → 502.
- Groq args "null" edge: `json.loads("null")` → None → crash. Guarded with `or {}`.

**2424cb5 — read_page bare-slug fix**
- BROKE: agent passed "pricing"/"home" not exact URL → read_page failed silently → agent never actually read pages (report propped up by search + error-msg URL echoes). FIXED: slug→URL recovery (unambiguous match only) + prompt demands exact URLs.

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
