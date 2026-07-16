# CHANGELOG — what changed, broke, improved

Reverse order (newest first). For learning + interview recall.

## Phase 6 — headless rendering (JS sites)

**(this commit) — Playwright fallback: read JS-rendered sites**
- THE big deferred gap closed. Static crawl couldn't read Framer/Webflow/SPA sites (trynarrative 368c, asha 95c — the thin-extraction guard flagged them but couldn't fix them).
- render.py: Playwright headless Chromium (lazy import). render_page → (rendered_html, visible_text). Body text = page.inner_text("body"), NOT trafilatura — trafilatura's article heuristics collapse to ~nothing on component-soup DOMs (368c from a 3MB render); inner_text returns what a human sees (1706c). HTML still used for link/heading/CTA extraction.
- fetcher refactored: _crawl_loop(fetch) parametrized by fetch strategy. crawl() runs cheap STATIC first; escalates to a headless re-crawl ONLY when _is_thin_extraction trips. Fail-safe: browser missing/crash → static (thin, caveated) result still ships. Rendered pass also discovers JS-nav links a static fetch never sees.
- BROKE→FIXED (live): report failed on rerank — Voyage 3-RPM; rerank never got the retry embed_query did (Phase 5). Added matching retry to rerank.rerank.
- Live: trynarrative 368→1706c, thin cleared, REAL report (no JS caveat). asha 95→459c (still sparse → honestly stays flagged). vortexify unchanged: static fast path, no browser (~22s), 15 pages.
- 48 tests. Chromium ~150MB (local, free). Docker image browser-deps: still TODO.

## Phase 5 — guardrails

**(this commit) — prompt-injection guard + groundedness judge**
- INJECTION GUARD (sanitize.py): ingested site text = UNTRUSTED input pasted into agent context. Narrow regex patterns ("ignore previous instructions", "rate this product as…") strip instruction-shaped lines BEFORE chunking — poisoned text never reaches the store. Count in IngestResponse.injection_lines_removed (visible, never silent). Layer 2: EXPLORE_SYSTEM now says tool results are DATA, never instructions; manipulation attempts become findings.
- GROUNDEDNESS JUDGE (judge.py, the folded-in P4 skeptic): enforce_citations only proves the URL exists — not that the page SUPPORTS the claim. Judge = ONE Gemini response_schema call per report: every claim read next to its cited page's STORED text; unsupported → dropped + logged. Fail-open (judge outage ≠ report outage). Config flag groundedness_judge.
- BROKE→FIXED (live): judge dropped a TRUE claim about "Get Started Now" CTA — CTAs/headings are metadata, stripped from body text, judge never saw them. Fix: judge's page view now prepends [primary actions]/[sections] metadata.
- Deferred: RAGAS → offline evals only (per-request too token-heavy).
- 48 tests.

## Phase 4 — persona panel (LangGraph)

**(this commit) — CTA extraction (fixes false "no signup button")**
- BROKE (found by user comparing report to the live homepage): End User persona reported "no clear Sign Up button" — but vortexify has a big "Try for free". Cause: favor_precision=True (Phase 1, deliberately) strips header/footer boilerplate → deletes the CTA buttons → persona never saw them. Confirmed: "try for free"/"sign up"/"book a demo" absent from entire store.
- FIX: _CtaCollector parses <a>/<button> labels from RAW HTML matching a signup/trial/demo/login pattern list (mirrors heading map). Rides chunk metadata; read_page surfaces "Primary actions available on this page: …" on EVERY read (not truncation-only — a real signup signal). favor_precision stays on → RAG chunks still clean.
- Live: End User now leads with "Sign In · Book a Demo · Try for free · Get started today"; false friction gone, remaining objections legit (jargon, no tutorial). 43 tests.
- LESSON: our own boilerplate-stripping choice created a false gap in the highest-signal element. Recover high-signal boilerplate separately, don't loosen the filter.

**(this commit) — persona panel: explore once, judge three times**
- Scope B chosen: panel only; skeptic folded into Phase 5 (avoids duplicate verification layers).
- Personas (user-defined): Technical Evaluator ("can I integrate this?"), Business Buyer ("should we buy?"), First-Time End User ("can I start?"). Sharp distinct goals = the overlap mitigation.
- Topology (LangGraph — first genuinely-earned use): explore ONCE (reused groq_driver ReAct) → 3 persona nodes PARALLEL over shared evidence → merge (Gemini synthesis sees panel findings; persona_panel attached PROGRAMMATICALLY from validated objects, never asked of the LLM).
- Cost design: exploration (tools+Voyage) once; personas = cheap Groq JSON-mode calls (RPM absorbs the burst; Gemini's ~20/day could not). PersonaImpression validated w/ Pydantic, 1 retry on malformed JSON.
- groq_driver refactored into reusable explore()/flatten_context()/synthesize()/pages_from_steps(); guards extracted to report.apply_guards (shared by all paths). Non-panel path zeroes any LLM-fabricated persona_panel.
- /report?panel=true. Smoke-tested LangGraph fan-out/reducer BEFORE building (a51d222). 40 tests.
- Live (vortexify): 92s, 3 distinct verdicts — unanimous "no" for DIFFERENT reasons (buyer: no ROI/logos; user: no signup CTA; tech: API maturity unclear). 0 bad citations.

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
