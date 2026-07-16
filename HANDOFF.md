# HANDOFF — First Impression project

Paste this into a new Claude Code session to continue.

## What this is
AI system: analyze a startup's PUBLIC site, output a cited "first impression" report to email founders. Spec in PROJECT.md. Repo: github.com/keshav9926/First-Impression- (branch main).

## Hard rules (enforced in code)
1. Public data only — robots.txt respected, no logins, rate-limited.
2. Grounded output only — every claim cites a source chunk.
3. Observational tone — describe, never grade/attack.

## Working style (IMPORTANT — user wants this)
- One phase at a time. Stop after each phase. Explain each file. Give a study list.
- Build simple logic by hand; buy hard machinery. Explain every line (interview prep).
- Verify new external APIs with a smoke test BEFORE building on them.
- Commit + push after each phase. Never commit .env.
- Be terse. No filler. Run tools, show result, stop.

## Stack (all free tier)
- FastAPI + uvicorn, uv, Docker. Python 3.13.
- RAG: Voyage embeddings (voyage-3.5) + Chroma (local). Hybrid: BM25 (rank-bm25) + vectors → RRF fusion (hand-written, k=60, guaranteed_per_list=3) → Voyage rerank (rerank-2.5-lite) → min_relevance 0.45 gate.
- LLM: provider-switched. /ask → Groq (default), Gemini, or Anthropic. /report agent → Groq or Gemini.
- Keys in .env: VOYAGE_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, LLM_PROVIDER, AGENT_PROVIDER=groq.

## Done: Phases 0–3
- P0: FastAPI skeleton, /health, config, Docker.
- P1: /ingest (crawl→chunk→embed→store), /ask (RAG Q&A, cited).
- P2: hybrid retrieval + rerank + relevance gate. Evals in evals/ (hit@5, MRR, false-answer rate). hit@5 100%, false-answer 0% on vortexify.
- P3: /report — hand-rolled ReAct agent. Tools: list_pages, read_page, search_content (agent/tools.py). Loop: agent/react.py (Gemini), agent/groq_driver.py (Groq). Prompts: agent/prompts.py. Output: FirstImpressionReport schema (schemas.py) — citations required = rule #2 structural.

## Layout
- app/main.py — endpoints (/health /ingest /ask /report)
- app/config.py — settings
- app/schemas.py — Pydantic models
- app/ingestion/ — robots, fetcher, chunker
- app/rag/ — embeddings, store, keyword(BM25), fusion(RRF), rerank, pipeline(shared funnel), qa
- app/agent/ — tools, react(Gemini loop), groq_driver, report(dispatch), prompts, llm(Gemini retry)
- tests/ — 19 passing. evals/ — retrieval eval + debug tool.

## Rate-limit lessons (recurring theme)
- Voyage free: 3 req/min → paced batches in embeddings.py.
- Gemini free: ~20 req/DAY on 2.5-flash → too small for agent; retry honors Retry-After (agent/llm.py).
- Groq free: 12K tokens/MINUTE → bounded tool outputs (tools.py: READ_PAGE_MAX_CHARS 4000, SEARCH_TOP_K 3) + MAX_STEPS 5.

## Phase 3 CLOSED (2026-07-15) — hardening done before Phase 4
Live /report on Groq verified: ~26s, no 413, real page reads, 0 bad citations.

### Accuracy/robustness pass (round 2)
- CITATION VERIFICATION (rule #2 structural): agent/grounding.py.enforce_citations
  drops any Observation whose GENERATED source_url isn't a real ingested page.
  Runs in report.generate_report() → covers both drivers. url-normalized
  (trailing slash / case tolerant). Was the top risk: synthesis LLM invents urls.
- search_content near-miss: a best match just under min_relevance now returns
  "uncertain", not a hard "not covered" → stops FALSE unanswered_questions.
  (min_relevance still tuned on ONE site — real fix is multi-site eval later.)
- read_page truncation now LOGGED (docs page is 44K chars → 4K). Silent before.
- Groq tool_use_failed 400 (Llama emits malformed tool-call syntax): _complete
  retries it up to 3x (stochastic glitch); persists → 502. Caught live.
- embed_query retries Voyage 3-RPM 429 into next minute-window (agent fires a
  burst of search_content). Caught live. Was killing the whole report.
- 28 tests passing.

### Coverage pass: heading-aware section map (closes the truncation blind spot)
- fetcher.py: _HeadingCollector parses h1-h3 from raw HTML (additive — the
  trafilatura text pipeline untouched, zero retrieval regression). Page gains
  .headings; capped 40 headings × 80 chars.
- Chunk metadata now carries "headings" (joined string; Chroma needs scalar).
  store.replace_all/all_chunks pass it through (.get default "" = old stores work).
- read_page: when TRUNCATING, prepends "Sections on this page: …" (~150 tokens)
  → the agent now SEES what exists past the 4000-char cut and can
  search_content into any section (kills the unknown-unknown).
- Re-ingested vortexify: 15 pages, 44 chunks, docs page = 40 headings.
- Also fixes the weak-citation open item partially (agent knows doc sections).
- 32 tests. Live verified: map shown, report clean, 0 bad citations.

### Product pass: friendly outside-in framing + improvement suggestions
- Reframed prompts (EXPLORE_SYSTEM + SYNTHESIZE_INSTRUCTION): warm colleague
  tone, OUTSIDE-IN delta (what a stranger takes away vs what the site intends —
  the founder's blind spot = the value), kind about gaps.
- NEW report field improvement_opportunities: list[ImprovementOpportunity]
  (schemas.py). Each = observed (real experience) + suggestion (gentle "you
  might consider…") + source_url. Kept a SEPARATE type/field from Observation so
  opinion never contaminates cited fact (rules #2/#3 preserved). Defaults to []
  so the model can honestly return none.
- grounding.enforce_citations now verifies suggestion source_urls too — advice
  pinned to a hallucinated page is dropped like a fabricated observation.
- Live: warm tone confirmed, 3 grounded suggestions, 0 bad citations. 29 tests.

### Round 1 fixes:
- read_page bare-slug bug (agent passed "home"/"pricing" as urls — every read
  silently failed; prompt now demands exact urls + tool recovers unambiguous slugs).
- Step cap unified → settings.agent_max_steps (was duplicated in prompts.py + groq_driver).
- Repeat-call guard (tools.repeat_call_reminder) in BOTH drivers — identical
  (tool, args) → reminder, no re-execution, no wasted tokens.
- Phase B synthesis (groq_driver) now uses generate_with_retry — a Gemini 429
  at the last step no longer throws away the whole exploration.
- parsed=None guard in both drivers → ValueError → 502 with clear message.
- /report maps Gemini 429 → HTTP 429 (was 500).
- Groq arguments "null" edge (json.loads → None) → {} guard.
Tests: 24 passing.

## IMMEDIATE NEXT STEP
Phase 4 planning: multi-agent crew (researcher / user-sim / evaluator / skeptic) via LangGraph.

## Thin-extraction guard (2026-07-15, after testing trynarrative.com)
- JS-rendered (Framer) site: static crawl got 0.1% of homepage → report invented
  false gaps ("no pricing" etc). Added extraction-ratio detection (aggregate OR
  seed page < 1%) → chunk metadata flag → list_pages warns agent + scope_note
  caveat appended in code. 34 tests. Store currently holds trynarrative (18
  chunks, flagged thin) — re-ingest vortexify before demoing rich reports.
- REAL CURE (deferred): headless rendering via Playwright for JS sites.
- LESSON: one-site eval at 100% masked an entire failure class.

## Phase 4 DONE (2026-07-16) — persona panel via LangGraph
- Scope B: panel only (skeptic → Phase 5). agent/personas.py (3 user-defined
  personas), agent/panel.py (StateGraph: explore once → 3 parallel persona
  nodes → merge). groq_driver refactored: explore/flatten_context/synthesize/
  pages_from_steps reusable; guards → report.apply_guards (all paths).
- /report?panel=true. Personas = Groq JSON mode + Pydantic validate (1 retry).
  persona_panel attached programmatically. 40 tests. Live: 3 distinct verdicts,
  0 bad citations, 92s.

## Phase 5 DONE (2026-07-16) — guardrails
- Injection guard: ingestion/sanitize.py strips instruction-shaped lines
  pre-chunking (narrow regexes); IngestResponse.injection_lines_removed;
  EXPLORE_SYSTEM marks tool output as DATA-not-instructions.
- Groundedness judge: agent/judge.py — 1 Gemini call/report fact-checks every
  claim vs its cited page's STORED text (+ CTA/heading metadata, live-caught
  false drop); unsupported dropped, fail-open, flag groundedness_judge.
- RAGAS deferred to offline evals. 48 tests.
- NOTE: store currently holds asha.health (2 chunks, thin) — re-ingest
  vortexify before demos.

## Playwright headless rendering DONE (2026-07-16) — JS gap closed
- ingestion/render.py (Playwright chromium, lazy import): render_page → (html,
  inner_text). fetcher._crawl_loop parametrized; crawl() static-first,
  escalates to headless re-crawl only when thin. Fail-safe to static.
- rerank.rerank got the Voyage-429 retry (was missing; embed_query had it).
- Live: trynarrative now readable (1706c, no JS caveat, real report). vortexify
  still fast static path. 48 tests. Chromium ~150MB local.
- TODO: Docker image needs chromium system-deps (playwright install-deps) before
  containerized deploy (P8).
- STORE STATE: currently holds trynarrative (18 chunks). Re-ingest vortexify
  before demos (Voyage 3-RPM was exhausted from testing; just wait + re-ingest).

## Then: Phase 6+ (renumber — headless was the old "P6" slot)
- Playwright headless rendering for JS sites (agreed, deferred — in memory).
- P6: streaming dashboard. P7: MCP server. P8: Langfuse, Docker deploy.

## Known open items (deferred, documented)
- Chunk metadata: only url, no section headings (docs is one URL → weak citations). Add heading-aware chunking on big docs sites.
- BM25 no stemming ("shift" ≠ "shifts").
- /report blocks (sync) 2-4 min — async job later.
- README must mention LangChain as chunking alternative (in memory).

## Commands
- Run: uv run uvicorn app.main:app --reload
- Test: uv run pytest -q
- Lint: uv run ruff check app/
- Eval: uv run python evals/run_retrieval_eval.py
