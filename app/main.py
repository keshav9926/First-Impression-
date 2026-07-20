# app/main.py — the FastAPI application entry point.
# Defines the app object that Uvicorn serves and registers all endpoints.
# Endpoints stay thin: validate → call the right module → shape the response.
# The real logic lives in app/ingestion/ and app/rag/.
#
# CALL FLOW (the whole system, from this file's point of view):
#
#   POST /ingest → ingest()
#       ├── _require_keys(voyage=True)         key present? else 503
#       ├── robots.is_allowed(seed_url)        rule #1 gate → 403 if refused
#       ├── fetcher.crawl(url, max_pages)      download + extract pages
#       ├── chunker.chunk_text(page.text)      pages → chunks (per page)
#       ├── embeddings.embed_documents(texts)  chunks → vectors (Voyage API)
#       └── store.replace_all(chunks, vectors) save into Chroma
#
#   POST /ask → ask()   [Phase 2: hybrid retrieval funnel]
#       ├── _require_keys(voyage=True, llm=True)
#       ├── store.count()                      empty? → 409 "ingest first"
#       ├── pipeline.retrieve(question)        embed → vector+BM25 → RRF → rerank
#       ├── min_relevance gate                 all below bar? → refuse, skip LLM
#       └── qa.answer(question, relevant)      LLM answers from chunks only
#
#   POST /report → report()   [Phase 3: the ReAct analysis agent]
#       ├── key guards (Voyage + the configured agent_provider's key)
#       ├── store.count()                      empty? → 409 "ingest first"
#       └── generate_report()                  agent explores (list_pages /
#                                              read_page / search_content) then
#                                              synthesizes the cited report
#                                              (provider: gemini or groq)

import json
import logging
import queue
import threading
from pathlib import Path

import openai
import voyageai.error
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app import events
from app.agent.report import InsufficientEvidenceError, generate_report
from app.config import settings
from app.ingestion.chunker import chunk_text
from app.ingestion.fetcher import crawl
from app.ingestion.robots import is_allowed
from app.ingestion import vision
from app.ingestion.sanitize import sanitize_text
from app.rag import embeddings, pipeline, qa, store
from app.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    ReportResponse,
    Source,
)

logger = logging.getLogger("first_impression")

app = FastAPI(
    title=settings.app_name,
    description="Analyzes a startup's public product experience and reports on the new-user journey.",
)


def _require_keys(*, voyage: bool = False, llm: bool = False) -> None:
    """Fail with a clear 503 if a needed API key isn't configured.

    Called by: ingest() and ask(), as their first line.
    Why: keys default to "" in config.py so /health and tests run without
    them — so endpoints that DO need a key must check explicitly, and a
    descriptive 503 beats a confusing auth error from deep inside an SDK.
    The llm check looks at whichever provider is active (config.llm_provider):
      "groq"      → GROQ_API_KEY   (default for /ask)
      "gemini"    → GEMINI_API_KEY
      "anthropic" → ANTHROPIC_API_KEY
    """
    if voyage and not settings.voyage_api_key:
        raise HTTPException(status_code=503, detail="VOYAGE_API_KEY is not set in .env")
    if llm:
        if settings.llm_provider == "groq" and not settings.groq_api_key:
            raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set in .env")
        if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
            raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set in .env")
        if settings.llm_provider == "gemini" and not settings.gemini_api_key:
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set in .env")


@app.get("/health")
def health() -> dict:
    """Liveness check: proves the server is up and config loaded correctly.

    Called by: humans, tests (tests/test_health.py), and — in production —
    load balancers / Docker healthchecks deciding whether we get traffic.
    Calls: nothing — deliberately cheap and dependency-free.
    """
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


def _ingest_site(url: str, max_pages: int) -> IngestResponse:
    """Crawl → sanitize → chunk → embed → store. The shared ingestion core,
    used by BOTH /ingest and the streaming dashboard, so the two can't drift.
    Emits progress events (no-op unless a collector is active). Raises plain
    exceptions; callers map them to HTTP or SSE error events.
    """
    if not is_allowed(url):
        raise PermissionError("robots.txt disallows fetching this URL (public-data rule).")

    events.emit("phase", name="crawl")
    result = crawl(url, max_pages=max_pages)
    if not result.pages:
        raise ValueError("No readable pages found at this URL.")

    # Prompt-injection guard (Phase 5): strip instruction-shaped lines from
    # untrusted page text BEFORE chunking — poisoned lines never reach the store.
    injection_lines_removed = 0
    sanitized_pages = []
    for page in result.pages:
        clean, removed = sanitize_text(page.text)
        injection_lines_removed += len(removed)
        sanitized_pages.append((page, clean))

    # Vision (Phase 9): caption product screenshots the text extractor is blind
    # to, so the report can reason about WHAT a visual shows — not just that one
    # exists. Fail-open + capped; {} when disabled/unconfigured.
    events.emit("phase", name="vision")
    captions_by_page = vision.caption_pages(result.pages)
    images_seen = sum(len(getattr(p, "image_urls", []) or []) for p in result.pages)
    images_captioned = sum(len(v) for v in captions_by_page.values())

    # Each chunk carries: url (citations), headings (read_page section map),
    # ctas (signup/demo actions), images (visual evidence: alt/filename labels +
    # any vision captions), extraction_warning (JS-thin caveat). Chroma metadata
    # must be scalar, so lists are joined.
    def _img_meta(page) -> str:
        parts = list(page.images)
        caps = captions_by_page.get(page.url, [])
        if caps:
            parts.append("VISION READS — " + " ; ".join(caps))
        return " · ".join(parts)

    chunks = [
        {
            "text": piece,
            "url": page.url,
            "headings": " · ".join(page.headings),
            "ctas": " · ".join(page.ctas),
            "images": _img_meta(page),
            "extraction_warning": result.thin_extraction,
        }
        for page, clean in sanitized_pages
        for piece in chunk_text(clean)
    ]

    events.emit("phase", name="embed", chunks=len(chunks))
    vectors = embeddings.embed_documents([c["text"] for c in chunks])
    stored = store.replace_all(chunks, vectors)

    summary = IngestResponse(
        pages_fetched=len(result.pages),
        chunks_stored=stored,
        skipped_by_robots=result.skipped_by_robots,
        extraction_warning=result.thin_extraction,
        injection_lines_removed=injection_lines_removed,
        images_seen=images_seen,
        images_captioned=images_captioned,
    )
    events.emit("ingest.done", **summary.model_dump())
    return summary


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Crawl a public site (robots.txt-compliant), chunk, embed, and store it.

    Called by: the client (you, via /docs or curl). Delegates to _ingest_site
    and maps its exceptions to HTTP status codes.
    """
    _require_keys(voyage=True)
    try:
        return _ingest_site(str(request.url), request.max_pages)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))  # rule #1
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except voyageai.error.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Voyage free-tier rate limit hit — wait a minute and retry, "
            "or ingest with a smaller max_pages.",
        )


NO_CONTENT_ANSWER = (
    "The ingested content does not appear to contain information relevant "
    "to this question, so no grounded answer can be given."
)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question using only the ingested content, with citations.

    Called by: the client (you, via /docs or curl).
    Calls, in order (the Phase 2 hybrid retrieval funnel):
        _require_keys → store.count (guard)
        → embeddings.embed_query + store.search     (vector arm, top 20)
        → keyword.search                            (BM25 arm, top 20)
        → fusion.rrf                                (merge → 10 candidates)
        → rerank.rerank                             (cross-encoder → top_k scored)
        → min_relevance threshold                   (all below? → refuse, no LLM call)
        → qa.answer                                 (LLM, unchanged)
    Request/response shapes: AskRequest / AskResponse in schemas.py.
    The `sources` list uses the same [n] numbering the LLM cites in the
    answer text, so every claim can be traced back to a page.
    """
    _require_keys(voyage=True, llm=True)

    if store.count() == 0:
        raise HTTPException(status_code=409, detail="Nothing ingested yet — call /ingest first.")

    try:
        # The full hybrid funnel (embed → vector + BM25 → RRF → rerank) now
        # lives in rag/pipeline.py so the agent's search_content tool shares it.
        ranked = pipeline.retrieve(request.question, top_k=request.top_k)
    except voyageai.error.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Voyage free-tier rate limit hit — wait ~20 seconds and ask again.",
        )
    except voyageai.error.VoyageError as exc:
        # FAIL CLOSED: if we can't verify relevance (rerank/embed API down,
        # timeout, 5xx), we refuse to answer rather than degrade to unranked
        # chunks. Wrong-but-confident is the worst outcome for a system whose
        # output is shown to third parties; "try again later" is recoverable.
        logger.error("retrieval failed closed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Could not verify retrieval relevance (embedding/rerank API "
            "error) — refusing to answer rather than guess. Try again shortly.",
        )

    # The "no relevant content" gate: top-k retrieval ALWAYS returns k chunks,
    # but the reranker's calibrated score tells us if they're actually about
    # the question. Nothing clears the bar → honest refusal, zero LLM tokens.
    relevant = [hit for hit in ranked if hit["relevance"] >= settings.min_relevance]
    if not relevant:
        # Log every refusal with its top score — this stream is eval data:
        # it shows where the threshold bites and feeds future tuning.
        top_score = ranked[0]["relevance"] if ranked else 0.0
        logger.info(
            "REFUSED (below min_relevance=%.2f): top_score=%.3f question=%r",
            settings.min_relevance,
            top_score,
            request.question,
        )
        return AskResponse(answer=NO_CONTENT_ANSWER, sources=[])

    answer_text = qa.answer(request.question, relevant)

    sources = [
        Source(index=i + 1, url=hit["url"], snippet=hit["text"][:200])
        for i, hit in enumerate(relevant)
    ]
    return AskResponse(answer=answer_text, sources=sources)


@app.post("/report", response_model=ReportResponse)
def report(panel: bool = False, deep: bool = False) -> ReportResponse:
    """Produce the structured First Impression report from ingested content.

    Called by: the client (you). Takes no body — it analyzes whatever site is
    currently ingested. ?panel=true additionally runs the Phase 4 persona
    panel (Technical Evaluator / Business Buyer / First-Time End User judge
    the same evidence in parallel; report gains persona_panel).
    Calls: generate_report() (agent/report.py), which runs the ReAct agent
    (explore with tools) then a schema-constrained synthesis call.

    Runtime: ~2-4 minutes under free-tier pacing (many Gemini calls + a few
    Voyage calls from the search_content tool); the request blocks meanwhile.
    Async/background jobs are a known future improvement, not this phase.
    """
    # The report agent needs Voyage (its search_content tool) plus the NVIDIA key
    # — the report pipeline runs entirely on the NVIDIA pool (GLM-led chain).
    _require_keys(voyage=True)
    if not settings.nvidia_api_key:
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY is not set in .env")

    if store.count() == 0:
        raise HTTPException(status_code=409, detail="Nothing ingested yet — call /ingest first.")

    try:
        report_obj, steps_log, pages_examined = generate_report(
            panel=panel, mode="deep" if deep else "normal"
        )
    except voyageai.error.RateLimitError:
        # The agent's search_content tool calls Voyage; free-tier limit → 429.
        raise HTTPException(
            status_code=429,
            detail="Voyage free-tier rate limit hit during analysis — wait a "
            "minute and retry.",
        )
    except openai.RateLimitError:
        # The whole NVIDIA chain exhausted its rate limit despite in-pool retry
        # and failover (the four models share one account quota).
        raise HTTPException(
            status_code=429,
            detail="NVIDIA rate limit hit during analysis — wait a minute and retry.",
        )
    except openai.BadRequestError as exc:
        # Persistent 'tool_use_failed' after in-pool retries: the model kept
        # emitting malformed tool-call syntax. Upstream failure, not our bug.
        if "tool_use_failed" not in str(exc):
            raise
        raise HTTPException(
            status_code=502,
            detail="The exploration model repeatedly produced malformed tool "
            "calls — please retry.",
        )
    except InsufficientEvidenceError as exc:
        # Store too thin to ground a report — a state problem, not a synthesis
        # failure. Refuse (409) instead of letting the LLM hallucinate one.
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        # The synthesis call returned no parseable report (see agent drivers).
        raise HTTPException(status_code=502, detail=str(exc))

    tool_calls = [f"{s['tool']}({s['args']})" for s in steps_log]
    return ReportResponse(
        report=report_obj,
        steps_taken=len(steps_log),
        pages_examined=pages_examined,
        tool_calls=tool_calls,
    )


# --- Streaming dashboard (Phase 6): ingest + report in one live SSE run ---

_STATIC_DIR = Path(__file__).parent / "static"


def _analysis_worker(url: str, max_pages: int, panel: bool, deep: bool, q: queue.Queue) -> None:
    """Run the FULL pipeline (ingest → report) on a background thread, emitting
    progress into `q` via events.collector. The sync clients (Groq/Voyage/
    LangGraph) are blocking, so a thread keeps the SSE generator responsive.
    Terminates with a 'report.done' (or 'error') event, then a None sentinel."""
    with events.collector(q):
        try:
            _ingest_site(url, max_pages)
            events.emit("phase", name="analyze")
            report_obj, steps_log, pages_examined = generate_report(
                panel=panel, mode="deep" if deep else "normal"
            )
            events.emit(
                "report.done",
                report=report_obj.model_dump(),
                steps_taken=len(steps_log),
                pages_examined=pages_examined,
            )
        except Exception as exc:  # any failure → one clean error event for the UI
            logger.exception("analysis stream failed")
            events.emit("error", message=f"{type(exc).__name__}: {exc}")
        finally:
            q.put(None)  # sentinel: tell the SSE generator to close


@app.get("/analyze/stream")
def analyze_stream(
    url: str, max_pages: int = 15, panel: bool = True, deep: bool = False
) -> StreamingResponse:
    """Server-Sent Events: run ingest+report for `url` and stream each step
    (crawl.page, render.escalate, ingest.done, tool, persona, report.done) so
    the dashboard can show the agent working live. One event per SSE 'data:'
    line as compact JSON. deep=true uses the extreme-depth pipeline."""
    _require_keys(voyage=True)
    q: queue.Queue = queue.Queue()
    threading.Thread(
        target=_analysis_worker, args=(url, max_pages, panel, deep, q), daemon=True
    ).start()

    def event_stream():
        while True:
            event = q.get()
            if event is None:  # worker finished
                break
            yield f"data: {json.dumps(event)}\n\n"

    # no-transform/no-cache so proxies don't buffer the stream.
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def dashboard() -> FileResponse:
    """Serve the single-page dashboard."""
    return FileResponse(_STATIC_DIR / "index.html")
