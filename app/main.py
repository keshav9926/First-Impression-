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
#       ├── embeddings.embed_query(question)   question → vector (Voyage API)
#       ├── store.search(vector, 20)           vector arm: MEANING matches
#       ├── keyword.search(question, 20)       BM25 arm: EXACT-WORD matches
#       ├── fusion.rrf(both, limit=10)         merge rankings (no score mixing)
#       ├── rerank.rerank(question, cands)     cross-encoder scores 0..1 (Voyage)
#       ├── min_relevance gate                 all below bar? → refuse, skip LLM
#       └── qa.answer(question, relevant)      LLM answers from chunks only

import logging

import voyageai.error
from fastapi import FastAPI, HTTPException

from app.config import settings
from app.ingestion.chunker import chunk_text
from app.ingestion.fetcher import crawl
from app.ingestion.robots import is_allowed
from app.rag import embeddings, fusion, keyword, qa, rerank, store
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse, Source

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
    The llm check looks at whichever provider is active (config.llm_provider).
    """
    if voyage and not settings.voyage_api_key:
        raise HTTPException(status_code=503, detail="VOYAGE_API_KEY is not set in .env")
    if llm:
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


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Crawl a public site (robots.txt-compliant), chunk, embed, and store it.

    Called by: the client (you, via /docs or curl).
    Calls, in order: _require_keys → robots.is_allowed → fetcher.crawl
                     → chunker.chunk_text → embeddings.embed_documents
                     → store.replace_all
    Request/response shapes: IngestRequest / IngestResponse in schemas.py
    (FastAPI validated the body against IngestRequest BEFORE this runs).
    """
    _require_keys(voyage=True)

    url = str(request.url)
    if not is_allowed(url):
        # Hard rule #1 enforced at the API boundary: robots.txt says no → we stop.
        raise HTTPException(
            status_code=403,
            detail="robots.txt disallows fetching this URL (public-data rule).",
        )

    result = crawl(url, max_pages=request.max_pages)
    if not result.pages:
        raise HTTPException(status_code=404, detail="No readable pages found at this URL.")

    # Page texts -> chunks, each remembering which page it came from (for citations).
    chunks = [
        {"text": piece, "url": page.url}
        for page in result.pages
        for piece in chunk_text(page.text)
    ]

    try:
        vectors = embeddings.embed_documents([c["text"] for c in chunks])
    except voyageai.error.RateLimitError:
        # Free-tier limit hit despite our pacing — tell the caller to retry,
        # with the right status code (429 = Too Many Requests), not a raw 500.
        raise HTTPException(
            status_code=429,
            detail="Voyage free-tier rate limit hit — wait a minute and retry, "
            "or ingest with a smaller max_pages.",
        )
    stored = store.replace_all(chunks, vectors)

    return IngestResponse(
        pages_fetched=len(result.pages),
        chunks_stored=stored,
        skipped_by_robots=result.skipped_by_robots,
    )


# Hybrid retrieval funnel widths: cast wide, then narrow.
CANDIDATES_PER_RETRIEVER = 20  # each retriever's contribution to fusion
CANDIDATES_TO_RERANK = 10  # fused list size sent to the (slower) re-ranker

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
        # Recall stage: two retrievers with complementary blind spots.
        query_vector = embeddings.embed_query(request.question)
        vector_hits = store.search(query_vector, top_k=CANDIDATES_PER_RETRIEVER)
        keyword_hits = keyword.search(request.question, top_k=CANDIDATES_PER_RETRIEVER)

        # Merge stage: rank-based fusion (no score mixing).
        candidates = fusion.rrf(vector_hits, keyword_hits, limit=CANDIDATES_TO_RERANK)

        # Precision stage: cross-encoder scores each candidate 0..1.
        ranked = rerank.rerank(request.question, candidates, top_k=request.top_k)
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
