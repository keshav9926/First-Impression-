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
#   POST /ask → ask()
#       ├── _require_keys(voyage=True, claude=True)
#       ├── store.count()                      empty? → 409 "ingest first"
#       ├── embeddings.embed_query(question)   question → vector (Voyage API)
#       ├── store.search(vector, top_k)        nearest chunks from Chroma
#       └── qa.answer(question, hits)          Claude answers from chunks only

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.ingestion.chunker import chunk_text
from app.ingestion.fetcher import crawl
from app.ingestion.robots import is_allowed
from app.rag import embeddings, qa, store
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse, Source

app = FastAPI(
    title=settings.app_name,
    description="Analyzes a startup's public product experience and reports on the new-user journey.",
)


def _require_keys(*, voyage: bool = False, claude: bool = False) -> None:
    """Fail with a clear 503 if a needed API key isn't configured.

    Called by: ingest() and ask(), as their first line.
    Why: keys default to "" in config.py so /health and tests run without
    them — so endpoints that DO need a key must check explicitly, and a
    descriptive 503 beats a confusing auth error from deep inside an SDK.
    """
    if voyage and not settings.voyage_api_key:
        raise HTTPException(status_code=503, detail="VOYAGE_API_KEY is not set in .env")
    if claude and not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set in .env")


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

    vectors = embeddings.embed_documents([c["text"] for c in chunks])
    stored = store.replace_all(chunks, vectors)

    return IngestResponse(
        pages_fetched=len(result.pages),
        chunks_stored=stored,
        skipped_by_robots=result.skipped_by_robots,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question using only the ingested content, with citations.

    Called by: the client (you, via /docs or curl).
    Calls, in order: _require_keys → store.count (guard) →
                     embeddings.embed_query → store.search → qa.answer
    Request/response shapes: AskRequest / AskResponse in schemas.py.
    The `sources` list uses the same [n] numbering Claude cites in the
    answer text, so every claim can be traced back to a page.
    """
    _require_keys(voyage=True, claude=True)

    if store.count() == 0:
        raise HTTPException(status_code=409, detail="Nothing ingested yet — call /ingest first.")

    query_vector = embeddings.embed_query(request.question)
    hits = store.search(query_vector, top_k=request.top_k)
    answer_text = qa.answer(request.question, hits)

    sources = [
        Source(index=i + 1, url=hit["url"], snippet=hit["text"][:200])
        for i, hit in enumerate(hits)
    ]
    return AskResponse(answer=answer_text, sources=sources)
