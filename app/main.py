# app/main.py — the FastAPI application entry point.
# Defines the app object that Uvicorn serves and registers all endpoints.
# Endpoints stay thin: validate → call the right module → shape the response.
# The real logic lives in app/ingestion/ and app/rag/.

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
    """Fail with a clear 503 if a needed API key isn't configured (see config.py)."""
    if voyage and not settings.voyage_api_key:
        raise HTTPException(status_code=503, detail="VOYAGE_API_KEY is not set in .env")
    if claude and not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set in .env")


@app.get("/health")
def health() -> dict:
    """Liveness check: proves the server is up and config loaded correctly."""
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Crawl a public site (robots.txt-compliant), chunk, embed, and store it."""
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
    """Answer a question using only the ingested content, with citations."""
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
