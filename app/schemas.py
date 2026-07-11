# app/schemas.py — Pydantic models defining the API's request/response shapes.
#
# WHERE THESE ARE USED:
#   IngestRequest / IngestResponse → main.py ingest()  (POST /ingest)
#   AskRequest / AskResponse       → main.py ask()     (POST /ask)
#   Source                         → nested inside AskResponse
#
# FastAPI uses these to (a) reject malformed requests with a 422 before our
# code runs, (b) guarantee responses match the declared shape, and (c) render
# the /docs page. Field(...) constraints are validation RULES, not comments —
# max_pages=999 is rejected by the framework; our code never sees it.

from pydantic import BaseModel, Field, HttpUrl


class IngestRequest(BaseModel):
    """Body of POST /ingest — what site to crawl and how far.

    Validated by FastAPI before main.ingest() runs.
    """

    url: HttpUrl  # HttpUrl (not str) = must parse as a valid http(s) URL
    max_pages: int = Field(default=15, ge=1, le=50)  # ge/le = min/max allowed


class IngestResponse(BaseModel):
    """What POST /ingest returns — a summary of what was crawled and stored.

    Built at the end of main.ingest(); skipped_by_robots comes from
    fetcher.CrawlResult so rule-#1 enforcement is visible to the caller.
    """

    pages_fetched: int
    chunks_stored: int
    skipped_by_robots: int


class AskRequest(BaseModel):
    """Body of POST /ask — the question, and how many chunks to retrieve.

    Validated by FastAPI before main.ask() runs.
    """

    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)  # how many chunks to retrieve


class Source(BaseModel):
    """One retrieved chunk shown to the caller as a citation.

    Nested inside AskResponse. `index` matches the [n] markers Claude
    writes in the answer text (numbering assigned in qa.answer()).
    """

    index: int
    url: str
    snippet: str


class AskResponse(BaseModel):
    """What POST /ask returns — the grounded answer plus its citations.

    Built at the end of main.ask(): `answer` from qa.answer(),
    `sources` from the same hits that were given to Claude.
    """

    answer: str
    sources: list[Source]
