# app/schemas.py — Pydantic models defining the API's request/response shapes.
# FastAPI uses these to (a) reject malformed requests with a 422 before our
# code runs, (b) guarantee responses match the declared shape, and (c) render
# the /docs page. Field(...) constraints are validation rules, not comments —
# max_pages=999 is rejected by the framework, we never see it.

from pydantic import BaseModel, Field, HttpUrl


class IngestRequest(BaseModel):
    url: HttpUrl  # HttpUrl (not str) = must parse as a valid http(s) URL
    max_pages: int = Field(default=15, ge=1, le=50)


class IngestResponse(BaseModel):
    pages_fetched: int
    chunks_stored: int
    skipped_by_robots: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)  # how many chunks to retrieve


class Source(BaseModel):
    index: int  # matches the [n] citations in the answer
    url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
