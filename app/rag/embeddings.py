# app/rag/embeddings.py — turns text into vectors via the Voyage AI API.
#
# CALL FLOW:
#   main.py: ingest() → embed_documents(all_chunk_texts)  (ingestion side)
#   main.py: ask()    → embed_query(question)             (question side)
#   Both produce vectors that store.py saves or searches with.
#
# WHAT an embedding is: a list of ~1000 floats representing a text's MEANING
# as a point in space. Texts with similar meaning land close together, so
# "how much does it cost" retrieves the pricing paragraph even though the
# page says "plans start at $20/mo" and shares no keywords with the question.
#
# WHY input_type matters: Voyage embeds documents and queries slightly
# differently (a question and its answer should land NEAR each other even
# though a question is not phrased like an answer). Passing the wrong type
# silently degrades retrieval quality — a classic RAG bug.

import time

import voyageai
import voyageai.error

from app.config import settings

# --- Free-tier pacing ---
# Without a payment method, Voyage allows 3 requests/minute and 10K tokens/
# minute. So each batch must stay under ~10K tokens (we approximate tokens
# with chars: ~4 chars ≈ 1 token → 28K chars ≈ 7K tokens, a safe margin),
# and batches must be ≥20s apart (3 per minute). Ingest gets slower; it
# stays free. With a paid account these could be 128 texts / no sleep.
_MAX_BATCH_TEXTS = 128  # hard API limit: max texts per request
_MAX_BATCH_CHARS = 28_000  # ~7K tokens — under the free tier's 10K TPM
_SECONDS_BETWEEN_BATCHES = 21  # free tier: max 3 requests per minute
# The /report agent fires several search_content calls in one run, each an
# embed_query — a burst easily trips the 3-requests/minute cap. A single 429
# there would discard the whole (already-paid-for) exploration, so embed_query
# paces and retries into the next minute-window instead of failing the run.
_MAX_QUERY_RETRIES = 4


def _client() -> voyageai.Client:
    """Build a Voyage API client using the key from config.py.

    Called by: embed_documents() and embed_query() below.
    Kept as a tiny helper so the key wiring lives in exactly one place.
    """
    return voyageai.Client(api_key=settings.voyage_api_key)


def _make_batches(texts: list[str]) -> list[list[str]]:
    """Group texts into batches that fit BOTH free-tier limits:
    ≤ _MAX_BATCH_TEXTS texts and ≤ _MAX_BATCH_CHARS characters per batch.

    Called by: embed_documents().
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        if current and (
            len(current) >= _MAX_BATCH_TEXTS or current_chars + len(text) > _MAX_BATCH_CHARS
        ):
            batches.append(current)
            current, current_chars = [], 0
        current.append(text)
        current_chars += len(text)
    if current:
        batches.append(current)
    return batches


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks for STORAGE: one vector per chunk, in the same order.

    Called by: main.py ingest(), right after chunking.
    Calls: _make_batches() to fit free-tier limits, then the Voyage API
    per batch — sleeping between batches to respect the 3-requests/minute cap.
    Output goes to: store.replace_all(), paired back up with its chunks.
    """
    client = _client()
    vectors: list[list[float]] = []
    batches = _make_batches(texts)
    for i, batch in enumerate(batches):
        if i > 0:
            time.sleep(_SECONDS_BETWEEN_BATCHES)  # pace to 3 requests/minute
        result = client.embed(batch, model=settings.embedding_model, input_type="document")
        vectors.extend(result.embeddings)
    return vectors


def embed_query(question: str) -> list[float]:
    """Embed a user question for SEARCHING against stored document vectors.

    Called by: main.py ask().
    Calls: _client(), then the Voyage API.
    Output goes to: store.search(), which finds the nearest stored chunks.

    Note input_type="query" (vs "document" above) — see the header comment.
    Retries free-tier 429s by pacing into the next per-minute window; other
    Voyage errors propagate (callers map them to HTTP status codes).
    """
    for attempt in range(_MAX_QUERY_RETRIES):
        try:
            result = _client().embed(
                [question], model=settings.embedding_model, input_type="query"
            )
            return result.embeddings[0]
        except voyageai.error.RateLimitError:
            if attempt == _MAX_QUERY_RETRIES - 1:
                raise  # exhausted — let the caller surface a 429
            time.sleep(_SECONDS_BETWEEN_BATCHES)
