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

import voyageai

from app.config import settings

_BATCH_SIZE = 128  # Voyage API limit: max texts per request


def _client() -> voyageai.Client:
    """Build a Voyage API client using the key from config.py.

    Called by: embed_documents() and embed_query() below.
    Kept as a tiny helper so the key wiring lives in exactly one place.
    """
    return voyageai.Client(api_key=settings.voyage_api_key)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks for STORAGE: one vector per chunk, in the same order.

    Called by: main.py ingest(), right after chunking.
    Calls: _client(), then the Voyage API (network call, costs quota).
    Output goes to: store.replace_all(), paired back up with its chunks.

    Batched because the API accepts at most 128 texts per request —
    a big site can produce many hundreds of chunks.
    """
    client = _client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        result = client.embed(batch, model=settings.embedding_model, input_type="document")
        vectors.extend(result.embeddings)
    return vectors


def embed_query(question: str) -> list[float]:
    """Embed a user question for SEARCHING against stored document vectors.

    Called by: main.py ask().
    Calls: _client(), then the Voyage API.
    Output goes to: store.search(), which finds the nearest stored chunks.

    Note input_type="query" (vs "document" above) — see the header comment.
    """
    result = _client().embed([question], model=settings.embedding_model, input_type="query")
    return result.embeddings[0]
