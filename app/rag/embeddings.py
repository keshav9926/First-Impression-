# app/rag/embeddings.py — turns text into vectors via the Voyage AI API.
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
    return voyageai.Client(api_key=settings.voyage_api_key)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks for storage. Batched to respect the API's per-request limit."""
    client = _client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        result = client.embed(batch, model=settings.embedding_model, input_type="document")
        vectors.extend(result.embeddings)
    return vectors


def embed_query(question: str) -> list[float]:
    """Embed a user question for searching against stored document vectors."""
    result = _client().embed([question], model=settings.embedding_model, input_type="query")
    return result.embeddings[0]
