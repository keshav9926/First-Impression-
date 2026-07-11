# app/rag/store.py — the vector store (Chroma).
# Chroma runs INSIDE our process and persists to a local folder — no separate
# server. It stores (id, text, metadata, vector) rows and answers the one
# query that matters for RAG: "give me the k stored vectors nearest to this
# query vector" (nearest = most similar meaning).
#
# Phase 1 simplification: one collection holding one company's docs;
# re-ingesting replaces it. Multi-company support can come later.

import chromadb

from app.config import settings


def _collection(reset: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    if reset:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass  # collection didn't exist yet — nothing to delete
    return client.get_or_create_collection(settings.collection_name)


def replace_all(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """Wipe the collection and store these chunks. Each chunk: {"text", "url"}.

    We pass our own Voyage embeddings explicitly — otherwise Chroma would
    silently embed with its built-in default model, and queries embedded by
    Voyage would be compared against vectors from a different model
    (meaningless distances, terrible retrieval).
    """
    collection = _collection(reset=True)
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"url": c["url"]} for c in chunks],
        embeddings=embeddings,
    )
    return collection.count()


def search(query_embedding: list[float], top_k: int) -> list[dict]:
    """Return the top_k most similar chunks: [{"text", "url", "distance"}]."""
    collection = _collection()
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"text": text, "url": meta["url"], "distance": distance})
    return hits


def count() -> int:
    return _collection().count()
