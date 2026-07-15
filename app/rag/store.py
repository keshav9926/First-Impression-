# app/rag/store.py — the vector store (Chroma).
# Chroma runs INSIDE our process and persists to a local folder — no separate
# server. It stores (id, text, metadata, vector) rows and answers the one
# query that matters for RAG: "give me the k stored vectors nearest to this
# query vector" (nearest = most similar meaning).
#
# CALL FLOW:
#   main.py: ingest() → replace_all(chunks, vectors)   save everything
#   main.py: ask()    → count()                        anything ingested?
#                     → search(query_vector, top_k)    find relevant chunks
#   All three go through _collection() to open the same on-disk collection.
#
# Phase 1 simplification: one collection holding one company's docs;
# re-ingesting replaces it. Multi-company support can come later.

import chromadb

from app.config import settings


def _collection(reset: bool = False) -> chromadb.Collection:
    """Open (or create) our one Chroma collection from the ./chroma_data folder.

    Called by: replace_all(), search(), count() — every function below.
    reset=True first deletes the existing collection — used only by
    replace_all() so each /ingest starts from a clean slate.
    """
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    if reset:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass  # collection didn't exist yet — nothing to delete
    return client.get_or_create_collection(settings.collection_name)


def replace_all(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """Wipe the collection and store these chunks. Each chunk: {"text", "url"}.

    Called by: main.py ingest(), as the LAST step of ingestion.
    Receives: chunks from chunker.py + matching vectors from embeddings.py
    (same order — chunk i belongs to vector i).

    We pass our own Voyage embeddings explicitly — otherwise Chroma would
    silently embed with its built-in default model, and queries embedded by
    Voyage would be compared against vectors from a different model
    (meaningless distances, terrible retrieval).
    """
    collection = _collection(reset=True)
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        # url kept for citations; headings (the page's section map, one joined
        # string) for the agent's read_page; extraction_warning tells the agent
        # a static crawl only captured a fraction of a JS-rendered site.
        # .get defaults: chunks from before these features keep working.
        metadatas=[
            {
                "url": c["url"],
                "headings": c.get("headings", ""),
                "extraction_warning": c.get("extraction_warning", False),
            }
            for c in chunks
        ],
        embeddings=embeddings,
    )
    return collection.count()


def search(query_embedding: list[float], top_k: int) -> list[dict]:
    """Return the top_k most similar chunks: [{"id", "text", "url", "distance"}].

    Called by: main.py ask(), with the vector from embeddings.embed_query().
    Output goes to: fusion.rrf() — the "id" is the dedup key that lets fusion
    recognize when vector search and BM25 found the SAME chunk.

    "distance": lower = more similar. Chroma compares the query vector
    against every stored vector and returns the nearest ones. top_k is
    capped at the collection size (Chroma rejects asking for more rows
    than exist).
    """
    collection = _collection()
    n_results = min(top_k, collection.count())
    if n_results == 0:
        return []
    result = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    hits = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"id": chunk_id, "text": text, "url": meta["url"], "distance": distance})
    return hits


def all_chunks() -> list[dict]:
    """Return EVERY stored chunk: [{"id", "text", "url"}] — no ranking.

    Called by: keyword.py search(), which needs the full corpus to build
    its BM25 index (keyword scoring is relative to the whole collection).
    """
    collection = _collection()
    result = collection.get()  # no filter = everything
    chunks = [
        # .get defaults: data ingested before the heading-map / thin-extraction
        # features has no such metadata — old stores keep working.
        {
            "id": chunk_id,
            "text": text,
            "url": meta["url"],
            "headings": meta.get("headings", ""),
            "extraction_warning": meta.get("extraction_warning", False),
        }
        for chunk_id, text, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]
    # Chroma .get() does not guarantee order; sort by the numeric chunk index
    # ("chunk-0", "chunk-1", ...) so a page's chunks come back in reading order
    # (the agent's read_page tool concatenates them and needs them in sequence).
    chunks.sort(key=lambda c: int(c["id"].split("-")[1]))
    return chunks


def count() -> int:
    """How many chunks are stored right now?

    Called by: main.py ask() — if 0, it returns a 409 ("ingest first")
    instead of searching an empty store.
    """
    return _collection().count()
