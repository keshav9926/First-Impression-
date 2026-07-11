# app/rag/keyword.py — BM25 keyword search over the stored chunks.
# The second half of hybrid search: vectors find MEANING matches, BM25 finds
# EXACT-WORD matches (product names, feature names, "SKU", "OEE") that
# embeddings often rank too low.
#
# CALL FLOW:
#   main.py: ask() → search(question, top_k)
#       ├── store.all_chunks()      pull every stored chunk out of Chroma
#       ├── _tokenize()             both corpus and query
#       └── rank_bm25.BM25Okapi     score every chunk against the query
#   Output (ranked hits) goes to fusion.rrf() to be merged with vector hits.
#
# BM25 INTUITION (the formula in words): a chunk scores high for a query word if
#   1. the word is RARE across the whole corpus  (matching "OEE" says more
#      than matching "the" — rarity is measured by inverse document frequency)
#   2. the word appears in this chunk, with diminishing returns (2 mentions
#      beat 1; 20 don't beat 10 by much)
#   3. the chunk is SHORT (a match in 3 lines is stronger evidence than a
#      match buried in 3 pages)
# Sum over the query's words = the chunk's score. No ML, no vectors — pure
# counting. That's why it catches exact terms that embeddings blur away.
#
# SCALE NOTE: we rebuild the index on every call — fine at our size (a few
# hundred chunks, ~milliseconds). At tens of thousands of chunks you'd cache
# the index and invalidate it on re-ingest.

import re

from rank_bm25 import BM25Okapi

from app.rag import store


def _tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens — applied identically to the
    corpus and the query so they meet in the same token space.

    Called by: search(), for every chunk and for the question.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def search(question: str, top_k: int) -> list[dict]:
    """Return the top_k chunks by BM25 keyword score, best first.

    Called by: main.py ask() — in parallel with store.search() (vectors).
    Calls: store.all_chunks() for the corpus, then BM25Okapi for scoring.
    Output goes to: fusion.rrf(), keyed by chunk "id".

    Chunks with a score of 0 (no query word appears at all) are dropped —
    an all-zeros "ranking" would just be noise for the fusion step.
    """
    chunks = store.all_chunks()
    if not chunks:
        return []

    corpus_tokens = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenize(question))

    scored = [
        {**chunk, "keyword_score": float(score)}
        for chunk, score in zip(chunks, scores)
        if score > 0
    ]
    scored.sort(key=lambda h: h["keyword_score"], reverse=True)
    return scored[:top_k]
