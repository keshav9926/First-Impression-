# app/rag/rerank.py — cross-encoder re-ranking via Voyage's rerank API.
# The precision stage of retrieval: candidates found by hybrid search get
# re-scored by a model that reads question + chunk TOGETHER.
#
# CALL FLOW:
#   main.py: ask() → rerank(question, candidates, top_k)
#   Input came from: fusion.rrf() (the ~10 fused candidates).
#   Output (top_k hits with a calibrated "relevance" score) goes back to
#   main.py, which applies settings.min_relevance — the "no relevant
#   content" gate — before anything reaches the LLM.
#
# WHY A SEPARATE MODEL: embeddings are a BI-encoder — question and chunk are
# encoded separately and compared as vectors (fast: chunk vectors are
# precomputed; coarse: the two texts never "see" each other). A RERANKER is a
# CROSS-encoder — one transformer reads "question [SEP] chunk" as a single
# input and outputs a relevance score (sharp: full attention between every
# question word and every chunk word; slow: must run per question-chunk pair).
# So the classic funnel: cheap search over everything → expensive scoring
# over the top few. Its 0..1 score is also CALIBRATED, which is what makes a
# fixed relevance threshold meaningful — raw vector distances are not.

import time

import voyageai
import voyageai.error

from app.config import settings

# The /report agent fires several search_content calls in a burst, each ending
# in a rerank — the same 3-requests/minute free-tier cap that embed_query paces
# around. Without this, a mid-loop 429 kills the whole report. Mirror
# embeddings.embed_query: retry into the next minute-window.
_MAX_RERANK_RETRIES = 4
_SECONDS_BETWEEN = 21


def rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates against the question; return top_k, best first.

    Called by: main.py ask(), after fusion.
    Calls: the Voyage rerank API (network call — counts toward the same
    free-tier request budget as embeddings; RateLimitError is mapped to a
    429 by the endpoint).

    Each returned hit keeps its original fields and gains "relevance"
    (0..1-ish, higher = better). The API returns results sorted with an
    .index pointing back into the documents list we sent — we use that to
    reattach scores to the right chunks.
    """
    if not candidates:
        return []

    client = voyageai.Client(api_key=settings.voyage_api_key)
    for attempt in range(_MAX_RERANK_RETRIES):
        try:
            result = client.rerank(
                query=question,
                documents=[c["text"] for c in candidates],
                model=settings.rerank_model,
                top_k=top_k,
            )
            break
        except voyageai.error.RateLimitError:
            if attempt == _MAX_RERANK_RETRIES - 1:
                raise  # exhausted — caller maps to a 429
            time.sleep(_SECONDS_BETWEEN)

    reranked = []
    for item in result.results:  # already sorted by relevance, best first
        hit = {**candidates[item.index], "relevance": float(item.relevance_score)}
        reranked.append(hit)
    return reranked
