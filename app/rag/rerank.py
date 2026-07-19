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

import math
import time

import httpx
import voyageai
import voyageai.error

from app.config import settings

# --- NVIDIA reranking (rerank_provider="nvidia") --------------------------
# ai.api /v1/retrieval/nvidia/reranking. Returns unbounded LOGITS (higher =
# better), not Voyage's calibrated 0..1. We sigmoid-map each logit into (0,1)
# so the rest of the app (min_relevance gate, near-miss margin, /ask refusal)
# keeps working on a 0..1 scale unchanged. No 3-req/min throttle → no pacing.
_NVIDIA_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
_NVIDIA_RERANK_RETRIES = 4


def _nvidia_rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    payload = {
        "model": settings.nvidia_rerank_model,
        "query": {"text": question},
        "passages": [{"text": c["text"]} for c in candidates],
    }
    headers = {"Authorization": f"Bearer {settings.nvidia_api_key}"}
    for attempt in range(_NVIDIA_RERANK_RETRIES):
        try:
            resp = httpx.post(_NVIDIA_RERANK_URL, headers=headers, json=payload, timeout=40)
            resp.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == _NVIDIA_RERANK_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    rankings = resp.json()["rankings"]  # [{index, logit}], best-first
    center, scale = settings.nvidia_rerank_center, settings.nvidia_rerank_scale
    out = []
    for item in rankings[:top_k]:
        # shifted sigmoid → (0,1): recenters strongly-negative logits so relevant
        # separates from junk (a plain sigmoid bunches them all near 0).
        relevance = 1.0 / (1.0 + math.exp(-(item["logit"] - center) / scale))
        out.append({**candidates[item["index"]], "relevance": relevance})
    return out

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

    if settings.rerank_provider == "nvidia":
        return _nvidia_rerank(question, candidates, top_k)

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
