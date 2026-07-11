# app/rag/fusion.py — Reciprocal Rank Fusion (RRF), hand-written.
# Merges the vector ranking and the BM25 ranking into ONE candidate list.
#
# CALL FLOW:
#   main.py: ask() → rrf(vector_hits, keyword_hits, limit)
#   Output (fused candidates) goes to rerank.rerank() for final scoring.
#
# THE PROBLEM RRF SOLVES: the two retrievers speak different score languages —
# Chroma returns distances (lower = better, unbounded), BM25 returns scores
# (higher = better, unbounded). You cannot average apples and oranges.
# RRF sidesteps scores entirely and uses only each chunk's RANK in each list:
#
#     fused_score(chunk) = Σ over lists   1 / (K + rank_in_that_list)
#
# with rank starting at 1 and K = 60 (the constant from the original RRF
# paper; it damps the gap between rank 1 and rank 2 so one list can't
# dominate). A chunk ranked #1 by BOTH lists gets 2/61 ≈ 0.033 — the maximum.
# A chunk found by only one list still earns that list's contribution, so
# hybrid keeps unique finds from either side.

K = 60  # standard RRF damping constant


def rrf(*ranked_lists: list[dict], limit: int, guaranteed_per_list: int = 0) -> list[dict]:
    """Fuse any number of ranked hit-lists into one, best-first, deduplicated.

    Called by: main.py ask(), with the vector list and the keyword list.
    Calls: nothing — pure function (like the chunker), trivially testable.

    Each hit must carry a unique "id" (set by store.py) — that's how we know
    two lists found the SAME chunk and should add their contributions.

    guaranteed_per_list: each input list's top-N hits are ALWAYS included,
    even if consensus scoring would drop them. Why (found by eval debugging,
    2026-07-12): with K=60, a chunk two lists rank mediocrely (~1/78 + 1/73)
    outscores a chunk ONE list ranks #1 (1/61) — so a vector-rank-1 chunk
    that BM25's naive tokenizer misses ("shifts" ≠ "shift", no stemming)
    was eliminated before the reranker ever saw it. Fusion's job is to
    NOMINATE candidates; the cross-encoder is the judge. Excellence in a
    single list must reach the judge. Output may exceed `limit` by a few —
    the reranker's top_k makes the final cut anyway.
    """
    fused: dict[str, dict] = {}  # id -> hit with accumulated "rrf_score"

    for ranked_list in ranked_lists:
        for position, hit in enumerate(ranked_list):
            rank = position + 1  # ranks are 1-based
            contribution = 1.0 / (K + rank)
            entry = fused.get(hit["id"])
            if entry is None:
                fused[hit["id"]] = {**hit, "rrf_score": contribution}
            else:
                entry["rrf_score"] += contribution

    by_score = sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)
    selected = by_score[:limit]
    selected_ids = {h["id"] for h in selected}

    # Guaranteed seats: any list's top-N that consensus dropped gets appended.
    for ranked_list in ranked_lists:
        for hit in ranked_list[:guaranteed_per_list]:
            if hit["id"] not in selected_ids:
                selected.append(fused[hit["id"]])
                selected_ids.add(hit["id"])

    return selected
