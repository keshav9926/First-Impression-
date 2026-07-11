# evals/run_retrieval_eval.py — measures retrieval quality, before vs after.
# Run:  uv run python evals/run_retrieval_eval.py
#
# WHAT IT DOES: for every question in retrieval_eval.json it runs BOTH
# retrieval pipelines against the live Chroma store —
#   OLD (Phase 1): vector search only, top 5
#   NEW (Phase 2): vector + BM25 → RRF fusion → rerank, top 5
# and checks whether a chunk from the EXPECTED page made the top 5.
#
# METRICS:
#   hit@5 — fraction of questions where a correct-page chunk is in the top 5.
#   MRR   — Mean Reciprocal Rank: 1/rank of the FIRST correct chunk, averaged.
#           (Correct chunk at rank 1 scores 1.0, at rank 5 scores 0.2, absent
#           scores 0 — so MRR rewards putting the right chunk NEAR THE TOP,
#           which matters because the LLM reads better-ranked chunks first.)
#
# FREE-TIER NOTE: each question costs 1 embed call (old+new share it) and
# 1 rerank call. Voyage free tier = 3 requests/minute, so we sleep between
# questions. A 10-question run takes ~7 minutes. Slow is fine — evals are
# batch jobs, not user-facing.

import json
import sys
import time
from pathlib import Path

# Make `app.*` importable when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag import embeddings, fusion, keyword, rerank, store  # noqa: E402

TOP_K = 5
SECONDS_BETWEEN_QUESTIONS = 45  # 2 Voyage calls per question @ 3/min budget


def old_pipeline(question: str) -> list[dict]:
    """Phase 1 retrieval: vector search only."""
    return store.search(embeddings.embed_query(question), top_k=TOP_K)


def new_pipeline(question: str) -> list[dict]:
    """Phase 2 retrieval: hybrid → fusion → rerank."""
    vector_hits = store.search(embeddings.embed_query(question), top_k=20)
    keyword_hits = keyword.search(question, top_k=20)
    candidates = fusion.rrf(vector_hits, keyword_hits, limit=10)
    return rerank.rerank(question, candidates, top_k=TOP_K)


def first_correct_rank(hits: list[dict], expected_url_contains: str) -> int | None:
    """Return the 1-based rank of the first hit from the expected page."""
    for position, hit in enumerate(hits):
        if expected_url_contains in hit["url"]:
            return position + 1
    return None


def main() -> None:
    cases = json.loads(
        (Path(__file__).parent / "retrieval_eval.json").read_text(encoding="utf-8")
    )["cases"]

    if store.count() == 0:
        sys.exit("Store is empty — run /ingest for the eval site first.")

    results = []  # (question, old_rank, new_rank)
    for i, case in enumerate(cases):
        question, expected = case["question"], case["expected_url_contains"]
        print(f"[{i + 1}/{len(cases)}] {question}")

        # One embed call, shared by both pipelines (identical input → reuse).
        query_vector = embeddings.embed_query(question)
        old_hits = store.search(query_vector, top_k=TOP_K)

        vector_hits = store.search(query_vector, top_k=20)
        keyword_hits = keyword.search(question, top_k=20)
        candidates = fusion.rrf(vector_hits, keyword_hits, limit=10)
        new_hits = rerank.rerank(question, candidates, top_k=TOP_K)

        old_rank = first_correct_rank(old_hits, expected)
        new_rank = first_correct_rank(new_hits, expected)
        results.append((question, old_rank, new_rank))
        print(f"    old: rank {old_rank}   new: rank {new_rank}")

        if i < len(cases) - 1:
            time.sleep(SECONDS_BETWEEN_QUESTIONS)  # free-tier pacing

    # --- Report ---
    def hit_rate(ranks: list[int | None]) -> float:
        return sum(1 for r in ranks if r is not None) / len(ranks)

    def mrr(ranks: list[int | None]) -> float:
        return sum(1 / r for r in ranks if r is not None) / len(ranks)

    old_ranks = [r[1] for r in results]
    new_ranks = [r[2] for r in results]

    print("\n===== RETRIEVAL EVAL =====")
    print(f"{'question':<55} {'old':>4} {'new':>4}")
    for question, old_rank, new_rank in results:
        print(f"{question[:53]:<55} {str(old_rank):>4} {str(new_rank):>4}")
    print("-" * 65)
    print(f"hit@{TOP_K}:  old {hit_rate(old_ranks):.0%}   new {hit_rate(new_ranks):.0%}")
    print(f"MRR:     old {mrr(old_ranks):.2f}   new {mrr(new_ranks):.2f}")


if __name__ == "__main__":
    main()
