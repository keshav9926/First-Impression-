# evals/run_retrieval_eval.py — measures retrieval quality, before vs after,
# and grounds the min_relevance threshold in data instead of vibes.
# Run:  uv run python evals/run_retrieval_eval.py
#
# WHAT IT DOES:
#   Part 1 (answerable set): runs BOTH retrieval pipelines per question —
#     OLD (Phase 1): vector search only, top 5
#     NEW (Phase 2): vector + BM25 → RRF fusion → rerank, top 5
#   and checks whether a chunk from the EXPECTED page made the top 5.
#   Part 2 (unanswerable set): runs the NEW pipeline on questions the site
#   cannot answer, recording the top rerank score for each — anything that
#   clears min_relevance is a FALSE ANSWER (retrieval vouching for garbage).
#
# METRICS:
#   hit@5 — fraction of answerable questions with a correct-page chunk in top 5.
#   MRR   — Mean Reciprocal Rank: 1/rank of the FIRST correct chunk, averaged.
#           (Rank 1 → 1.0, rank 5 → 0.2, absent → 0 — rewards putting the
#           right chunk NEAR THE TOP, which matters because the LLM reads
#           better-ranked chunks first.)
#   false-answer rate — fraction of unanswerable questions where the top
#           score still cleared the threshold. Want: 0%.
#
# THRESHOLD TUNING: the report prints each question's top rerank score.
# A good min_relevance sits in the gap BETWEEN the unanswerable questions'
# top scores (should be below it) and the answerable ones' (above it).
# The script suggests the midpoint of that gap.
#
# FREE-TIER NOTE: each question costs 1 embed call + 1 rerank call at
# 3 requests/minute — a full run takes ~12 minutes. Slow is fine: evals
# are batch jobs, not user-facing.

import json
import sys
import time
from pathlib import Path

# Make `app.*` importable when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings  # noqa: E402
from app.rag import embeddings, fusion, keyword, rerank, store  # noqa: E402

TOP_K = 5
SECONDS_BETWEEN_QUESTIONS = 45  # 2 Voyage calls per question @ 3/min budget


def new_pipeline(question: str, query_vector: list[float]) -> list[dict]:
    """Phase 2 retrieval: hybrid → fusion → rerank."""
    vector_hits = store.search(query_vector, top_k=20)
    keyword_hits = keyword.search(question, top_k=20)
    candidates = fusion.rrf(vector_hits, keyword_hits, limit=10, guaranteed_per_list=3)
    return rerank.rerank(question, candidates, top_k=TOP_K)


def first_correct_rank(hits: list[dict], expected_url_contains: str) -> int | None:
    """Return the 1-based rank of the first hit from the expected page."""
    for position, hit in enumerate(hits):
        if expected_url_contains in hit["url"]:
            return position + 1
    return None


def main() -> None:
    data = json.loads(
        (Path(__file__).parent / "retrieval_eval.json").read_text(encoding="utf-8")
    )
    answerable, unanswerable = data["answerable"], data["unanswerable"]

    if store.count() == 0:
        sys.exit("Store is empty — run /ingest for the eval site first.")

    # ---- Part 1: answerable questions, old vs new ----
    results = []  # (question, old_rank, new_rank, new_top_score)
    for i, case in enumerate(answerable):
        question, expected = case["question"], case["expected_url_contains"]
        print(f"[answerable {i + 1}/{len(answerable)}] {question}")

        # One embed call, shared by both pipelines (identical input → reuse).
        query_vector = embeddings.embed_query(question)
        old_hits = store.search(query_vector, top_k=TOP_K)
        new_hits = new_pipeline(question, query_vector)

        old_rank = first_correct_rank(old_hits, expected)
        new_rank = first_correct_rank(new_hits, expected)
        top_score = new_hits[0]["relevance"] if new_hits else 0.0
        results.append((question, old_rank, new_rank, top_score))
        print(f"    old rank: {old_rank}   new rank: {new_rank}   top score: {top_score:.3f}")
        time.sleep(SECONDS_BETWEEN_QUESTIONS)

    # ---- Part 2: unanswerable questions, false-answer check ----
    refusals = []  # (question, top_score, falsely_answered)
    for i, case in enumerate(unanswerable):
        question = case["question"]
        print(f"[unanswerable {i + 1}/{len(unanswerable)}] {question}")

        query_vector = embeddings.embed_query(question)
        new_hits = new_pipeline(question, query_vector)
        top_score = new_hits[0]["relevance"] if new_hits else 0.0
        falsely_answered = top_score >= settings.min_relevance
        refusals.append((question, top_score, falsely_answered))
        print(f"    top score: {top_score:.3f}   false answer: {falsely_answered}")
        if i < len(unanswerable) - 1:
            time.sleep(SECONDS_BETWEEN_QUESTIONS)

    # ---- Report ----
    def hit_rate(ranks: list[int | None]) -> float:
        return sum(1 for r in ranks if r is not None) / len(ranks)

    def mrr(ranks: list[int | None]) -> float:
        return sum(1 / r for r in ranks if r is not None) / len(ranks)

    old_ranks = [r[1] for r in results]
    new_ranks = [r[2] for r in results]

    print("\n===== RETRIEVAL EVAL =====")
    print(f"{'answerable question':<50} {'old':>4} {'new':>4} {'score':>6}")
    for question, old_rank, new_rank, score in results:
        print(f"{question[:48]:<50} {str(old_rank):>4} {str(new_rank):>4} {score:>6.3f}")
    print("-" * 68)
    print(f"hit@{TOP_K}:  old {hit_rate(old_ranks):.0%}   new {hit_rate(new_ranks):.0%}")
    print(f"MRR:     old {mrr(old_ranks):.2f}   new {mrr(new_ranks):.2f}")

    print(f"\n{'unanswerable question':<50} {'score':>6} {'false?':>7}")
    for question, score, falsely in refusals:
        print(f"{question[:48]:<50} {score:>6.3f} {str(falsely):>7}")
    false_rate = sum(1 for r in refusals if r[2]) / len(refusals)
    print("-" * 68)
    print(f"false-answer rate @ threshold {settings.min_relevance}: {false_rate:.0%}  (want 0%)")

    # ---- Threshold suggestion: the gap between the two distributions ----
    lowest_answerable = min(r[3] for r in results)
    highest_unanswerable = max(r[1] for r in refusals)
    print(f"\nlowest answerable top-score:    {lowest_answerable:.3f}")
    print(f"highest unanswerable top-score: {highest_unanswerable:.3f}")
    if highest_unanswerable < lowest_answerable:
        midpoint = (lowest_answerable + highest_unanswerable) / 2
        print(f"clean separation — suggested min_relevance ≈ {midpoint:.2f}")
    else:
        print(
            "distributions OVERLAP — no threshold separates them cleanly; "
            "inspect the overlapping questions before choosing."
        )


if __name__ == "__main__":
    main()
