# evals/debug_retrieval.py — "why did retrieval rank THAT?" diagnostic.
# Run:  uv run python evals/debug_retrieval.py "your question here" [more...]
#
# For each question it prints the whole funnel with provenance:
#   which rank each candidate got from the VECTOR arm and the BM25 arm,
#   its fused RRF position, and its final rerank relevance score.
# Use it whenever the eval flags a miss — it shows WHERE the wrong chunk
# won (retriever, fusion, or reranker) instead of leaving you guessing.
#
# Free-tier pacing: 2 Voyage calls per question (embed + rerank), so the
# script sleeps between questions when given several.

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252

from app.rag import embeddings, fusion, keyword, rerank, store  # noqa: E402

SECONDS_BETWEEN_QUESTIONS = 45


def debug(question: str) -> None:
    print(f"\n{'=' * 78}\nQUESTION: {question}\n{'=' * 78}")

    query_vector = embeddings.embed_query(question)
    vector_hits = store.search(query_vector, top_k=20)
    keyword_hits = keyword.search(question, top_k=20)

    vector_rank = {h["id"]: i + 1 for i, h in enumerate(vector_hits)}
    keyword_rank = {h["id"]: i + 1 for i, h in enumerate(keyword_hits)}

    candidates = fusion.rrf(vector_hits, keyword_hits, limit=10, guaranteed_per_list=3)
    ranked = rerank.rerank(question, candidates, top_k=10)  # score ALL candidates

    print(f"{'#':>2} {'rerank':>7} {'vec':>4} {'bm25':>4}  {'page':<38} snippet")
    for i, hit in enumerate(ranked):
        page = hit["url"].replace("https://www.", "")[:38]
        snippet = hit["text"][:70].replace("\n", " ")
        v = vector_rank.get(hit["id"], "-")
        b = keyword_rank.get(hit["id"], "-")
        print(f"{i + 1:>2} {hit['relevance']:>7.3f} {str(v):>4} {str(b):>4}  {page:<38} {snippet}")


def main() -> None:
    questions = sys.argv[1:]
    if not questions:
        sys.exit('usage: uv run python evals/debug_retrieval.py "question" ["another"...]')
    if store.count() == 0:
        sys.exit("Store is empty — run /ingest first.")

    for i, question in enumerate(questions):
        debug(question)
        if i < len(questions) - 1:
            time.sleep(SECONDS_BETWEEN_QUESTIONS)


if __name__ == "__main__":
    main()
