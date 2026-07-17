# app/rag/pipeline.py — the shared hybrid retrieval funnel, in one place.
#
# WHY THIS EXISTS: the Phase 2 funnel (embed → vector + BM25 → RRF → rerank)
# was written inline inside main.ask(). Phase 3's agent needs the EXACT same
# retrieval for its search_content tool. Rather than copy-paste it (and risk
# the two drifting apart), both callers now go through retrieve() here.
#
# CALL FLOW:
#   main.py ask()                 → retrieve(question, top_k)
#   agent/tools.py search_content → retrieve(question, top_k)
#
# retrieve() returns ranked hits WITH their "relevance" score but does NOT
# apply the min_relevance threshold — each caller decides what to do with a
# weak result (ask() refuses; the tool tells the agent "nothing found").

from app import observability
from app.rag import embeddings, fusion, keyword, rerank, store

CANDIDATES_PER_RETRIEVER = 20  # each retriever's contribution to fusion
CANDIDATES_TO_RERANK = 10  # fused list size sent to the (slower) re-ranker
GUARANTEED_PER_LIST = 3  # each arm's top-N always reach the reranker (see fusion.py)


def retrieve(question: str, top_k: int) -> list[dict]:
    """Run the full hybrid funnel; return top_k reranked hits, best first.

    Each hit: {"id", "text", "url", "distance", "rrf_score", "relevance"}.
    May raise voyageai.error.* — callers map those to HTTP status codes.
    """
    # Trace the hybrid funnel as a `retriever` observation (input = the query,
    # output = the ranked hits) so grounding is visible in the trace. No-op
    # unless tracing is on.
    with observability.span("retrieve-context", as_type="retriever", input=question) as obs:
        query_vector = embeddings.embed_query(question)
        vector_hits = store.search(query_vector, top_k=CANDIDATES_PER_RETRIEVER)
        keyword_hits = keyword.search(question, top_k=CANDIDATES_PER_RETRIEVER)
        candidates = fusion.rrf(
            vector_hits,
            keyword_hits,
            limit=CANDIDATES_TO_RERANK,
            guaranteed_per_list=GUARANTEED_PER_LIST,
        )
        hits = rerank.rerank(question, candidates, top_k=top_k)
        if obs:
            obs.update(
                output=[{"url": h.get("url"), "relevance": h.get("relevance")} for h in hits]
            )
        return hits
