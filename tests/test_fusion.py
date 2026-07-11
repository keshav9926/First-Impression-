# tests/test_fusion.py — unit tests for Reciprocal Rank Fusion.
#
# TARGET: app/rag/fusion.py rrf() — pure function, like the chunker, so it
# tests without any network or store. Each test pins one property the /ask
# pipeline relies on.

from app.rag.fusion import K, rrf


def _hit(chunk_id: str) -> dict:
    return {"id": chunk_id, "text": f"text-{chunk_id}", "url": "https://x.com"}


def test_chunk_ranked_top_by_both_lists_wins():
    """A chunk that BOTH retrievers rank #1 must beat a chunk only one found."""
    shared, vector_only, keyword_only = _hit("both"), _hit("vec"), _hit("kw")
    fused = rrf(
        [shared, vector_only],  # vector ranking
        [shared, keyword_only],  # keyword ranking
        limit=3,
    )
    assert fused[0]["id"] == "both"  # two contributions beat one


def test_scores_accumulate_across_lists():
    """The winner's rrf_score must equal the sum of its per-list contributions."""
    shared = _hit("both")
    fused = rrf([shared], [shared], limit=1)
    assert fused[0]["rrf_score"] == 2 * (1.0 / (K + 1))  # rank 1 in both lists


def test_deduplicates_by_id():
    """A chunk found by both retrievers appears ONCE in the fused output."""
    shared = _hit("dup")
    fused = rrf([shared], [shared], limit=10)
    assert len(fused) == 1


def test_unique_finds_from_either_list_survive():
    """Hybrid's whole point: a chunk only ONE retriever found is still a candidate."""
    fused = rrf([_hit("vec")], [_hit("kw")], limit=10)
    assert {h["id"] for h in fused} == {"vec", "kw"}


def test_limit_caps_output():
    many = [_hit(f"c{i}") for i in range(10)]
    fused = rrf(many, [], limit=3)
    assert len(fused) == 3


def test_empty_lists_give_empty_output():
    assert rrf([], [], limit=5) == []


def test_guaranteed_seat_rescues_single_list_topper():
    """The 2026-07-12 eval finding, pinned as a test: a chunk ranked #1 by ONE
    list must reach the output even when consensus chunks fill the limit."""
    # 5 chunks that BOTH lists rank (mediocre consensus)...
    consensus = [_hit(f"c{i}") for i in range(5)]
    # ...and one chunk only the vector arm found — at rank 1.
    vec_top = _hit("vec-champion")

    without = rrf([vec_top] + consensus, consensus, limit=5)
    assert "vec-champion" not in {h["id"] for h in without}  # the observed bug

    with_seats = rrf([vec_top] + consensus, consensus, limit=5, guaranteed_per_list=3)
    assert "vec-champion" in {h["id"] for h in with_seats}  # the fix
