# evals/embed_rerank_bakeoff.py — compare embedders x rerankers on retrieval quality.
#
# Isolates the two swappable retrieval components against the labeled vortexify
# eval set (evals/retrieval_eval.json), across the 2x2 matrix:
#     embedder  in {voyage voyage-3.5 (1024d), nvidia nemotron-3-embed-1b (2048d)}
#     reranker  in {voyage rerank-2.5-lite (0..1), nvidia rerank-qa-mistral-4b (logit)}
#
# HOW (no Chroma churn): the chunk TEXTS come from the store (a fuller vortexify
# re-ingest is forced first so all eval pages exist). Each embedder embeds those
# texts in-memory; vector search is numpy cosine. BM25 is shared (keyword.py).
# RRF fusion is the real fusion.rrf. Only the embed + rerank calls differ per
# combo, so hit@5 / MRR differences are attributable to those components alone.
#
# METRICS: hit@5 and MRR on 20 answerable questions per combo. For the 10
# unanswerable questions we record the top rerank score per RERANKER (the two
# rerankers live on different scales — Voyage 0..1 vs NVIDIA logit — so
# separation is reported per reranker, not compared across them).
#
# Usage:
#   python -m evals.embed_rerank_bakeoff            # full 2x2
#   python -m evals.embed_rerank_bakeoff --limit 3  # smoke: first 3 Q of each set
#
# Output: <OUT_DIR>/embed_rerank_summary.json + printed tables.

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import openai

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings  # noqa: E402
from app.main import _ingest_site  # noqa: E402
from app.rag import embeddings, fusion, keyword, rerank, store  # noqa: E402

EVAL_URL = "https://www.vortexify.ai"
EVAL_PAGES = 12          # enough to capture the eval set's ~10 labeled pages
TOP_K = 5
CANDIDATE_LIMIT = 10
_NVIDIA_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
_NVIDIA_RERANK_MODEL = "nvidia/rerank-qa-mistral-4b"
_NVIDIA_EMBED_BASE = "https://integrate.api.nvidia.com/v1"
_NVIDIA_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

OUT_DIR = Path(
    r"C:\Users\KESHAV~1\AppData\Local\Temp\claude\c--Users-Keshav-Kakani-Desktop-PROJECTS-e2e-ai"
    r"\3c1fd7b3-9faa-4b5a-a1d1-04ee9dfaa44a\scratchpad\bakeoff"
)

_nv_client = openai.OpenAI(base_url=_NVIDIA_EMBED_BASE, api_key=settings.nvidia_api_key)


# ---- embedders: text -> vectors -------------------------------------------
def voyage_embed(texts: list[str], kind: str) -> np.ndarray:
    """kind = 'document' | 'query'. Uses the app's Voyage client (paced)."""
    if kind == "query":
        return np.array([embeddings.embed_query(texts[0])], dtype=np.float32)
    return np.array(embeddings.embed_documents(texts), dtype=np.float32)


def nvidia_embed(texts: list[str], kind: str) -> np.ndarray:
    """kind -> NVIDIA input_type ('passage' for docs, 'query' for queries)."""
    input_type = "query" if kind == "query" else "passage"
    resp = _nv_client.embeddings.create(
        model=_NVIDIA_EMBED_MODEL, input=texts, encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


EMBEDDERS = {"voyage": voyage_embed, "nvidia": nvidia_embed}


# ---- rerankers: (query, candidates) -> candidates + 'relevance' -----------
def voyage_rerank(question: str, candidates: list[dict]) -> list[dict]:
    return rerank.rerank(question, candidates, top_k=TOP_K)


def nvidia_rerank(question: str, candidates: list[dict]) -> list[dict]:
    r = httpx.post(
        _NVIDIA_RERANK_URL,
        headers={"Authorization": f"Bearer {settings.nvidia_api_key}"},
        json={"model": _NVIDIA_RERANK_MODEL, "query": {"text": question},
              "passages": [{"text": c["text"]} for c in candidates]},
        timeout=40,
    )
    r.raise_for_status()
    rankings = r.json()["rankings"]  # [{index, logit}] best-first
    out = []
    for item in rankings[:TOP_K]:
        out.append({**candidates[item["index"]], "relevance": float(item["logit"])})
    return out


RERANKERS = {"voyage": voyage_rerank, "nvidia": nvidia_rerank}


def _cosine_topk(doc_mat: np.ndarray, q_vec: np.ndarray, chunks: list[dict], k: int) -> list[dict]:
    """In-memory dense search: cosine similarity, top-k hits (with id/url/text)."""
    dn = doc_mat / (np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-9)
    qn = q_vec[0] / (np.linalg.norm(q_vec[0]) + 1e-9)
    sims = dn @ qn
    order = np.argsort(-sims)[:k]
    return [chunks[i] for i in order]


def _first_correct_rank(hits: list[dict], expected: str) -> int | None:
    for pos, h in enumerate(hits):
        if expected in h["url"]:
            return pos + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap questions per set (smoke test)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads((Path(__file__).parent / "retrieval_eval.json").read_text(encoding="utf-8"))
    answerable = data["answerable"][: args.limit or None]
    unanswerable = data["unanswerable"][: args.limit or None]

    # Force a fuller re-ingest so every labeled page exists in the corpus.
    print(f"[er] re-ingesting {EVAL_URL} (max_pages={EVAL_PAGES}) ...", flush=True)
    ing = _ingest_site(EVAL_URL, EVAL_PAGES)
    print(f"[er]   pages={ing.pages_fetched} chunks={ing.chunks_stored}", flush=True)

    chunks = store.all_chunks()
    texts = [c["text"] for c in chunks]
    print(f"[er] corpus: {len(chunks)} chunks, {len({c['url'] for c in chunks})} pages", flush=True)

    # Embed the corpus once per embedder (cache the doc matrix).
    doc_mats = {}
    for name, fn in EMBEDDERS.items():
        print(f"[er] embedding corpus with {name} ...", flush=True)
        doc_mats[name] = fn(texts, "document")
        print(f"[er]   {name} dims={doc_mats[name].shape}", flush=True)

    # Per (embedder), cache the fused candidate list per question (needs a query
    # embed for that embedder), then rerank with BOTH rerankers.
    combos = {f"{e}+{r}": {"hits": [], "unans": []} for e in EMBEDDERS for r in RERANKERS}

    def fused_for(embedder: str, question: str) -> list[dict]:
        qv = EMBEDDERS[embedder]([question], "query")
        vector_hits = _cosine_topk(doc_mats[embedder], qv, chunks, 20)
        keyword_hits = keyword.search(question, top_k=20)
        return fusion.rrf(vector_hits, keyword_hits, limit=CANDIDATE_LIMIT, guaranteed_per_list=3)

    for phase, cases in (("answerable", answerable), ("unanswerable", unanswerable)):
        for i, case in enumerate(cases):
            q = case["question"]
            print(f"[er] {phase} {i+1}/{len(cases)}: {q[:55]}", flush=True)
            for e in EMBEDDERS:
                cands = fused_for(e, q)
                for r in RERANKERS:
                    ranked = RERANKERS[r](q, cands)
                    key = f"{e}+{r}"
                    if phase == "answerable":
                        combos[key]["hits"].append(
                            _first_correct_rank(ranked, case["expected_url_contains"]))
                    else:
                        combos[key]["unans"].append(ranked[0]["relevance"] if ranked else None)

    # ---- report ----
    def hit_rate(rs): return sum(1 for r in rs if r) / len(rs) if rs else 0.0
    def mrr(rs): return sum(1 / r for r in rs if r) / len(rs) if rs else 0.0

    print("\n===== EMBED x RERANK BAKE-OFF (vortexify) =====", flush=True)
    print(f"{'combo':18} {'hit@5':>6} {'MRR':>6}  {'ans_min':>8} {'unans_max':>9}  (reranker scale)")
    summary = {}
    for key, d in combos.items():
        h, m = hit_rate(d["hits"]), mrr(d["hits"])
        ans_scores = []  # top score on answerable is not tracked here; separation uses unans
        unans = [u for u in d["unans"] if u is not None]
        unans_max = max(unans) if unans else None
        summary[key] = {"hit@5": round(h, 3), "MRR": round(m, 3),
                        "unans_max_score": unans_max, "n_answerable": len(d["hits"])}
        print(f"{key:18} {h:>6.0%} {m:>6.2f}  {'-':>8} {str(round(unans_max,2) if unans_max is not None else '-'):>9}")

    (OUT_DIR / "embed_rerank_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[er] wrote {OUT_DIR / 'embed_rerank_summary.json'}", flush=True)
    print("[er] NOTE: hit@5/MRR are comparable across combos; unans_max is per-reranker "
          "scale (voyage 0..1 vs nvidia logit) — do NOT compare across rerankers.", flush=True)


if __name__ == "__main__":
    main()
