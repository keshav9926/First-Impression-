# evals/model_bakeoff2.py — round 2: 10 models x 3 FRESH companies.
#
# Goal: highest report ACCURACY under a HARD 10-minute-per-run budget. Any model
# that exceeds 600s is SCRAPPED (and skipped on its remaining companies — fail
# fast). GLM-5.2 is the accuracy REFEREE (not a contestant — it's the known-good
# reference). I pick the retrieval backend for speed+accuracy:
#   embed  = NVIDIA nemotron-3-embed-1b  (fast, MRR-tied with Voyage)
#   rerank = NVIDIA rerank-qa-mistral-4b (shifted-sigmoid gate) — removes Voyage's
#            3-req/min throttle so wall-time reflects the MODEL, not the reranker.
# Plus: explore capped at 5 steps and the new history-trim fix, so context stays
# small. Single-agent (panel off) — the unit under test is the core report.
#
# Usage:
#   python -m evals.model_bakeoff2              # full 10 x 3
#   python -m evals.model_bakeoff2 --validate   # 1 model x 1 company
#   python -m evals.model_bakeoff2 --score-only # re-score reports on disk

import argparse
import json
from pathlib import Path

import openai

from app.config import settings
from app.main import _ingest_site
from evals.model_bakeoff import _RUBRIC, _loads_json_object, run_one

MODELS = [
    "minimaxai/minimax-m3",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "moonshotai/kimi-k2.6",
    "mistralai/mistral-medium-3.5-128b",
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "mistralai/mistral-small-4-119b-2603",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3.5-122b-a10b",
    "openai/gpt-oss-120b",
]

# 3 fresh companies (NOT used in prior runs). B2B/dev sites likely to have real
# multi-page surfaces; the new shallow-crawl->render escalation covers SPAs.
COMPANIES = [
    ("credal", "https://credal.ai"),
    ("edgebit", "https://edgebit.io"),
    ("commonpaper", "https://commonpaper.com"),
]

REFEREE_MODEL = "z-ai/glm-5.2"   # accuracy reference, not a contestant
TIME_GATE_S = 600                # >10 min on a run -> scrap the model
MAX_PAGES = 8
BAKEOFF_STEPS = 5
_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"

OUT_DIR = Path(
    r"C:\Users\KESHAV~1\AppData\Local\Temp\claude\c--Users-Keshav-Kakani-Desktop-PROJECTS-e2e-ai"
    r"\3c1fd7b3-9faa-4b5a-a1d1-04ee9dfaa44a\scratchpad\bakeoff2"
)


def score_one(client, report: dict) -> dict:
    rep = {k: report[k] for k in (
        "what_the_product_is", "likely_new_user_journey", "friction_points",
        "standout_strengths", "unanswered_questions", "improvement_opportunities")}
    msg = client.chat.completions.create(
        model=REFEREE_MODEL,
        messages=[{"role": "system", "content": _RUBRIC},
                  {"role": "user", "content": json.dumps(rep)[:12000]}],
        response_format={"type": "json_object"}, max_tokens=1500,
    ).choices[0].message.content
    return _loads_json_object(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--models", nargs="*", help="substring filters — re-test only these")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Retrieval + loop config for THIS bake-off (prod config untouched at import;
    # we set the process here).
    settings.embed_provider = "nvidia"
    settings.rerank_provider = "nvidia"
    settings.agent_max_steps = BAKEOFF_STEPS
    ref_client = openai.OpenAI(base_url=_NVIDIA_BASE, api_key=settings.nvidia_api_key)

    if args.score_only:
        _leaderboard(ref_client)
        return

    models = [MODELS[0]] if args.validate else MODELS
    if args.models:
        models = [m for m in MODELS if any(f in m for f in args.models)]
    companies = [COMPANIES[0]] if args.validate else COMPANIES
    print(f"[bo2] {len(models)}x{len(companies)} | embed={settings.embed_provider} "
          f"rerank={settings.rerank_provider} steps={BAKEOFF_STEPS} gate={TIME_GATE_S}s", flush=True)

    scrapped: set[str] = set()
    for ckey, curl in companies:
        print(f"\n[bo2] ingesting {ckey} ({curl}) ...", flush=True)
        try:
            ing = _ingest_site(curl, MAX_PAGES)
            print(f"[bo2]   pages={ing.pages_fetched} chunks={ing.chunks_stored} "
                  f"thin={ing.extraction_warning}", flush=True)
        except Exception as exc:
            print(f"[bo2]   INGEST FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        for model_id in models:
            if model_id in scrapped:
                print(f"[bo2]   skip {model_id} (scrapped: over {TIME_GATE_S}s earlier)", flush=True)
                continue
            print(f"[bo2]   run {ckey} x {model_id} ...", end=" ", flush=True)
            row = run_one(model_id, ckey)  # single-agent (PANEL=False in model_bakeoff)
            over = row.get("wall_s", 0) > TIME_GATE_S
            row["scrapped"] = over
            # Skip a model on its REMAINING companies if it exceeded the gate OR
            # failed (a 400/404/timeout on company 1 recurs identically) — don't
            # burn time re-hitting a dead/degraded model on every company.
            if over or not row["ok"]:
                scrapped.add(model_id)
            (OUT_DIR / f"{ckey}__{model_id.replace('/', '_')}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8")
            tag = "SCRAP(>10m)" if over else ("OK" if row["ok"] else f"FAIL {row.get('error_type')}")
            print(f"{tag} {row.get('wall_s')}s obs={row.get('total_observations','-')}", flush=True)

    _leaderboard(ref_client)


def _leaderboard(ref_client) -> None:
    """Referee-score every OK, non-scrapped report; rank models by mean accuracy
    (averaged across companies), with mean wall-time and scrap status."""
    print("\n[bo2] scoring with referee GLM-5.2 ...", flush=True)
    from collections import defaultdict
    acc: dict = defaultdict(list)
    tim: dict = defaultdict(list)
    scrap: dict = defaultdict(bool)
    for f in sorted(OUT_DIR.glob("*__*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d["model"]
        if d.get("wall_s") is not None:
            tim[m].append(d["wall_s"])
        if d.get("scrapped"):
            scrap[m] = True
        if not d.get("ok") or d.get("scrapped"):
            continue
        try:
            s = score_one(ref_client, d["report"])
            acc[m].append(s.get("overall") or 0)
            print(f"[bo2]   {d['company']} x {m}: overall={s.get('overall')}", flush=True)
        except Exception as exc:
            print(f"[bo2]   score error {m}: {type(exc).__name__}", flush=True)

    rows = []
    for m in MODELS:
        scores = acc.get(m, [])
        rows.append({
            "model": m,
            "mean_accuracy": round(sum(scores) / len(scores), 2) if scores else 0,
            "n_scored": len(scores),
            "mean_wall_s": round(sum(tim[m]) / len(tim[m])) if tim.get(m) else None,
            "scrapped_over_10min": scrap.get(m, False),
        })
    rows.sort(key=lambda r: (not r["scrapped_over_10min"], r["mean_accuracy"]), reverse=True)

    lines = ["model,mean_accuracy,n_scored,mean_wall_s,scrapped_over_10min"]
    for r in rows:
        lines.append(f"{r['model']},{r['mean_accuracy']},{r['n_scored']},"
                     f"{r['mean_wall_s']},{r['scrapped_over_10min']}")
    (OUT_DIR / "leaderboard.csv").write_text("\n".join(lines), encoding="utf-8")
    print("\n===== ROUND 2 LEADERBOARD (accuracy, 10-min gate) =====", flush=True)
    print(f"{'model':46} {'acc':>4} {'n':>2} {'sec':>5}  scrapped", flush=True)
    for r in rows:
        print(f"{r['model']:46} {r['mean_accuracy']:>4} {r['n_scored']:>2} "
              f"{str(r['mean_wall_s']):>5}  {r['scrapped_over_10min']}", flush=True)


if __name__ == "__main__":
    main()
