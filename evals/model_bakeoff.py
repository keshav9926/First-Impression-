# evals/model_bakeoff.py — bake off NVIDIA report LLMs head-to-head.
#
# Runs the WHOLE report pipeline (crawl -> ingest -> explore -> synthesize -> guards
# -> groundedness judge) once per (model, company), forcing every pooled LLM call
# onto a SINGLE model (no failover), so the result reflects that model alone.
#
# Design:
#   - Ingest each company ONCE (crawl + Voyage embed), then run every model over
#     the same stored evidence — fair, and it minimizes crawl/embed cost.
#   - Monkeypatch llm_pool.chat so explore/synthesis/judge all hit the one model.
#     A model that can't tool-call fails at explore; one that rejects json_object
#     fails at synthesis — both are recorded as findings, not crashes.
#   - Each run's full report + metrics are written IMMEDIATELY (checkpointed), so
#     a mid-run death still leaves every completed run on disk.
#
# Usage:
#   python -m evals.model_bakeoff                 # full matrix
#   python -m evals.model_bakeoff --models glm-5.2 --companies vortexify  # subset
#   python -m evals.model_bakeoff --validate      # 1 model x 1 company smoke run
#
# Output: <OUT_DIR>/<company>__<model>.json (per run) + summary.json + summary.csv

import argparse
import json
import time
import traceback
from pathlib import Path

import openai

from app.agent import report as report_mod
from app.config import settings
from app.main import _ingest_site

# ---- what to run -----------------------------------------------------------
MODELS = [
    "z-ai/glm-5.2",
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",           # no tool-calling — explore expected to fail
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "qwen/qwen3.5-397b-a17b",
    "moonshotai/kimi-k2.6",
    "minimaxai/minimax-m3",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "mistralai/mistral-medium-3.5-128b",
    "meta/llama-4-maverick-17b-128e-instruct",
    "openai/gpt-oss-120b",
]

# The 6 frontier models — the default screening pass (see --all for the full 12).
FRONTIER = [
    "z-ai/glm-5.2",
    "deepseek-ai/deepseek-v4-pro",           # no tool-calling — explore fails FAST (documents it)
    "nvidia/nemotron-3-ultra-550b-a55b",
    "qwen/qwen3.5-397b-a17b",
    "moonshotai/kimi-k2.6",
    "mistralai/mistral-large-3-675b-instruct-2512",
]

COMPANIES = [
    ("vortexify", "https://www.vortexify.ai"),
    ("asha", "https://asha.health"),
]

MAX_PAGES = 6            # crawl once per company; bounds the shared evidence
# Cap the explore loop for the bake-off ONLY (prod stays at settings' 7). The
# 47-min GLM validation showed cost is dominated by re-sending a growing history
# each turn on NVIDIA's slow free tier; 4 turns (list_pages + a couple reads +
# a search) still exercises explore->synthesize->judge while keeping runs sane.
BAKEOFF_MAX_STEPS = 4
PANEL = False           # single-agent path; half the calls of the panel
# Referee that scores every finished report (see score_reports). It is itself a
# contestant, so there is a mild self-preference caveat — noted in the summary.
# Chosen for being strong AND fast (186 t/s) so 12 scoring calls don't add hours.
REFEREE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
_SHIM_RETRIES = 2       # retry a transient 5xx/empty ONCE so a blip doesn't unfairly fail a model

OUT_DIR = Path(
    r"C:\Users\KESHAV~1\AppData\Local\Temp\claude\c--Users-Keshav-Kakani-Desktop-PROJECTS-e2e-ai"
    r"\3c1fd7b3-9faa-4b5a-a1d1-04ee9dfaa44a\scratchpad\bakeoff"
)


# ---- single-model override -------------------------------------------------
def install_single_model(model_id: str) -> dict:
    """Point llm_pool.chat at exactly `model_id` (no pool, no failover). Returns
    a per-run call counter the shim increments, so we can see how many LLM calls
    each report cost even though the pool's own tally is bypassed."""
    from app.agent import llm_pool

    counter = {"calls": 0}
    # Per-call timeout so a hung/degraded model fails in ~2 min instead of
    # blocking on the SDK's 10-min default (a runaway model would otherwise
    # exceed the bake-off's time gate only after a huge wall-clock).
    client = openai.OpenAI(
        base_url=_NVIDIA_BASE, api_key=settings.nvidia_api_key, timeout=120, max_retries=0
    )

    def chat(messages, prefer=None, label="llm-call", gemini_model=None, **kwargs):
        kwargs.pop("name", None)
        kwargs.pop("metadata", None)
        last = None
        for attempt in range(_SHIM_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=model_id, messages=messages, **kwargs
                )
                msg = resp.choices[0].message
                # blank-and-no-tool-calls -> transient; retry like the real pool
                if not (msg.content or "").strip() and not getattr(msg, "tool_calls", None):
                    last = RuntimeError("empty completion")
                    time.sleep(1.5)
                    continue
                counter["calls"] += 1
                return msg
            except (openai.InternalServerError, openai.APIConnectionError) as exc:
                last = exc
                time.sleep(1.5)
            # BadRequest / RateLimit / auth -> real, surface immediately
        counter["calls"] += 1
        raise last if last else RuntimeError("no response")

    llm_pool.chat = chat
    return counter


def _section_counts(rep) -> dict:
    return {
        "what_the_product_is": len(rep.what_the_product_is),
        "likely_new_user_journey": len(rep.likely_new_user_journey),
        "friction_points": len(rep.friction_points),
        "standout_strengths": len(rep.standout_strengths),
        "unanswered_questions": len(rep.unanswered_questions),
        "improvement_opportunities": len(rep.improvement_opportunities),
    }


def run_one(model_id: str, company: str) -> dict:
    """One (model, company) report. Store must already hold the company's chunks."""
    counter = install_single_model(model_id)
    row = {"model": model_id, "company": company}
    t0 = time.time()
    try:
        rep, steps_log, pages = report_mod.generate_report(panel=PANEL)
        row.update(
            ok=True,
            wall_s=round(time.time() - t0, 1),
            llm_calls=counter["calls"],
            steps=len(steps_log),
            pages_examined=len(pages),
            tool_calls=[s["tool"] for s in steps_log],
            sections=_section_counts(rep),
            total_observations=sum(_section_counts(rep).values()),
            company_name=rep.company,
            scope_note=rep.scope_note,
            report=rep.model_dump(),
        )
    except Exception as exc:  # any failure is a FINDING (no tools / no json / quota)
        row.update(
            ok=False,
            wall_s=round(time.time() - t0, 1),
            llm_calls=counter["calls"],
            error_type=type(exc).__name__,
            error=str(exc)[:500],
            traceback=traceback.format_exc()[-1500:],
        )
    return row


_RUBRIC = """\
You are a strict evaluator of a "first impression" analysis report about a
startup's public website. Score the report on five axes, integers 1-5 (5 best):
- grounding: claims are specific and evidence-backed, not generic filler
- completeness: covers what the product is, the new-user journey, friction, strengths
- citation_quality: every observation has a plausible, specific source_url
- tone: observational and founder-respectful (not harsh, not fawning)
- insight: friction/strengths are non-obvious, useful to a founder
Reply ONLY with JSON: {"grounding":int,"completeness":int,"citation_quality":int,
"tone":int,"insight":int,"overall":int,"one_line":"<=15 word verdict"}"""


def _loads_json_object(text: str) -> dict:
    """Parse a JSON object out of a referee reply, tolerating markdown fences and
    reasoning-model preamble (<think>… then the JSON). Strict parse first, then
    salvage the outermost {...} block."""
    try:
        return json.loads(text or "")
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def score_reports() -> None:
    """Referee pass: REFEREE_MODEL scores every successful report on disk and
    writes leaderboard.csv (mean overall per model). One bounded call per report."""
    client = openai.OpenAI(base_url=_NVIDIA_BASE, api_key=settings.nvidia_api_key)
    rows = []
    for f in sorted(OUT_DIR.glob("*__*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if not d.get("ok"):
            rows.append({**{k: d[k] for k in ("company", "model")}, "overall": 0,
                         "note": f"FAILED: {d.get('error_type')}"})
            continue
        rep = {k: d["report"][k] for k in (
            "what_the_product_is", "likely_new_user_journey", "friction_points",
            "standout_strengths", "unanswered_questions", "improvement_opportunities")}
        try:
            msg = client.chat.completions.create(
                model=REFEREE_MODEL,
                messages=[{"role": "system", "content": _RUBRIC},
                          {"role": "user", "content": json.dumps(rep)[:12000]}],
                response_format={"type": "json_object"},
                max_tokens=1200,  # reasoning models emit think-tokens before the JSON
            ).choices[0].message.content
            s = _loads_json_object(msg)
            rows.append({"company": d["company"], "model": d["model"],
                         **{k: s.get(k) for k in ("grounding", "completeness",
                            "citation_quality", "tone", "insight", "overall")},
                         "note": s.get("one_line", "")})
        except Exception as exc:
            rows.append({"company": d["company"], "model": d["model"], "overall": "",
                         "note": f"score error: {type(exc).__name__}"})
        print(f"[score] {d['company']} x {d['model']}: overall={rows[-1].get('overall')}", flush=True)

    cols = ["model", "company", "grounding", "completeness", "citation_quality",
            "tone", "insight", "overall", "note"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")).replace(",", ";") for c in cols))
    (OUT_DIR / "leaderboard.csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"[score] leaderboard.csv written (referee={REFEREE_MODEL}; note: referee "
          "is itself a contestant — mild self-preference caveat).", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="substring filters on model id")
    ap.add_argument("--companies", nargs="*", help="substring filters on company key")
    ap.add_argument("--all", action="store_true", help="full 12-model pool (default: 6 frontier)")
    ap.add_argument("--validate", action="store_true", help="1 model x 1 company smoke run")
    ap.add_argument("--score-only", action="store_true", help="just re-score reports already on disk")
    args = ap.parse_args()

    if args.score_only:
        score_reports()
        return

    # Bake-off-only explore cap (prod config untouched).
    settings.agent_max_steps = BAKEOFF_MAX_STEPS

    models = MODELS if args.all else FRONTIER
    companies = COMPANIES
    if args.validate:
        models, companies = [FRONTIER[0]], [COMPANIES[0]]
    if args.models:
        models = [m for m in MODELS if any(f in m for f in args.models)]
    if args.companies:
        companies = [(k, u) for k, u in COMPANIES if any(f in k for f in args.companies)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[bakeoff] {len(models)} models x {len(companies)} companies = "
          f"{len(models) * len(companies)} runs -> {OUT_DIR}", flush=True)

    results = []
    for ckey, curl in companies:
        print(f"\n[bakeoff] ingesting {ckey} ({curl}) ...", flush=True)
        try:
            ing = _ingest_site(curl, MAX_PAGES)
            print(f"[bakeoff]   pages={ing.pages_fetched} chunks={ing.chunks_stored} "
                  f"thin={ing.extraction_warning}", flush=True)
        except Exception as exc:
            print(f"[bakeoff]   INGEST FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        for model_id in models:
            slug = model_id.replace("/", "_")
            print(f"[bakeoff]   run {ckey} x {model_id} ...", end=" ", flush=True)
            row = run_one(model_id, ckey)
            (OUT_DIR / f"{ckey}__{slug}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            results.append(row)
            if row["ok"]:
                print(f"OK {row['wall_s']}s calls={row['llm_calls']} "
                      f"obs={row['total_observations']} steps={row['steps']}", flush=True)
            else:
                print(f"FAIL {row['error_type']} ({row['wall_s']}s)", flush=True)

    # summary
    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    cols = ["company", "model", "ok", "wall_s", "llm_calls", "steps",
            "pages_examined", "total_observations", "error_type"]
    lines = [",".join(cols)]
    for r in results:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    (OUT_DIR / "summary.csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[bakeoff] matrix done — {sum(r['ok'] for r in results)}/{len(results)} ok. "
          f"summary.csv written.", flush=True)

    if not args.validate:
        print("\n[bakeoff] scoring reports with referee ...", flush=True)
        score_reports()


if __name__ == "__main__":
    main()
