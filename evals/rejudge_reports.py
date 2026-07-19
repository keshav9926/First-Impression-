# evals/rejudge_reports.py — re-apply the guard pass (citations → judge) to
# saved report JSONs, without re-running explore/synthesis.
#
# Exists because two blind spots were found AFTER the deep runs (2026-07-19):
# unjudged persona/question false negatives (SOC 2, trynarrative.com) and
# text-only image blindness ("no screenshots", vortexify.ai). The fixed judge
# needs the store to hold the SAME company (with the new image metadata), so
# each company is re-ingested (cheap) before its guards re-run.
# Usage: python -m evals.rejudge_reports

import json
from pathlib import Path

from app.agent import llm_pool
from app.agent.report import apply_guards
from app.main import _ingest_site
from app.schemas import FirstImpressionReport

REPORTS = Path(__file__).resolve().parent.parent / "reports"
COMPANIES = [
    ("vortexify", "https://www.vortexify.ai", 8),
    ("narrative", "https://www.trynarrative.com/", 8),
]


def main() -> None:
    for key, url, max_pages in COMPANIES:
        path = REPORTS / f"{key}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if not row.get("ok"):
            print(f"[rejudge] {key}: original run failed, skipping")
            continue
        print(f"[rejudge] {key}: re-ingesting {url}", flush=True)
        _ingest_site(url, max_pages)
        report = FirstImpressionReport.model_validate(row["report"])
        before = {
            "personas": sum(len(p.what_resonated) + len(p.friction) for p in report.persona_panel),
            "questions": len(report.unanswered_questions),
            "observations": sum(
                len(getattr(report, f))
                for f in ("what_the_product_is", "likely_new_user_journey",
                          "friction_points", "standout_strengths", "improvement_opportunities")
            ),
        }
        with llm_pool.use_mode("deep"):  # judge on the same quality-first chain
            report = apply_guards(report)
        after = {
            "personas": sum(len(p.what_resonated) + len(p.friction) for p in report.persona_panel),
            "questions": len(report.unanswered_questions),
            "observations": sum(
                len(getattr(report, f))
                for f in ("what_the_product_is", "likely_new_user_journey",
                          "friction_points", "standout_strengths", "improvement_opportunities")
            ),
        }
        row["report"] = report.model_dump()
        row["rejudged"] = {"before": before, "after": after}
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"[rejudge] {key}: {before} -> {after}", flush=True)
    print("[rejudge] done", flush=True)


if __name__ == "__main__":
    main()
