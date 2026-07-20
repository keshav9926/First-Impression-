# evals/run_report.py — run the pipeline on ONE company, any mode.
# Generalizes run_deep_reports so new outreach targets don't need a code edit.
#
# Usage:
#   python -m evals.run_report <name> <url> [normal|deep] [max_pages]
#   python -m evals.run_report unitedtechlab https://unitedtechlab.com/ normal

import json
import sys
import time
import traceback
from pathlib import Path

from app.agent.report import generate_report
from app.main import _ingest_site

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"


def run(name: str, url: str, mode: str = "normal", max_pages: int = 100) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[{mode}] ===== {name} ({url}) =====", flush=True)
    row = {"company": name, "url": url, "mode": mode}
    t0 = time.time()
    try:
        ing = _ingest_site(url, max_pages)
        row["ingest"] = ing.model_dump()
        print(f"[{mode}] ingested pages={ing.pages_fetched} chunks={ing.chunks_stored} "
              f"thin={ing.extraction_warning}", flush=True)
        rep, steps_log, pages = generate_report(panel=True, mode=mode)
        row.update(
            ok=True, wall_s=round(time.time() - t0, 1), steps=len(steps_log),
            pages_examined=pages,
            panel_verdicts={p.persona: p.would_sign_up for p in rep.persona_panel},
            report=rep.model_dump(),
        )
        print(f"[{mode}] OK {row['wall_s']}s steps={row['steps']} "
              f"verdicts={row['panel_verdicts']}", flush=True)
    except Exception as exc:
        row.update(ok=False, wall_s=round(time.time() - t0, 1),
                   error_type=type(exc).__name__, error=str(exc)[:400],
                   traceback=traceback.format_exc()[-1200:])
        print(f"[{mode}] FAIL {row['error_type']}: {row['error'][:150]}", flush=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"[{mode}] saved -> {OUT_DIR / (name + '.json')}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m evals.run_report <name> <url> [normal|deep] [max_pages]")
    name, url = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "normal"
    max_pages = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    run(name, url, mode, max_pages)
