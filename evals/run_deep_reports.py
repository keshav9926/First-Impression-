# evals/run_deep_reports.py — real deliverable runs: deep mode (GLM-5.2-led
# chain) + persona panel on outreach targets. Output feeds the UI template and
# the outreach Excel, so these land in the repo's reports/ dir, not scratch.
# Sequential (the store holds one company at a time).
# Usage: python -m evals.run_deep_reports

import json
import time
import traceback
from pathlib import Path

from app.agent.report import generate_report
from app.main import _ingest_site

COMPANIES = [
    ("narrative", "https://www.trynarrative.com/"),
    ("vortexify", "https://www.vortexify.ai"),
]
MAX_PAGES = 8
PANEL = True     # explore once -> 3 personas -> merged report
MODE = "deep"    # glm -> dspro -> dsflash -> nemo (quality-first, no time cap)

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, url in COMPANIES:
        print(f"\n[deep] ===== {key} ({url}) =====", flush=True)
        row = {"company": key, "url": url, "mode": MODE}
        t0 = time.time()
        try:
            ing = _ingest_site(url, MAX_PAGES)
            row["ingest"] = ing.model_dump()
            print(f"[deep] ingested pages={ing.pages_fetched} chunks={ing.chunks_stored} "
                  f"thin={ing.extraction_warning}", flush=True)
            rep, steps_log, pages = generate_report(panel=PANEL, mode=MODE)
            row.update(
                ok=True, wall_s=round(time.time() - t0, 1), steps=len(steps_log),
                pages_examined=pages,
                panel_verdicts={p.persona: p.would_sign_up for p in rep.persona_panel},
                report=rep.model_dump(),
            )
            print(f"[deep] OK {row['wall_s']}s steps={row['steps']} "
                  f"verdicts={row['panel_verdicts']}", flush=True)
        except Exception as exc:
            row.update(ok=False, wall_s=round(time.time() - t0, 1),
                       error_type=type(exc).__name__, error=str(exc)[:400],
                       traceback=traceback.format_exc()[-1200:])
            print(f"[deep] FAIL {row['error_type']}: {row['error'][:150]}", flush=True)
        (OUT_DIR / f"{key}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"\n[deep] done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
