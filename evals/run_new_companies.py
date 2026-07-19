# evals/run_new_companies.py — run the REAL prod pipeline on fresh companies.
# No model overrides: ingest -> panel report on the configured pool (dspro-led).
# Sequential (the store holds one company at a time); each report saved before
# the next ingest wipes the store.
# Usage: python -m evals.run_new_companies

import json
import time
import traceback
from pathlib import Path

from app.agent.report import generate_report
from app.main import _ingest_site

COMPANIES = [
    ("telli", "https://telli.com"),
    ("collectwise", "https://collectwise.com"),
    ("sre_ai", "https://sre.ai"),
]
MAX_PAGES = 8
PANEL = True  # the flagship path: explore once -> 3 personas -> merged report

OUT_DIR = Path(
    r"C:\Users\KESHAV~1\AppData\Local\Temp\claude\c--Users-Keshav-Kakani-Desktop-PROJECTS-e2e-ai"
    r"\3c1fd7b3-9faa-4b5a-a1d1-04ee9dfaa44a\scratchpad\new_companies"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, url in COMPANIES:
        print(f"\n[new] ===== {key} ({url}) =====", flush=True)
        row = {"company": key, "url": url}
        t0 = time.time()
        try:
            ing = _ingest_site(url, MAX_PAGES)
            row["ingest"] = ing.model_dump()
            print(f"[new] ingested pages={ing.pages_fetched} chunks={ing.chunks_stored} "
                  f"thin={ing.extraction_warning}", flush=True)
            rep, steps_log, pages = generate_report(panel=PANEL)
            row.update(
                ok=True, wall_s=round(time.time() - t0, 1), steps=len(steps_log),
                pages_examined=pages,
                panel_verdicts={p.persona: p.would_sign_up for p in rep.persona_panel},
                report=rep.model_dump(),
            )
            print(f"[new] OK {row['wall_s']}s steps={row['steps']} "
                  f"verdicts={row['panel_verdicts']}", flush=True)
        except Exception as exc:
            row.update(ok=False, wall_s=round(time.time() - t0, 1),
                       error_type=type(exc).__name__, error=str(exc)[:400],
                       traceback=traceback.format_exc()[-1200:])
            print(f"[new] FAIL {row['error_type']}: {row['error'][:150]}", flush=True)
        (OUT_DIR / f"{key}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"\n[new] done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
