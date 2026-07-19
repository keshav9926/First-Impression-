# web/render_report.py — turn a verified report JSON (reports/<key>.json) into
# a standalone, shareable HTML page in the exact web/report.html design.
#
# HOW THE PIECES CONNECT:
#   pipeline (deep run) -> reports/<key>.json      (FirstImpressionReport + meta)
#   web/report.html     -> the design TEMPLATE     (all data read from `var REPORT`)
#   this script         -> replaces the demo REPORT object with the real one
#                          and writes web/dist/<key>.html
#   hosting (GitHub Pages / Vercel / Netlify)      -> each file becomes a clean
#                          link like https://<user>.github.io/fie/vortexify
#
# The founder receives a LINK, never a file. One static page per company,
# zero backend at view time.
# Usage: python -m web.render_report            (renders every reports/*.json)
#        python -m web.render_report vortexify  (just one)

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "web" / "report.html"
REPORTS = ROOT / "reports"
DIST = ROOT / "web" / "dist"

# Persona name -> template icon key
_ROLE = {"business buyer": "briefcase", "technical evaluator": "code",
         "first-time end user": "person"}


def _score(rep: dict) -> tuple[int, str]:
    """Overall score derived from REAL signals, never invented:
    base 50 + panel verdicts (30) + strengths-vs-friction balance (20)."""
    panel = rep.get("persona_panel", [])
    yes = sum(1 for p in panel if p.get("would_sign_up"))
    verdict_part = 30 * (yes / len(panel)) if panel else 15
    s, f = len(rep.get("standout_strengths", [])), len(rep.get("friction_points", []))
    balance_part = 20 * (s / (s + f)) if (s + f) else 10
    score = round(50 + verdict_part + balance_part)
    potential = ("Strong potential" if score >= 75 else
                 "Promising" if score >= 60 else "Early signals")
    return score, potential


def _persona_strength(p: dict) -> float:
    """0-10 from real signals: signup verdict + resonated/friction balance."""
    r, f = len(p.get("what_resonated", [])), len(p.get("friction", []))
    base = 6.0 if p.get("would_sign_up") else 3.5
    balance = 3.0 * (r / (r + f)) if (r + f) else 1.5
    return round(min(9.9, base + balance), 1)


def _claims(items: list, limit: int) -> list[str]:
    return [o["claim"] for o in items[:limit]]


def _sentences(text: str, n: int) -> str:
    """First n whole sentences — used where a field must be shortened without
    ever cutting mid-word (the page never shows a chopped 'self' or 'secti')."""
    text = (text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n]).strip()


def build_report_obj(key: str, row: dict) -> dict:
    rep = row["report"]
    pages = row.get("pages_examined", [])
    ingest = row.get("ingest", {})
    score, potential = _score(rep)
    domain = re.sub(r"https?://(www\.)?", "", row["url"]).strip("/")
    gen = datetime.now().strftime("%b %d, %Y  •  %I:%M %p").upper()

    personas = []
    for p in rep.get("persona_panel", []):
        strength = _persona_strength(p)
        friction_lead = (p.get("friction") or [""])[0]
        resonated_lead = (p.get("what_resonated") or [""])[0]
        # full sentence(s), never a mid-word cut
        text = p.get("reason") or resonated_lead or friction_lead
        personas.append({
            "role": _ROLE.get(p["persona"].lower(), "person"),
            "name": p["persona"],
            "strength": strength,
            "dots": max(1, min(7, round(strength / 10 * 7))),
            "text": text.strip(),
        })

    strengths = _claims(rep.get("standout_strengths", []), 5)
    risks = _claims(rep.get("friction_points", []), 5)
    improvements = [
        {"text": (s.get("suggestion") or "").strip(),
         "impact": "High impact" if i < 2 else "Medium impact",
         "dots": 5 - min(i, 3)}
        for i, s in enumerate(rep.get("improvement_opportunities", [])[:5])
    ]
    questions = rep.get("unanswered_questions", [])[:6]

    # verdict headline: strongest strength theme vs strongest friction theme
    yes = sum(1 for p in rep.get("persona_panel", []) if p.get("would_sign_up"))
    total = len(rep.get("persona_panel", [])) or 1
    line1 = "Strong signal." if yes == total else "Real substance."
    line2 = "Room to open up." if risks else "Keep going."

    crawled = [[re.sub(r"https?://[^/]+", "", u) or "/",
                re.sub(r"https?://[^/]+", "", u) or "/"] for u in pages]
    crawled = [[(p[0].strip("/").split("/")[-1] or "Homepage").replace("-", " ").title(), p[1]]
               for p in crawled]

    chunks = ingest.get("chunks_stored", 0)
    n_pages = len(pages)
    coverage_pct = min(98, round(100 * n_pages / max(n_pages, ingest.get("pages_fetched", n_pages) or 1)))

    n_claims = sum(len(rep.get(f, [])) for f in
                   ("what_the_product_is", "likely_new_user_journey",
                    "friction_points", "standout_strengths"))

    cited: dict[str, int] = {}
    for f in ("what_the_product_is", "likely_new_user_journey",
              "friction_points", "standout_strengths"):
        for o in rep.get(f, []):
            cited[o["source_url"]] = cited.get(o["source_url"], 0) + 1
    cited_rows = sorted(cited.items(), key=lambda kv: -kv[1])[:5]

    return {
        "company": rep.get("company", key.title()),
        "subject": domain,
        "generated": gen,
        "reportId": f"FIE-{datetime.now().strftime('%d%m%Y')}-{key[:4].upper()}",
        "build": datetime.now().strftime("%d%m.%y"),
        "modelStack": "GLM-5.2 | DeepSeek-V4 | Nemotron | NV-Embed",
        "score": score,
        "potential": potential,
        "stats": {"pages": n_pages, "personas": len(personas) or 3,
                  "mode": row.get("mode", "deep").upper(),
                  "models": "GLM-5.2 (deep)", "progress": "100%"},
        "personas": personas,
        "execSummary": " ".join(
            _claims(rep.get("what_the_product_is", []), 1)
            + _claims(rep.get("standout_strengths", []), 1)
            + _claims(rep.get("friction_points", []), 1)
        ),
        "strengths": strengths,
        "risks": risks,
        "improvements": improvements,
        "questions": questions,
        "verdict": {
            "line1": line1, "line2": line2,
            "sub": _sentences(rep.get("scope_note", ""), 2),
            "rec": [{"k": "Impact", "v": "High", "dots": 4},
                    {"k": "Effort", "v": "Low-Med", "dots": 3},
                    {"k": "Personas positive", "v": f"{yes}/{total}", "dots": round(6 * yes / total)}],
        },
        "coverage": {"pct": coverage_pct, "pages": n_pages,
                     "sources": len(cited), "chunks": chunks,
                     "citations": str(n_claims)},
        "crawledPages": crawled[:9],
        "citedSources": [[u, n] for u, n in cited_rows],
    }


def render(key: str) -> Path:
    row = json.loads((REPORTS / f"{key}.json").read_text(encoding="utf-8"))
    if not row.get("ok"):
        raise SystemExit(f"{key}: report run failed, nothing to render")
    template = TEMPLATE.read_text(encoding="utf-8")
    obj = build_report_obj(key, row)
    blob = json.dumps(obj, indent=2, ensure_ascii=False)
    # Replace the whole demo REPORT literal: from `var REPORT = {` to the
    # closing `};` before the RENDER banner.
    out, n = re.subn(r"var REPORT = \{.*?\n  \};",
                     f"var REPORT = {blob};", template, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("template REPORT block not found — check web/report.html")
    DIST.mkdir(parents=True, exist_ok=True)
    dest = DIST / f"{key}.html"
    dest.write_text(out, encoding="utf-8")
    return dest


def main() -> None:
    keys = sys.argv[1:] or [p.stem for p in REPORTS.glob("*.json")]
    for key in keys:
        print("rendered ->", render(key))


if __name__ == "__main__":
    main()
