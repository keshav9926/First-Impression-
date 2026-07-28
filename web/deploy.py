"""
===============================================================================
FILE: web/deploy.py
ORIGIN      : CLI execution / Deployment script
PURPOSE     : Publishes static HTML report artifacts (web/dist/) to Netlify platform
DESTINATION : Netlify Production Edge CDN endpoints (Public share links)
===============================================================================
"""

import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DIST = Path(__file__).resolve().parent / "dist"


def _env(name: str) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    # fall back to .env (not exported into this process)
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _stage(use_slug: bool) -> tuple[Path, dict[str, str]]:
    """Build a clean-URL publish dir: each <company>.html becomes
    <path>/index.html so the live URL is /<company> (no '.html'). With --slug the
    path gets a random suffix (/<company>-9f3a2k) so links aren't guessable.
    Returns (staging_dir, {company: url_path})."""
    stage = Path(tempfile.mkdtemp(prefix="fie_pub_"))
    mapping = {}
    for html in sorted(DIST.glob("*.html")):
        stem = html.stem
        seg = f"{stem}-{secrets.token_hex(3)}" if use_slug else stem
        (stage / seg).mkdir(parents=True, exist_ok=True)
        shutil.copy(html, stage / seg / "index.html")
        mapping[stem] = seg
    return stage, mapping


def main() -> None:
    # accept either naming (NETLIFY_AUTH_TOKEN/SITE_ID or NETLIFY_TOKEN/PROJECT_ID)
    token = _env("NETLIFY_AUTH_TOKEN") or _env("NETLIFY_TOKEN")
    site = _env("NETLIFY_SITE_ID") or _env("NETLIFY_PROJECT_ID")
    if not token or not site:
        raise SystemExit(
            "Missing Netlify token/site in .env — set NETLIFY_TOKEN and "
            "NETLIFY_PROJECT_ID (see the one-time setup atop web/deploy.py)."
        )

    use_slug = "--slug" in sys.argv
    stage, names = _stage(use_slug)  # clean folder URLs: /<company> (no .html)

    # Windows ships npx as npx.cmd; subprocess (no shell) needs the resolved path.
    npx = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
    print("Deploying report pages -> Netlify ...", flush=True)
    proc = subprocess.run(
        [npx, "--yes", "netlify-cli", "deploy", "--prod", "--dir", str(stage),
         "--auth", token, "--site", site],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    shutil.rmtree(stage, ignore_errors=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"https://[^\s]+\.netlify\.app", out)
    base = m.group(0) if m else None
    if proc.returncode != 0 or not base:
        print(out)
        raise SystemExit(f"deploy failed (exit {proc.returncode})")

    print(f"\nLive base: {base}\n--- report links ---")
    for company, seg in sorted(names.items()):
        print(f"  {company:16} {base}/{seg}")


if __name__ == "__main__":
    main()
