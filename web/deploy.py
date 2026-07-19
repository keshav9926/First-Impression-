# web/deploy.py — publish the rendered report pages to a real, non-claude link.
#
# One-time setup (you, once):
#   1. Create a free Netlify account: https://app.netlify.com
#   2. User settings → Applications → Personal access tokens → New token.
#      Put it in .env:              NETLIFY_AUTH_TOKEN=nfp_xxx
#   3. Add a site (Add new site → any name, or drag web/dist once). Open the
#      site → Site configuration → copy the "Site ID", put it in .env:
#                                   NETLIFY_SITE_ID=xxxxxxxx-....
#
# Then, any time — I run:
#   python -m web.deploy                 # deploy every web/dist/*.html
# and it prints the public link for each report. Rendered pages are static, so
# each link just serves that founder's report; nothing is indexed.
#
# Cross-company privacy: reports share one site, so links look like
# <site>.netlify.app/<company>.html. Pass --slug to add a random suffix
# (<company>-9f3a2k.html) so one founder's link can't be used to guess another's.

import os
import re
import secrets
import shutil
import subprocess
import sys
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


def _slugify_dist() -> dict[str, str]:
    """Copy each <company>.html to <company>-<rand>.html so the deployed path is
    unguessable. Returns {company: deployed_filename}."""
    mapping = {}
    for html in sorted(DIST.glob("*.html")):
        if re.search(r"-[0-9a-f]{6}\.html$", html.name):
            continue  # already slugged
        slug = f"{html.stem}-{secrets.token_hex(3)}.html"
        (DIST / slug).write_bytes(html.read_bytes())
        mapping[html.stem] = slug
    return mapping


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
    names = _slugify_dist() if use_slug else {
        p.stem: p.name for p in DIST.glob("*.html")
    }

    # Windows ships npx as npx.cmd; subprocess (no shell) needs the resolved path.
    npx = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
    print("Deploying web/dist → Netlify ...", flush=True)
    proc = subprocess.run(
        [npx, "--yes", "netlify-cli", "deploy", "--prod", "--dir", str(DIST),
         "--auth", token, "--site", site],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"https://[^\s]+\.netlify\.app", out)
    base = m.group(0) if m else None
    if proc.returncode != 0 or not base:
        print(out)
        raise SystemExit(f"deploy failed (exit {proc.returncode})")

    print(f"\nLive base: {base}\n--- report links ---")
    for company, filename in sorted(names.items()):
        print(f"  {company:16} {base}/{filename}")


if __name__ == "__main__":
    main()
