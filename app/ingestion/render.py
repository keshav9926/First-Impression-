# app/ingestion/render.py — headless-browser rendering for JS sites (Phase 6).
#
# WHY: fetcher's static httpx fetch sees only the HTML shell of a JS-rendered
# site (Framer/Webflow/Next). trynarrative.com → 368 chars, asha.health → 95.
# Playwright drives real Chromium, waits for the page's JS to hydrate, then
# hands back the FULLY-RENDERED HTML — the same DOM a human sees. Everything
# downstream (trafilatura, heading/CTA/link extraction) is unchanged; only the
# HTML source improves.
#
# USED ONLY AS A FALLBACK: fetcher.crawl() runs the cheap static path first and
# escalates here only when _is_thin_extraction trips — a browser is ~1000x
# heavier than an HTTP GET, so most (server-rendered) sites never touch this.
#
# CALL FLOW:
#   fetcher.crawl() → with browser_session() as b: render_html(b, url)

import logging
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger("first_impression")

# networkidle can hang on sites with long-poll/analytics sockets; cap hard.
_NAV_TIMEOUT_MS = 20_000
# Best-effort settle AFTER the DOM is ready: most JS content has hydrated by
# now; if analytics sockets keep the network busy we don't wait the full nav
# budget for an "idle" that never comes.
_IDLE_SETTLE_MS = 4_000


@contextmanager
def browser_session():
    """Launch ONE headless Chromium for a whole crawl (launch-per-page would be
    absurdly slow). Yields a browser; always closes it. Playwright imported
    lazily so the dependency is only needed when a JS site is actually hit."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def render_page(browser, url: str) -> tuple[str, str]:
    """Return (rendered_html, visible_text) for `url`, or ("", "") on failure.

    Two outputs on purpose:
      - html  → link / heading / CTA extraction (needs the tag structure)
      - text  → the page's VISIBLE text via inner_text("body")
    We use inner_text, NOT trafilatura, for the body: trafilatura's article
    heuristics collapse to near-nothing on component-soup JS sites (Framer:
    368 chars from a 3MB DOM), while inner_text returns what a human actually
    sees (1700+). The cost is some nav/footer noise in the text — acceptable on
    JS sites where the alternative is no content at all.

    Returns ("", "") (never raises) so one bad page skips like a static fetch
    error. Waits for network-idle so client-side content has hydrated first.
    """
    page = browser.new_page(user_agent=settings.crawler_user_agent)
    try:
        # domcontentloaded is fast + reliable; networkidle alone hangs the full
        # nav budget on pages with persistent analytics/long-poll sockets.
        page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
        try:
            page.wait_for_load_state("networkidle", timeout=_IDLE_SETTLE_MS)
        except Exception:
            pass  # never idle → take the DOM we have; better than nothing
        return page.content(), page.inner_text("body")
    except Exception as exc:
        logger.warning("render failed for %s: %s", url, exc)
        return "", ""
    finally:
        page.close()
