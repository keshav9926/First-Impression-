"""
===============================================================================
FILE: app/ingestion/render.py
ORIGIN      : app.ingestion.fetcher (when dynamic JS rendering is forced/needed)
PURPOSE     : Headless Chromium execution via Playwright for SPA JS rendering & DOM extraction
DESTINATION : app.ingestion.fetcher (Returns rendered HTML and visible DOM text)
===============================================================================
"""

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


def render_page(browser, url: str) -> tuple[str, str, str]:
    """Return (rendered_html, visible_text, final_url), or ("", "", "") on
    failure or any HTTP error status.

    Three outputs on purpose:
      - html  → link / heading / CTA extraction (needs the tag structure)
      - text  → the page's VISIBLE text via inner_text("body")
      - final_url → where we actually landed, after redirects
    We use inner_text, NOT trafilatura, for the body: trafilatura's article
    heuristics collapse to near-nothing on component-soup JS sites (Framer:
    368 chars from a 3MB DOM), while inner_text returns what a human actually
    sees (1700+). The cost is some nav/footer noise in the text — acceptable on
    JS sites where the alternative is no content at all.

    Returns ("", "", "") (never raises) so one bad page skips like a static
    fetch error. Waits for network-idle so client-side content has hydrated.
    """
    page = browser.new_page(user_agent=settings.crawler_user_agent)
    try:
        # domcontentloaded is fast + reliable; networkidle alone hangs the full
        # nav budget on pages with persistent analytics/long-poll sockets.
        response = page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
        # The static path rejects non-200s; this one did not, so a 404's error
        # page ("HTTP Status: 404 (not found)") was non-empty text and got
        # chunked, embedded and stored as if it were content. On vortexify.ai
        # that was 14 of 15 "pages" — 78% of the corpus (caught 2026-08-01).
        if response is not None and response.status >= 400:
            logger.info("render: %s returned HTTP %d — skipping", url, response.status)
            return "", "", ""
        try:
            page.wait_for_load_state("networkidle", timeout=_IDLE_SETTLE_MS)
        except Exception:
            pass  # never idle → take the DOM we have; better than nothing
        # Scroll top-to-bottom to trigger lazy-loaded images/dashboards (many
        # product sites defer screenshots until they scroll into view). Without
        # this the vision captioner never sees them. Best-effort; never fatal.
        try:
            page.evaluate("""async () => {
              await new Promise(res => {
                let y = 0;
                const step = () => {
                  window.scrollTo(0, y); y += Math.round(window.innerHeight * 0.9);
                  if (y < document.body.scrollHeight) setTimeout(step, 140);
                  else { window.scrollTo(0, 0); setTimeout(res, 500); }
                };
                step();
              });
            }""")
            page.wait_for_timeout(600)  # let freshly-triggered images finish loading
        except Exception:
            pass
        # page.url is the POST-redirect address — the caller resolves this
        # page's links against it, so a redirect can't queue dead URLs.
        return page.content(), page.inner_text("body"), page.url
    except Exception as exc:
        logger.warning("render failed for %s: %s", url, exc)
        return "", "", ""
    finally:
        page.close()
