# app/ingestion/fetcher.py — polite, same-domain crawler.
# Enforces the "public data only" rule in code, not in prompts:
#   1. every URL is checked against robots.txt BEFORE fetching (robots.py)
#   2. a fixed delay between requests (rate limiting — never hammer a site)
#   3. never leaves the starting domain, never follows login/signup paths
# Page HTML is reduced to readable article text with trafilatura (nav bars,
# cookie banners, footers stripped) — clean text in = good retrieval later.
#
# CALL FLOW (who calls what):
#
#   main.py: ingest()                      ← the /ingest endpoint
#       └── crawl(start_url, max_pages)    ← entry point of THIS file
#             ├── robots.is_allowed(url)   ← permission check (robots.py)
#             ├── httpx client.get(url)    ← download the HTML
#             ├── trafilatura.extract()    ← HTML → readable text
#             └── _extract_links()         ← find next URLs to visit
#                   └── _LinkCollector     ← pulls every <a href> out of the HTML
#
#   crawl() returns a CrawlResult back to main.py, which then passes each
#   page's text to chunker.chunk_text().

import logging
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import trafilatura

from app import events
from app.config import settings
from app.ingestion import render
from app.ingestion.robots import is_allowed

logger = logging.getLogger("first_impression")

# Paths that hint at authenticated / non-public areas — rule #1 says we never try these.
BLOCKED_PATH_HINTS = ("login", "signin", "sign-in", "signup", "sign-up", "account", "logout")

# File types we can't turn into text (PDF support could come later).
SKIP_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".mp4", ".css", ".js", ".xml", ".json",
)


@dataclass
class Page:
    """One successfully fetched page: its URL + the readable text we extracted
    + its section headings (h1-h3), in document order.

    Produced by: crawl().
    Consumed by: main.py ingest(), which sends .text to chunker.chunk_text()
    and keeps .url so every chunk can cite the page it came from. .headings
    becomes chunk metadata → the agent's read_page shows it as a section map,
    so a truncated long page still reveals WHAT EXISTS deeper (the agent can
    then search_content into any section it never saw).
    """

    url: str
    text: str
    headings: list[str]
    ctas: list[str]  # primary call-to-action button/link labels (Sign up, Try free, ...)


# Detecting a JS-rendered site (SPA / Framer / Webflow) where the static HTML
# is a shell and the real content hydrates in the browser (a static fetch never
# sees it). FAIL-SAFE by design (2026-07-18): the earlier text-AND-ratio rule
# was tuned on only 2 sites and its dangerous failure mode was a FALSE NEGATIVE
# — a partly-rendered SPA slipping through, so the report confidently claims
# "the site doesn't mention X" about content we simply never read. The ratio
# signal was the fragile half (modern HTML is bloated even when server-rendered),
# so it's dropped. We now escalate on the single robust signal — the seed page
# came back with very little text — because escalation is CHEAP and SAFE: it
# just tries a headless render and keeps the static result if that doesn't help.
# Over-escalating a small-but-fine page costs one browser spin-up; under-
# escalating a real SPA silently corrupts the report. We bias toward the former.
# The same function also flags the POST-render result: a genuinely content-rich
# rendered page has far more than this, so it won't be caveated.
_THIN_SEED_MAX_TEXT_CHARS = 1200  # seed text below this → treat as thin, try rendering


def _is_thin_extraction(seed_text_chars: int, seed_html_chars: int) -> bool:
    """True when the SEED page came back with too little text to trust — the
    robust JS-shell signal. Fail-safe: err toward True (escalate to render),
    since render falls back to the static result if it doesn't help.

    Called by: crawl() (to decide whether to escalate to headless) and
    _crawl_loop (to flag the final result). The seed (usually the homepage) is
    the right page to judge on: it carries the first impression, and it's exactly
    the page that hydrates client-side on JS sites."""
    if seed_html_chars == 0:
        return False  # nothing fetched at all — not a "thin" signal, a dead page
    return seed_text_chars < _THIN_SEED_MAX_TEXT_CHARS


@dataclass
class CrawlResult:
    """What crawl() hands back to the /ingest endpoint:
    the list of Pages plus a count of URLs robots.txt made us skip
    (reported to the user in the API response).

    extraction_ratio: total readable text ÷ total raw HTML across all fetched
    pages. thin_extraction: True when that ratio is under THIN_EXTRACTION_RATIO
    — the signal that this analysis is built on a FRACTION of the real site,
    so "the site doesn't mention X" claims are unsafe (our blindness, not
    their gap). Rides through chunk metadata to the agent + final report."""

    pages: list[Page]
    skipped_by_robots: int
    extraction_ratio: float = 1.0
    thin_extraction: bool = False


class _LinkCollector(HTMLParser):
    """Tiny HTML parser whose only job: collect every <a href="..."> on a page.

    Called by: _extract_links() — it feeds the raw HTML in, then reads .hrefs out.
    How it works: HTMLParser walks the HTML tag by tag and calls
    handle_starttag() for each opening tag; we grab the href when tag == "a".
    (The leading underscore in the name = "private, only used inside this file".)
    """

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.hrefs.append(value)


class _HeadingCollector(HTMLParser):
    """Pulls the text of every <h1>/<h2>/<h3> out of a page — the page's own
    table of contents. Same HTMLParser pattern as _LinkCollector above.

    Called by: _extract_headings(). Trafilatura's plain-text extraction
    FLATTENS headings into ordinary lines (structure lost), so we parse them
    from the raw HTML separately — additive, the text pipeline is untouched.
    """

    def __init__(self) -> None:
        super().__init__()
        self.headings: list[str] = []
        self._inside: str | None = None  # the heading tag we're currently in
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("h1", "h2", "h3"):
            self._inside = tag
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._inside == tag:
            text = " ".join("".join(self._buffer).split())  # collapse whitespace
            if text:
                self.headings.append(text)
            self._inside = None

    def handle_data(self, data: str) -> None:
        if self._inside:
            self._buffer.append(data)


# Bounds so a pathological page can't bloat the metadata: plenty for a real
# docs ToC, tiny in tokens either way.
_MAX_HEADINGS = 40
_MAX_HEADING_CHARS = 80


# The single most important signal for the "can I get started?" persona lives
# in the header/footer — "Try for free", "Sign up", "Book a demo" — which
# trafilatura's favor_precision=True strips as boilerplate (deliberately, to
# keep nav-debris out of RAG chunks). So we recover JUST these high-signal
# calls-to-action from the raw HTML, separately, without re-polluting retrieval.
# Match on visible LABEL text (a signup button rarely lies about being one).
_CTA_PATTERNS = (
    "sign up", "signup", "sign in", "signin", "log in", "login",
    "get started", "try for free", "try free", "start free", "free trial",
    "book a demo", "request a demo", "get a demo", "request access",
    "get started for free", "start now", "join",
)
_MAX_CTAS = 12
_MAX_CTA_CHARS = 40


class _CtaCollector(HTMLParser):
    """Collect visible labels of <a>/<button> elements that look like primary
    calls to action (signup / trial / demo / login). Same buffer-on-tag pattern
    as _HeadingCollector; matches the accumulated label against _CTA_PATTERNS."""

    def __init__(self) -> None:
        super().__init__()
        self.ctas: list[str] = []
        self._depth = 0  # >0 while inside an <a>/<button> (handles nested spans)
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("a", "button"):
            if self._depth == 0:
                self._buffer = []
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("a", "button") and self._depth > 0:
            self._depth -= 1
            if self._depth == 0:
                label = " ".join("".join(self._buffer).split())
                if label and any(p in label.lower() for p in _CTA_PATTERNS):
                    self.ctas.append(label[:_MAX_CTA_CHARS])

    def handle_data(self, data: str) -> None:
        if self._depth > 0:
            self._buffer.append(data)


def _extract_ctas(html: str) -> list[str]:
    """Primary call-to-action labels on a page, deduped, in document order.

    Called by: crawl(), once per fetched page. Rides on Page.ctas → chunk
    metadata → read_page surfaces them so the 'can I sign up?' persona sees
    the entry points that boilerplate-stripping removed from the body text."""
    collector = _CtaCollector()
    try:
        collector.feed(html)
    except Exception:
        return []
    seen: set[str] = set()
    ctas = []
    for c in collector.ctas:
        if c.lower() not in seen:
            seen.add(c.lower())
            ctas.append(c)
        if len(ctas) >= _MAX_CTAS:
            break
    return ctas


def _extract_headings(html: str) -> list[str]:
    """The page's section headings (h1-h3), deduped, in document order.

    Called by: crawl(), once per fetched page. Output rides on Page.headings.
    """
    collector = _HeadingCollector()
    try:
        collector.feed(html)
    except Exception:
        return []  # malformed HTML — a missing map, not a failed page
    seen: set[str] = set()
    headings = []
    for h in collector.headings:
        h = h[:_MAX_HEADING_CHARS]
        if h.lower() not in seen:
            seen.add(h.lower())
            headings.append(h)
        if len(headings) >= _MAX_HEADINGS:
            break
    return headings


def _extract_links(html: str, base_url: str, domain: str) -> list[str]:
    """Turn a page's HTML into the list of URLs the crawler may visit NEXT.

    Called by: crawl(), once per fetched page.
    Calls: _LinkCollector (above) to pull out raw hrefs.

    Raw hrefs are messy ("/pricing", "#features", "mailto:x", full URLs to other
    sites), so each one goes through a cleanup + filter pipeline:
      1. urljoin   — make relative links absolute ("/pricing" → "https://site.com/pricing")
      2. urldefrag — drop "#section" anchors (same page, would cause duplicates)
      3. keep only http(s), same domain, not a blocked auth path, not a binary file
    Whatever survives goes into crawl()'s queue.
    """
    collector = _LinkCollector()
    try:
        collector.feed(html)
    except Exception:
        return []  # malformed HTML — just don't follow links from this page

    links = []
    for href in collector.hrefs:
        absolute, _fragment = urldefrag(urljoin(base_url, href))
        parts = urlparse(absolute)
        path = parts.path.lower()
        if (
            parts.scheme in ("http", "https")
            and parts.netloc == domain  # stay on the starting site
            and not path.endswith(SKIP_EXTENSIONS)
            and not any(hint in path for hint in BLOCKED_PATH_HINTS)
        ):
            links.append(absolute)
    return links


def _static_fetch(client: httpx.Client, url: str) -> tuple[str, str]:
    """One page via static HTTP. Returns (html, text) or ("", "") on failure.
    text = trafilatura article extraction (favor_precision strips nav/footer —
    added after a nav-debris chunk scored 0.490 in the 2026-07-12 eval)."""
    try:
        response = client.get(url)
    except httpx.HTTPError:
        return "", ""
    content_type = response.headers.get("content-type", "")
    if response.status_code != 200 or "text/html" not in content_type:
        return "", ""
    html = response.text
    return html, (trafilatura.extract(html, favor_precision=True) or "")


def _crawl_loop(start_url: str, max_pages: int, fetch) -> CrawlResult:
    """Breadth-first crawl driven by a pluggable `fetch(url) -> (html, text)`.

    A queue holds URLs to visit; each fetched page's links join the back of it,
    so we sweep the start page, then everything one click away, etc., until
    max_pages readable pages or the queue empties. `seen` blocks revisits.
    `fetch` is the only thing that differs between the static and headless
    (JS-rendered) passes — everything else (robots gate, extraction, link
    discovery, thin detection) is identical, so the two passes can't drift."""
    domain = urlparse(start_url).netloc
    queue = [start_url]
    seen = {start_url}
    pages: list[Page] = []
    skipped_by_robots = 0
    seed_text_chars = 0  # the FIRST fetched page's text/html — the thin signal
    seed_html_chars = 0

    while queue and len(pages) < max_pages:
        url = queue.pop(0)  # FRONT of the queue = breadth-first

        # Gate: robots.txt permission (rule #1). No permission → no request.
        if not is_allowed(url):
            skipped_by_robots += 1
            continue

        html, text = fetch(url)  # static HTTP or headless render
        if seed_html_chars == 0 and html:
            seed_text_chars = len(text)
            seed_html_chars = len(html)
        if text.strip():
            pages.append(
                Page(
                    url=url,
                    text=text,
                    headings=_extract_headings(html),
                    ctas=_extract_ctas(html),
                )
            )
            events.emit("crawl.page", url=url, chars=len(text))
        # Feed new same-domain links into the queue (works for JS nav too —
        # the rendered pass discovers links a static fetch never sees).
        for link in _extract_links(html, base_url=url, domain=domain):
            if link not in seen:
                seen.add(link)
                queue.append(link)

        time.sleep(settings.request_delay_seconds)  # politeness delay

    return CrawlResult(
        pages=pages,
        skipped_by_robots=skipped_by_robots,
        thin_extraction=_is_thin_extraction(seed_text_chars, seed_html_chars),
    )


def crawl(start_url: str, max_pages: int) -> CrawlResult:
    """THE entry point: static crawl first; if the site reads as JS-rendered
    (thin extraction), re-crawl once with a headless browser.

    Called by: main.py ingest(). The cheap static path serves the majority
    (server-rendered) sites; a ~1000x-heavier browser spins up only when the
    static seed page came back near-empty. Fails safe: if the browser is
    unavailable, the static (thin, caveated) result still ships."""
    headers = {"User-Agent": settings.crawler_user_agent}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=15.0) as client:
        result = _crawl_loop(start_url, max_pages, lambda u: _static_fetch(client, u))

    if not result.thin_extraction:
        return result

    logger.info("thin static extraction (%s) — escalating to headless render", start_url)
    events.emit("render.escalate", url=start_url)
    try:
        with render.browser_session() as browser:
            rendered = _crawl_loop(
                start_url, max_pages, lambda u: render.render_page(browser, u)
            )
    except Exception as exc:
        # Browser missing / crashed → keep the static result (already thin-flagged).
        logger.error("headless render unavailable (%s) — keeping static result", exc)
        return result
    return rendered if rendered.pages else result
