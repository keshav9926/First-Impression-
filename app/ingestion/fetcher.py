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

import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import trafilatura

from app.config import settings
from app.ingestion.robots import is_allowed

# Paths that hint at authenticated / non-public areas — rule #1 says we never try these.
BLOCKED_PATH_HINTS = ("login", "signin", "sign-in", "signup", "sign-up", "account", "logout")

# File types we can't turn into text (PDF support could come later).
SKIP_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".mp4", ".css", ".js", ".xml", ".json",
)


@dataclass
class Page:
    """One successfully fetched page: its URL + the readable text we extracted.

    Produced by: crawl().
    Consumed by: main.py ingest(), which sends .text to chunker.chunk_text()
    and keeps .url so every chunk can cite the page it came from.
    """

    url: str
    text: str


@dataclass
class CrawlResult:
    """What crawl() hands back to the /ingest endpoint:
    the list of Pages plus a count of URLs robots.txt made us skip
    (reported to the user in the API response)."""

    pages: list[Page]
    skipped_by_robots: int


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


def crawl(start_url: str, max_pages: int) -> CrawlResult:
    """THE entry point of this file: breadth-first crawl from start_url.

    Called by: main.py ingest() (the /ingest endpoint).
    Calls: robots.is_allowed() per URL, then httpx to fetch,
           trafilatura.extract() to get text, _extract_links() to grow the queue.

    "Breadth-first" = we keep a queue: start with one URL, and every page we
    fetch adds its links to the back of the queue. So we visit the start page,
    then everything one click away, then two clicks away... until we have
    max_pages readable pages or run out of URLs.

    `seen` prevents visiting the same URL twice (pages link to each other
    constantly — without this the queue would loop forever).
    """
    domain = urlparse(start_url).netloc
    queue = [start_url]
    seen = {start_url}
    pages: list[Page] = []
    skipped_by_robots = 0

    headers = {"User-Agent": settings.crawler_user_agent}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=15.0) as client:
        while queue and len(pages) < max_pages:
            url = queue.pop(0)  # take from the FRONT of the queue (= breadth-first)

            # Gate 1: robots.txt permission (robots.py). No permission → no request.
            if not is_allowed(url):
                skipped_by_robots += 1
                continue

            # Gate 2: the actual download. Network errors skip the page, not the crawl.
            try:
                response = client.get(url)
            except httpx.HTTPError:
                continue

            # Gate 3: only successful HTML responses are worth parsing.
            content_type = response.headers.get("content-type", "")
            if response.status_code != 200 or "text/html" not in content_type:
                continue

            # HTML → readable text (nav/footer/cookie-banner stripped).
            html = response.text
            text = trafilatura.extract(html) or ""
            if text.strip():
                pages.append(Page(url=url, text=text))

            # Feed new same-domain links into the queue for later iterations.
            for link in _extract_links(html, base_url=url, domain=domain):
                if link not in seen:
                    seen.add(link)
                    queue.append(link)

            time.sleep(settings.request_delay_seconds)  # politeness delay (rate limit)

    return CrawlResult(pages=pages, skipped_by_robots=skipped_by_robots)
