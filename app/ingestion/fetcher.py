# app/ingestion/fetcher.py — polite, same-domain crawler.
# Enforces the "public data only" rule in code, not in prompts:
#   1. every URL is checked against robots.txt BEFORE fetching (robots.py)
#   2. a fixed delay between requests (rate limiting — never hammer a site)
#   3. never leaves the starting domain, never follows login/signup paths
# Page HTML is reduced to readable article text with trafilatura (nav bars,
# cookie banners, footers stripped) — clean text in = good retrieval later.

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
    url: str
    text: str


@dataclass
class CrawlResult:
    pages: list[Page]
    skipped_by_robots: int


class _LinkCollector(HTMLParser):
    """Minimal HTML parser that collects every <a href="..."> value on a page."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.hrefs.append(value)


def _extract_links(html: str, base_url: str, domain: str) -> list[str]:
    """Find crawlable links: same domain, http(s), no auth pages, no binary files."""
    collector = _LinkCollector()
    try:
        collector.feed(html)
    except Exception:
        return []

    links = []
    for href in collector.hrefs:
        absolute, _fragment = urldefrag(urljoin(base_url, href))  # resolve relative, drop #anchors
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
    """Breadth-first crawl from start_url, up to max_pages readable pages."""
    domain = urlparse(start_url).netloc
    queue = [start_url]
    seen = {start_url}
    pages: list[Page] = []
    skipped_by_robots = 0

    headers = {"User-Agent": settings.crawler_user_agent}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=15.0) as client:
        while queue and len(pages) < max_pages:
            url = queue.pop(0)

            if not is_allowed(url):
                skipped_by_robots += 1
                continue

            try:
                response = client.get(url)
            except httpx.HTTPError:
                continue  # unreachable page — skip, don't crash the whole crawl

            content_type = response.headers.get("content-type", "")
            if response.status_code != 200 or "text/html" not in content_type:
                continue

            html = response.text
            text = trafilatura.extract(html) or ""
            if text.strip():
                pages.append(Page(url=url, text=text))

            for link in _extract_links(html, base_url=url, domain=domain):
                if link not in seen:
                    seen.add(link)
                    queue.append(link)

            time.sleep(settings.request_delay_seconds)  # politeness delay (rate limit)

    return CrawlResult(pages=pages, skipped_by_robots=skipped_by_robots)
