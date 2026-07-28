"""
===============================================================================
FILE: app/ingestion/robots.py
ORIGIN      : app.main (POST /ingest) / app.ingestion.fetcher (crawl_site)
PURPOSE     : Politeness guard validating site crawl permission via robots.txt
DESTINATION : app.ingestion.fetcher (Proceed with crawl or abort URL)
===============================================================================
"""

import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from app.config import settings

# Cache: "https://example.com" -> (parsed robots.txt, fetched_at). Entries
# expire after _TTL so a long-lived server re-checks permissions instead of
# honoring a stale allow/deny (or a transient fail-closed) forever.
_TTL_SECONDS = 3600.0
_parsers: dict[str, tuple[RobotFileParser, float]] = {}


def is_allowed(url: str) -> bool:
    """Return True if robots.txt permits our crawler to fetch this URL.

    Called by: main.py ingest() (once, for the seed URL) and
               fetcher.crawl() (for every URL before it is downloaded).
    Calls: stdlib RobotFileParser — .read() downloads+parses robots.txt,
           .can_fetch() answers "may THIS user agent visit THIS path?".

    Steps:
      1. Reduce the URL to its site root ("https://site.com/docs/x" → "https://site.com").
      2. If we haven't seen this site yet, download and parse its robots.txt
         (fail-closed on network errors), then cache the parser.
      3. Ask the cached parser about this specific URL.
    """
    parts = urlparse(url)
    site_root = f"{parts.scheme}://{parts.netloc}"

    cached = _parsers.get(site_root)
    if cached is None or time.time() - cached[1] > _TTL_SECONDS:
        parser = RobotFileParser()
        parser.set_url(f"{site_root}/robots.txt")
        try:
            parser.read()  # downloads and parses the file
        except Exception:
            # Network failure — we can't verify permission, so we don't fetch.
            parser.disallow_all = True
        _parsers[site_root] = (parser, time.time())
    else:
        parser = cached[0]

    return parser.can_fetch(settings.crawler_user_agent, url)
