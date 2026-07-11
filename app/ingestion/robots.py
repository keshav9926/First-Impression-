# app/ingestion/robots.py — robots.txt compliance (hard rule #1: public data only).
# Before ANY page is fetched, this module checks whether the site's robots.txt
# allows our user agent to access that URL. robots.txt is the standard file
# where site owners declare which paths crawlers may and may not visit.
#
# Design choices worth explaining:
# - Parsers are cached per site so we download robots.txt once, not per page.
# - If robots.txt can't be retrieved due to a network error, we choose the
#   CONSERVATIVE interpretation: treat the site as off-limits. (A missing
#   robots.txt (404) is different — the standard says that means "allow all",
#   and the stdlib parser already handles that case.)

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from app.config import settings

# Cache: "https://example.com" -> parsed robots.txt for that site
_parsers: dict[str, RobotFileParser] = {}


def is_allowed(url: str) -> bool:
    """Return True if robots.txt permits our crawler to fetch this URL."""
    parts = urlparse(url)
    site_root = f"{parts.scheme}://{parts.netloc}"

    parser = _parsers.get(site_root)
    if parser is None:
        parser = RobotFileParser()
        parser.set_url(f"{site_root}/robots.txt")
        try:
            parser.read()  # downloads and parses the file
        except Exception:
            # Network failure — we can't verify permission, so we don't fetch.
            parser.disallow_all = True
        _parsers[site_root] = parser

    return parser.can_fetch(settings.crawler_user_agent, url)
