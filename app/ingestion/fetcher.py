"""
===============================================================================
FILE: app/ingestion/fetcher.py
ORIGIN      : app.main (ingest / analyze_stream) / app.mcp_server
PURPOSE     : Dual-mode HTTP & Playwright web crawler with link extraction & HTML parsing
DESTINATION : app.ingestion.chunker / app.ingestion.vision / app.rag.store
===============================================================================
"""

import logging
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import trafilatura

from app import events
from app.config import settings
from app.ingestion import render
from app.ingestion.robots import is_allowed

logger = logging.getLogger("first_impression")

# Path SEGMENTS that mark authenticated / non-public areas — rule #1 says we
# never try these. Matched as whole segments (not substrings) so real content
# paths like /accounts-receivable or /joining-guide are NOT wrongly blocked
# (substring "account" was blocking a collections product's core pages).
BLOCKED_PATH_HINTS = (
    "login", "signin", "sign-in", "signup", "sign-up",
    "account", "accounts", "logout", "auth", "register",
)

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
    # Visual-content evidence: alt texts / filenames of meaningful <img>/<video>
    # elements. The text extractor is BLIND to visuals, and models were filling
    # that blind spot with confident false negatives ("no screenshot of the
    # product anywhere") on pages full of dashboard screenshots (caught live
    # 2026-07-19, vortexify.ai). `images` (labels) lets the agent and judge see
    # that visuals EXIST; `image_urls` feeds the vision captioner so they can
    # also see WHAT the visual shows (app/ingestion/vision.py).
    images: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    # Demo videos are the other half of that blind spot, and the more damaging
    # one: "is there a product demo?" is a question the report must answer, and
    # a <video>-only check saw none of them (marketing sites ship demos as
    # YouTube/Vimeo/Loom/Wistia iframes). `video_urls` are direct media files we
    # can actually send to the omni VLM (app/ingestion/vision.py caption_videos);
    # third-party embeds can't be downloaded, so we caption their poster frame
    # via image_urls instead and label the embed in `images`.
    video_urls: list[str] = field(default_factory=list)


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
# Below this many discovered pages, a static crawl likely missed JS-injected
# nav (SPA) — escalate to headless render, which can see those links.
_MIN_STATIC_PAGES = 3


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


def _canonical(url: str) -> str:
    """Canonical form for crawl de-duplication: strip trailing slash and drop
    query/fragment. Without this, 'site.com' vs 'site.com/' and 'site.com/?r=0'
    were treated as DISTINCT pages — inflating page counts (collectwise showed
    '2 pages' that were really 1) and wasting the crawl budget on duplicates.
    Marketing sites rarely key distinct content on query strings, so dropping
    them is safe here."""
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}{p.path.rstrip('/')}"


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


# Visual-evidence extraction. Icons/logos/pixels are noise; product screenshots
# and demo videos are SIGNAL (their absence is a legitimate finding, their
# presence must not be "not found"). Heuristic: skip tiny images and
# obvious chrome (logo/icon/favicon/avatar), keep the rest as alt-or-filename.
_MAX_IMAGES = 15
_MAX_IMAGE_CHARS = 90
_IMG_NOISE_HINTS = ("logo", "icon", "favicon", "avatar", "sprite", "badge", "arrow", "pixel")

# Video extraction. Detecting only <video>/<source type=video> found almost
# nothing in practice: product demos live in third-party player iframes, and
# self-hosted hero videos are usually lazy-mounted (data-src) so the tag has no
# src at parse time. Each host below maps to a display name; the player URL also
# yields a poster/thumbnail we CAN read even when the video itself is off-limits.
_MAX_VIDEOS = 5
_VIDEO_HOSTS = {
    "youtube.com": "YouTube", "youtube-nocookie.com": "YouTube", "youtu.be": "YouTube",
    "vimeo.com": "Vimeo", "loom.com": "Loom", "wistia.com": "Wistia",
    "wistia.net": "Wistia", "vidyard.com": "Vidyard", "videoask.com": "VideoAsk",
    "dailymotion.com": "Dailymotion", "hubspotvideo.com": "HubSpot",
}
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".m4v", ".ogv")
# Lazy-mount attributes: a hero <video> often carries no src until JS runs, and
# click-to-play embeds park the player URL in a data-* attribute on a wrapper.
_VIDEO_SRC_ATTRS = ("src", "href", "data-src", "data-lazy-src", "data-url",
                    "data-video-url", "data-video-src", "data-vimeo-url")
# A bare youtube.com/@company link is a social footer, not a demo. Only URLs
# that name ONE video count as a video on the page.
_SPECIFIC_VIDEO_MARKERS = ("/watch", "/embed/", "youtu.be/", "/share/", "/video/",
                           "player.vimeo.com/", "/medias/")


def _video_hit(tag: str, a: dict[str, str | None]) -> tuple[str, str, str] | None:
    """(label, media_url, player_url) if this tag is a video, else None.
    `media_url` is a directly downloadable media file when there is one;
    `player_url` is a third-party player page/iframe, which we cannot download
    but can often reduce to a poster frame."""
    urls = [(a.get(k) or "").strip() for k in _VIDEO_SRC_ATTRS]
    direct = next((u for u in urls if u.split("?")[0].lower().endswith(_VIDEO_EXTENSIONS)), "")

    if tag in ("video", "source"):
        return None  # handled statefully by _ImgCollector: one element, one video

    for url in urls:
        low = url.lower()
        provider = next((p for h, p in _VIDEO_HOSTS.items() if h in low), "")
        if not provider:
            continue
        # <a> is the loose case (a footer link to the company's channel is not a
        # demo); iframes/embeds always point at one player, so they need no check.
        if tag == "a" and not any(m in low for m in _SPECIFIC_VIDEO_MARKERS):
            continue
        title = (a.get("title") or "").strip()
        label = f"[video: {provider} embed" + (f' — "{title[:60]}"' if title else "") + "]"
        return label, "", url
    if direct:
        return f"[video: {direct.rsplit('/', 1)[-1].split('?')[0]}]", direct, ""
    return None


def _youtube_thumbnail(player_url: str) -> str:
    """A YouTube player URL → its poster-frame JPEG, or "". We can't download
    the video itself, but the thumbnail is a real frame of it and goes through
    the ordinary image captioner — turning "a video exists" into "the video
    shows the dashboard". hqdefault always exists; maxresdefault often 404s."""
    low = player_url.lower()
    if not any(h in low for h in ("youtube.com", "youtu.be", "youtube-nocookie.com")):
        return ""
    path = urlparse(player_url)
    vid = ""
    if "/embed/" in path.path:
        vid = path.path.split("/embed/", 1)[1]
    elif "youtu.be" in path.netloc:
        vid = path.path.lstrip("/")
    elif "v=" in path.query:
        vid = path.query.split("v=", 1)[1].split("&")[0]
    vid = vid.split("/")[0].split("?")[0]
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else ""


class _ImgCollector(HTMLParser):
    """Collect evidence that meaningful visuals exist: <img> alt/src (minus
    obvious chrome), plus every video on the page — self-hosted files, player
    embeds, and the poster frames that let us see what they show."""

    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []          # display labels (alt/filename)
        self.image_srcs: list[str] = []       # raw src of the SAME images, aligned
        self.videos: list[str] = []           # labels: "[video: YouTube embed …]"
        self.video_srcs: list[str] = []       # downloadable media files only
        self.video_players: list[str] = []    # third-party player URLs (→ poster frames)
        self._open_video: str | None = None   # src chosen for the <video> being parsed

    def _note_video(self, label: str, media: str, player: str) -> None:
        if label in self.videos or len(self.videos) >= _MAX_VIDEOS:
            return
        self.videos.append(label)
        if media:
            self.video_srcs.append(media)
        if player:
            self.video_players.append(player)

    def _pick_source(self, url: str, declared_video: bool = False) -> None:
        """A <video> lists the SAME content in several formats. Keep one, and
        prefer mp4 — it is what the VLM decodes most reliably. Anything we can't
        hand to a decoder (blob:, .m3u8 streams) is left out: the element still
        counts as a video, it just has no file we can watch."""
        if not url or not (declared_video or url.split("?")[0].lower().endswith(_VIDEO_EXTENSIONS)):
            return
        cur = self._open_video or ""
        if not cur or (not cur.split("?")[0].lower().endswith(".mp4")
                       and url.split("?")[0].lower().endswith(".mp4")):
            self._open_video = url

    def close_video(self) -> None:
        """Emit the <video> element we were parsing as exactly one video. Called
        at </video> and once more after feed(), for pages that never close it."""
        if self._open_video is None:
            return
        src = self._open_video
        name = src.rsplit("/", 1)[-1].split("?")[0] if src else "video element"
        self._open_video = None
        self._note_video(f"[video: {name}]", src, "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "video":
            self.close_video()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "video":
            self.close_video()  # unclosed previous element — don't merge the two
            self._open_video = ""  # "" = open, no source yet
            for key in _VIDEO_SRC_ATTRS:
                self._pick_source((a.get(key) or "").strip())
            # The poster is a real frame of the demo — the cheapest way to see a
            # video we may not be able to download. The embed equivalent (player
            # thumbnail) is resolved in _extract_images.
            if poster := (a.get("poster") or "").strip():
                self.images.append(f"video poster ({poster.rsplit('/', 1)[-1].split('?')[0]})")
                self.image_srcs.append(poster)
            return
        if tag == "source":
            declared = (a.get("type") or "").startswith("video")
            if self._open_video is not None:
                for key in _VIDEO_SRC_ATTRS:
                    self._pick_source((a.get(key) or "").strip(), declared)
            elif declared:  # malformed page: a <source> with no <video> around it
                self._open_video = ""
                self._pick_source((a.get("src") or "").strip(), True)
                self.close_video()
            return
        if tag in ("iframe", "embed", "a", "div"):
            if hit := _video_hit(tag, a):
                self._note_video(*hit)
            return
        if tag != "img":
            return
        src = (a.get("src") or a.get("data-src") or "").strip()
        alt = (a.get("alt") or "").strip()
        name = src.rsplit("/", 1)[-1].split("?")[0]
        blob = f"{alt} {name}".lower()
        if not (alt or name) or any(h in blob for h in _IMG_NOISE_HINTS):
            return
        # tiny declared size → decorative
        try:
            if int(a.get("width") or 999) < 64 or int(a.get("height") or 999) < 64:
                return
        except ValueError:
            pass
        self.images.append((alt or name)[:_MAX_IMAGE_CHARS])
        self.image_srcs.append(src)


def _extract_images(html: str, base_url: str = "") -> tuple[list[str], list[str], list[str]]:
    """(labels, image_urls, video_urls) for the substantive visuals on a page,
    deduped by label, document order. `labels` ride on Page.images → chunk
    metadata; `image_urls` ride on Page.image_urls → vision captioning;
    `video_urls` ride on Page.video_urls → video captioning. Together they close
    the text-only blind spot around visuals."""
    collector = _ImgCollector()
    try:
        collector.feed(html)
    except Exception:
        return [], [], []
    collector.close_video()  # a <video> the page never closed still counts

    def _absolute(src: str) -> str:
        abs_url = urljoin(base_url, src) if base_url else src
        return abs_url if abs_url.startswith(("http://", "https://")) else ""

    seen: set[str] = set()
    labels, urls = [], []
    for label, src in zip(collector.images, collector.image_srcs):
        if label.lower() in seen:
            continue
        seen.add(label.lower())
        labels.append(label)
        if abs_url := _absolute(src):
            urls.append(abs_url)
        if len(labels) >= _MAX_IMAGES:
            break
    # Videos lead the labels: "is there a product demo?" is its own question in
    # the report, and it must not fall off the end of a 15-image list.
    labels = collector.videos + labels
    video_urls = [u for u in (_absolute(s) for s in collector.video_srcs) if u]
    # Player embeds are undownloadable, but their poster frame is a real frame
    # of the demo — send it down the image path so the VLM still sees it.
    for player in collector.video_players:
        if thumb := _youtube_thumbnail(_absolute(player) or player):
            urls.append(thumb)
    return labels, urls, video_urls


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
        segments = [s for s in path.split("/") if s]
        if (
            parts.scheme in ("http", "https")
            and parts.netloc == domain  # stay on the starting site
            and not path.endswith(SKIP_EXTENSIONS)
            # whole-segment match: /login blocked, /accounts-receivable allowed
            and not any(seg in BLOCKED_PATH_HINTS for seg in segments)
        ):
            links.append(absolute)
    return links


def _static_fetch(client: httpx.Client, url: str) -> tuple[str, str, str]:
    """One page via static HTTP. Returns (html, text, final_url) or ("","","")
    on failure. final_url is the post-redirect address — the caller resolves
    links against it, never against what we asked for.
    text = trafilatura article extraction (favor_precision strips nav/footer —
    added after a nav-debris chunk scored 0.490 in the 2026-07-12 eval)."""
    try:
        response = client.get(url)
    except httpx.HTTPError:
        return "", "", ""
    content_type = response.headers.get("content-type", "")
    if response.status_code != 200 or "text/html" not in content_type:
        return "", "", ""
    html = response.text
    return html, (trafilatura.extract(html, favor_precision=True) or ""), str(response.url)


# Soft 404s: SPAs routinely serve "Not Found" with a 200 status, so the status
# check above can't catch them. A real page is never both this short and this
# shaped — the length guard is what keeps the pattern from eating real content.
_SOFT_404_MAX_CHARS = 200
_SOFT_404 = re.compile(
    r"^\s*(?:http\s*status:?\s*)?(?:404|error\s*404)\b|^\s*(?:page\s+)?not\s+found\b",
    re.IGNORECASE,
)


def _is_error_page(text: str) -> bool:
    """Is this an error page wearing a 200? Only judged on very short text."""
    stripped = text.strip()
    return bool(stripped) and len(stripped) <= _SOFT_404_MAX_CHARS and bool(_SOFT_404.match(stripped))


def _crawl_loop(start_url: str, max_pages: int, fetch) -> CrawlResult:
    """Breadth-first crawl driven by a pluggable `fetch(url) -> (html, text)`.

    A queue holds URLs to visit; each fetched page's links join the back of it,
    so we sweep the start page, then everything one click away, etc., until
    max_pages readable pages or the queue empties. `seen` blocks revisits.
    `fetch` is the only thing that differs between the static and headless
    (JS-rendered) passes — everything else (robots gate, extraction, link
    discovery, thin detection) is identical, so the two passes can't drift."""
    domain = urlparse(start_url).netloc
    start = _canonical(start_url)
    queue = [start]
    seen = {start}
    pages: list[Page] = []
    skipped_by_robots = 0
    seed_text_chars = 0  # the FIRST fetched page's text/html — the thin signal
    seed_html_chars = 0
    seed_done = False  # has the seed resolved? (fixes `domain` after a redirect)

    while queue and len(pages) < max_pages:
        url = queue.pop(0)  # FRONT of the queue = breadth-first

        # Gate: robots.txt permission (rule #1). No permission → no request.
        if not is_allowed(url):
            skipped_by_robots += 1
            continue

        html, text, final_url = fetch(url)  # static HTTP or headless render
        # Redirects change a page's IDENTITY. vortexify.ai redirects only "/"
        # to www and hard-404s every other apex path, so resolving links
        # against the REQUESTED url queued 14 dead URLs (2026-08-01). Adopt
        # where we actually landed: the seed's host becomes the crawl's
        # domain, and each page's links resolve against its final address.
        if final_url:
            landed = urlparse(final_url).netloc
            if not seed_done and landed:
                domain = landed  # the seed redirected — that IS the site
            elif landed and landed != domain:
                continue  # this page redirected off-site; not ours to report on
            canonical_final = _canonical(final_url)
            if canonical_final != url:
                if canonical_final in seen:
                    continue  # already crawled under its post-redirect URL
                seen.add(canonical_final)
                url = canonical_final
        if seed_html_chars == 0 and html:
            seed_text_chars = len(text)
            seed_html_chars = len(html)
            seed_done = True
        if _is_error_page(text):
            logger.info("skipping soft-404 at %s: %r", url, text.strip()[:60])
            continue
        if text.strip():
            labels, image_urls, video_urls = _extract_images(html, base_url=url)
            pages.append(
                Page(
                    url=url,
                    text=text,
                    headings=_extract_headings(html),
                    ctas=_extract_ctas(html),
                    images=labels,
                    image_urls=image_urls,
                    video_urls=video_urls,
                )
            )
            events.emit("crawl.page", url=url, chars=len(text))
        # Feed new same-domain links into the queue (works for JS nav too —
        # the rendered pass discovers links a static fetch never sees).
        for link in _extract_links(html, base_url=url, domain=domain):
            clink = _canonical(link)  # de-dupe trailing-slash / query variants
            if clink not in seen:
                seen.add(clink)
                queue.append(clink)

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

    # Escalate to headless render on EITHER signal:
    #   - thin_extraction: the seed page is a JS shell (little text), OR
    #   - too few pages: link discovery found almost nothing, which on an SPA
    #     means the nav is JS-injected and a static fetch can't see it (this is
    #     why collectwise analyzed only 1 real page). Rendering finds those links.
    if not settings.force_render and not result.thin_extraction and len(result.pages) >= _MIN_STATIC_PAGES:
        return result

    reason = ("force_render enabled" if settings.force_render else
              "thin static extraction" if result.thin_extraction else
              f"only {len(result.pages)} page(s) found (JS-injected nav?)")
    logger.info("%s (%s) — escalating to headless render", reason, start_url)
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
