"""
===============================================================================
FILE: app/ingestion/vision.py
ORIGIN      : app.main (ingest_site)
PURPOSE     : VLM Multimodal image captioning for web page product screenshots
DESTINATION : app.main (Attaches captions into chunk metadata)
===============================================================================
"""

import base64
import io
import logging
import time

import httpx

from app import observability
from app.config import settings

logger = logging.getLogger("first_impression")

_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
_MAX_W = 900          # downscale width — enough detail, small payload
_BUDGET_KB = 170      # NVIDIA inline-image ceiling (~180KB); stay under it
_CAPTION_CHARS = 320  # keep each caption compact in metadata

_PROMPT = (
    "This is an image from a B2B software company's public website. In ONE sentence, "
    "state concretely what it shows — the product UI, dashboard, chart, or data visible, "
    "and what it tells a first-time visitor the product does. If it is decorative, a logo, "
    "or not a product interface, reply exactly: 'non-product image'."
)


def _client():
    """OpenAI-compatible client on the NVIDIA endpoint (traced if Langfuse on)."""
    openai_cls = observability.openai_client_class()
    return openai_cls(base_url=_NVIDIA_BASE, api_key=settings.nvidia_api_key)


def _jpeg_data_url(raw: bytes) -> str | None:
    """Downscale/re-encode fetched bytes to a JPEG data: URL under the inline cap.
    Returns None if PIL can't read the bytes (e.g. an SVG or corrupt file)."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("vision: Pillow not installed — skipping image captioning")
        return None
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None  # not a raster image (SVG, etc.) — skip
    if img.width > _MAX_W:
        img = img.resize((_MAX_W, round(img.height * _MAX_W / img.width)))
    for q in (85, 75, 65, 55, 45, 35):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        if buf.tell() <= _BUDGET_KB * 1024:
            break
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# Transient signals worth a backoff-retry / failover to the next model, rather
# than a silent skip. The 503 "resource exhausted / worker local limit" storm is
# exactly this — a momentary concurrency cap, not a permanent failure.
_TRANSIENT = ("503", "504", "429", "resourceexhausted", "timeout",
              "temporarily", "overloaded", "service unavailable")
_FAILED = object()  # sentinel: this model errored out — try the next one


def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(t in s for t in _TRANSIENT)


def _try_model(client, model: str, data_url: str):
    """Caption `data_url` with ONE model, retrying transient errors with backoff.
    Returns the caption str, None (valid 'non-product' answer), or _FAILED (the
    model errored past its retries → caller should fail over to the next model)."""
    for attempt in range(settings.vision_retries_per_model):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
                max_tokens=180,
                temperature=0.0,
            )
            cap = (resp.choices[0].message.content or "").strip()
            if not cap or "non-product image" in cap.lower():
                return None  # a valid answer: nothing worth surfacing
            return cap[:_CAPTION_CHARS]
        except Exception as exc:
            if _is_transient(exc) and attempt < settings.vision_retries_per_model - 1:
                time.sleep(1.5 * (attempt + 1))  # 1.5s, 3s, ... backoff
                continue
            logger.info("vision: %s failed (%s) — failing over", model, type(exc).__name__)
            return _FAILED
    return _FAILED


def _caption_one(client, http: httpx.Client, url: str) -> str | None:
    try:
        raw = http.get(url).content
    except Exception as exc:
        logger.info("vision: fetch failed %s (%s)", url, type(exc).__name__)
        return None
    data_url = _jpeg_data_url(raw)
    if not data_url:
        return None
    # Try each vision model in turn; a transient/hard model error fails over to
    # the next (different worker = different concurrency pool). Only a real
    # caption or a definitive 'non-product' verdict stops the chain.
    for model in settings.vision_models:
        result = _try_model(client, model, data_url)
        if result is _FAILED:
            continue
        return result  # caption str, or None for a non-product image
    logger.info("vision: all models failed for %s", url)
    return None


# --- Video ---------------------------------------------------------------
# A demo video is often the clearest statement a company makes about what the
# product does, and knowing only that a <video> tag exists tells the report
# nothing. omni-30b accepts `video_url` parts (remote URL or inline base64), so
# the same failover machinery above carries over — only the payload changes.
_VIDEO_PROMPT = (
    "This is a video from a B2B software company's public website. In 2-3 sentences, "
    "state concretely what it shows — the product UI and workflow demonstrated, the "
    "specific screens/data visible, and what a first-time visitor would learn the "
    "product does. If it is purely decorative b-roll with no product in it, reply "
    "exactly: 'non-product video'."
)
_VIDEO_CAPTION_CHARS = 700  # richer than an image caption: a demo carries more


def _try_video_model(client, model: str, video_url: str):
    """Caption one video with ONE model. Same contract as _try_model: caption
    str, None ('nothing worth surfacing'), or _FAILED (→ try the next model)."""
    for attempt in range(settings.vision_retries_per_model):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": _VIDEO_PROMPT},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ]}],
                max_tokens=400,
                temperature=0.0,
                timeout=settings.video_timeout_s,
            )
            cap = (resp.choices[0].message.content or "").strip()
            if not cap or "non-product video" in cap.lower():
                return None
            return cap[:_VIDEO_CAPTION_CHARS]
        except Exception as exc:
            if _is_transient(exc) and attempt < settings.vision_retries_per_model - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            logger.info("vision: %s failed on video (%s: %s)", model,
                        type(exc).__name__, str(exc)[:160])
            return _FAILED
    return _FAILED


def _inline_video(http: httpx.Client, url: str) -> str | None:
    """Download the video and return it as a data: URL, or None if it is too
    big / unfetchable. Only used when NVIDIA's own fetch of the URL failed
    (hotlink protection, signed CDN, private network)."""
    try:
        raw = http.get(url).content
    except Exception as exc:
        logger.info("vision: video fetch failed %s (%s)", url, type(exc).__name__)
        return None
    if len(raw) > settings.video_max_bytes:
        logger.info("vision: video too large (%.1fMB) — skipping %s", len(raw) / 1e6, url)
        return None
    mime = "video/webm" if url.split("?")[0].lower().endswith(".webm") else "video/mp4"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _caption_video(client, http: httpx.Client, url: str) -> str | None:
    """Remote URL first (no download, no size ceiling — NVIDIA fetches it), then
    inline bytes as the fallback. Returns a caption or None."""
    for model in settings.video_models:
        result = _try_video_model(client, model, url)
        if result is not _FAILED:
            return result
    data_url = _inline_video(http, url)
    if not data_url:
        return None
    for model in settings.video_models:
        result = _try_video_model(client, model, data_url)
        if result is not _FAILED:
            return result
    logger.info("vision: all models failed for video %s", url)
    return None


def caption_videos(pages) -> dict[str, list[str]]:
    """{page_url: [captions]} for the self-hosted demo videos across the crawl.

    Called by main.py ingest, right after caption_pages. Third-party embeds
    (YouTube/Vimeo/Loom) can't be downloaded and never reach here — the fetcher
    routes their poster frame through the image path instead. Never raises."""
    if not settings.video_enabled or not settings.nvidia_api_key:
        return {}
    total = settings.video_max_total
    if total <= 0:
        return {}

    client = _client()
    out: dict[str, list[str]] = {}
    with httpx.Client(timeout=90, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (FIE vision)"}) as http:
        for page in pages:
            if total <= 0:
                break
            caps: list[str] = []
            for url in getattr(page, "video_urls", [])[: settings.video_max_per_page]:
                if total <= 0:
                    break
                total -= 1
                cap = _caption_video(client, http, url)
                if cap:
                    name = url.rsplit("/", 1)[-1].split("?")[0]
                    caps.append(f"{name} (video) — {cap}")
            if caps:
                out[page.url] = caps
    if out:
        logger.info("vision: watched %d video(s) across %d page(s)",
                    sum(len(v) for v in out.values()), len(out))
    return out


def caption_pages(pages) -> dict[str, list[str]]:
    """{page_url: [captions]} for the product images across the crawl.

    Called by main.py ingest after crawl, before chunking. Respects the
    per-page and total caps. Never raises — returns {} if vision is disabled or
    unconfigured, so ingestion is unaffected."""
    if not settings.vision_enabled or not settings.nvidia_api_key:
        return {}
    total = settings.vision_max_images_total
    if total <= 0:
        return {}

    client = _client()
    out: dict[str, list[str]] = {}
    with httpx.Client(timeout=30, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (FIE vision)"}) as http:
        for page in pages:
            if total <= 0:
                break
            caps: list[str] = []
            for url in page.image_urls[: settings.vision_max_images_per_page]:
                if total <= 0:
                    break
                total -= 1
                cap = _caption_one(client, http, url)
                if cap:
                    name = url.rsplit("/", 1)[-1].split("?")[0]
                    caps.append(f"{name} — {cap}")
            if caps:
                out[page.url] = caps
    if out:
        logger.info("vision: captioned %d image(s) across %d page(s)",
                    sum(len(v) for v in out.values()), len(out))
    return out
