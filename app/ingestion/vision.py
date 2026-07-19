# app/ingestion/vision.py — read product screenshots the text extractor can't.
#
# The HTML text pipeline is BLIND to images: it can note that a visual exists
# (alt/filename) but not what it SHOWS. A site whose product story lives in
# dashboard screenshots therefore reads as "no product UI visible" — a
# confident false negative (caught live on vortexify.ai, 2026-07-19).
#
# This module captions a page's product images with a vision-language model
# (settings.vision_model — omni-30b, the 2026-07-19 bake-off winner: 2.3s/img,
# read exact chart contents). Captions ride back into chunk metadata via
# main.py, so the agent, personas, and judge all SEE what a screenshot depicts.
#
# FAIL-OPEN by design: vision is an enrichment layer, never a gate. Any download
# or model error on an image is logged and skipped — ingestion always proceeds.
# A hard cost cap (settings.vision_max_images_total) bounds calls per ingest.

import base64
import io
import logging

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


def _caption_one(client, http: httpx.Client, url: str) -> str | None:
    try:
        raw = http.get(url).content
    except Exception as exc:
        logger.info("vision: fetch failed %s (%s)", url, type(exc).__name__)
        return None
    data_url = _jpeg_data_url(raw)
    if not data_url:
        return None
    try:
        resp = client.chat.completions.create(
            model=settings.vision_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            max_tokens=180,
            temperature=0.0,
        )
    except Exception as exc:
        logger.info("vision: caption failed %s (%s)", url, type(exc).__name__)
        return None
    cap = (resp.choices[0].message.content or "").strip()
    if not cap or "non-product image" in cap.lower():
        return None  # nothing worth surfacing
    return cap[:_CAPTION_CHARS]


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
