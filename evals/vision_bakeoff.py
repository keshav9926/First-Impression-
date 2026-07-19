# evals/vision_bakeoff.py — can an NVIDIA VLM read a company's product
# screenshots? Tests the candidate vision models on Vortexify's real images.
#
# WHY: the text pipeline is blind to images — it can only note that a visual
# EXISTS (alt/filename metadata). A VLM that captions the screenshot would let
# personas judge what the UI actually communicates, not just its presence.
# This harness measures which model does that well, fast, on your key.
#
# Each image is downscaled to JPEG < ~170KB (NVIDIA inline-image limit) and
# sent via the OpenAI-compatible chat API as a data: URL. Text-only models are
# expected to fail — kept as a control.
# Usage: python -m evals.vision_bakeoff

import base64
import io
import json
import time
from pathlib import Path

import httpx
from openai import OpenAI
from PIL import Image

from app.config import settings

OUT = Path(__file__).resolve().parent / "vision_out"
BASE = "https://integrate.api.nvidia.com/v1"

IMAGES = [
    "https://www.vortexify.ai/uc-capacity-planning.png",
    "https://www.vortexify.ai/connectors.png",
]

# (label, model_id, expected_vision)
MODELS = [
    ("llama-3.2-90b-vision", "meta/llama-3.2-90b-vision-instruct", True),
    ("nemotron-nano-vl-8b", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1", True),
    ("nemotron-nano-12b-v2-vl", "nvidia/nemotron-nano-12b-v2-vl", True),
    ("nemotron-3-nano-omni-30b", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", True),
    ("gemma-3n-e2b", "google/gemma-3n-e2b-it", True),
    ("inkling", "thinkingmachines/inkling", None),          # unknown — probe
    ("llama-3.2-1b (text-only)", "meta/llama-3.2-1b-instruct", False),  # control
]

PROMPT = (
    "This is a screenshot from a B2B software company's website. As a first-time "
    "visitor, describe concretely what product interface or UI this image shows — "
    "what screens, data, charts, or controls are visible, and what it tells you the "
    "product does. Be specific and factual. If the image is unclear or not a product "
    "UI, say so plainly. 3-4 sentences."
)


def _fetch_jpeg_data_url(url: str, max_w: int = 900, budget_kb: int = 170) -> str:
    with httpx.Client(timeout=30, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"}) as c:
        raw = c.get(url).content
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)))
    for q in (85, 75, 65, 55, 45, 35):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        if buf.tell() <= budget_kb * 1024:
            break
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _ask(client: OpenAI, model: str, data_url: str) -> tuple[str, float]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        max_tokens=350,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip(), round(time.time() - t0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=BASE, api_key=settings.nvidia_api_key)

    print("Preparing images...", flush=True)
    data_urls = {}
    for url in IMAGES:
        data_urls[url] = _fetch_jpeg_data_url(url)
        print(f"  {url.split('/')[-1]} -> {len(data_urls[url]) // 1024}KB b64", flush=True)

    results = []
    for label, model, expect in MODELS:
        # one image (the capacity-planning dashboard) is enough to rank quality;
        # only re-test the winner set on the second image to save calls.
        url = IMAGES[0]
        print(f"\n=== {label} ({model}) ===", flush=True)
        row = {"label": label, "model": model, "expected_vision": expect}
        try:
            caption, secs = _ask(client, model, data_urls[url])
            row.update(ok=True, secs=secs, caption=caption)
            print(f"[{secs}s] {caption[:400]}", flush=True)
        except Exception as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            print(f"FAILED: {row['error']}", flush=True)
        results.append(row)

    (OUT / "vortexify_vision.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n--- SUMMARY ---")
    for r in results:
        status = f"OK {r['secs']}s" if r.get("ok") else "FAIL"
        print(f"  {r['label']:28} {status}")
    print(f"\nsaved -> {OUT / 'vortexify_vision.json'}")


if __name__ == "__main__":
    main()
