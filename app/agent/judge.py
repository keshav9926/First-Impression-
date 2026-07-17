# app/agent/judge.py — groundedness judge (Phase 5; the folded-in "skeptic").
#
# WHAT grounding.py CANNOT catch: enforce_citations only proves the cited URL
# EXISTS in the store — not that the page actually SUPPORTS the claim. A
# synthesis LLM can attach a real URL to an invented claim and sail through.
# This judge closes that hole: one adversarial LLM pass that reads each claim
# NEXT TO the actual text of its cited page and votes supported / unsupported.
# Unsupported observations are DROPPED (accuracy over volume) and logged.
#
# DESIGN:
# - ONE pooled LLM call per report (not per claim), prefer Cerebras JSON mode —
#   Gemini's tiny daily quota is reserved for synthesis (llm_pool.py), and one
#   call judging all claims costs the same quota as judging one.
# - Page text comes from the STORE (the exact evidence the agent saw), capped
#   per page so the judge prompt stays bounded.
# - FAIL-OPEN: if the judge call itself dies (429/network), the report ships
#   unjudged with a warning log — the deterministic guards (citations, thin
#   flag) already ran, and refusing to ship over a judge outage would make
#   availability depend on a nice-to-have layer.
#
# CALL FLOW:
#   report.apply_guards() → verify_groundedness(report)   (all report paths)

import json
import logging
import re

import pydantic

from app.agent import llm_pool
from app.config import settings
from app.rag import store
from app.schemas import FirstImpressionReport

logger = logging.getLogger("first_impression")

_PAGE_EXCERPT_CHARS = 3000  # per cited page, keeps the judge prompt bounded
_JUDGED_FIELDS = (
    "what_the_product_is",
    "likely_new_user_journey",
    "friction_points",
    "standout_strengths",
)

_JUDGE_INSTRUCTION = """\
You are a strict fact-checker. For every numbered claim below, decide whether
the SOURCE PAGE TEXT actually supports it. A claim is supported only if the
page text states or clearly implies it — plausible-sounding is NOT enough.
Claims about something MISSING/unclear are supported when the page text indeed
does not show it. Return a verdict for EVERY claim, in order."""


class ClaimVerdict(pydantic.BaseModel):
    index: int  # the claim's number in the prompt
    supported: bool
    reason: str = ""  # optional — we never read it; kept so old payloads validate


class _Verdicts(pydantic.BaseModel):
    verdicts: list[ClaimVerdict]


# Salvage complete "index"+"supported" pairs from a response body — used when
# the model's JSON is truncated (a completion-token cap can cut the array
# mid-object). We only ever need index + supported, so counting the COMPLETE
# pairs recovers every finished verdict and safely ignores the cut-off tail.
_VERDICT_RE = re.compile(r'"index"\s*:\s*(\d+)\s*,\s*"supported"\s*:\s*(true|false)')


def _parse_verdicts(content: str) -> list[ClaimVerdict]:
    """Parse the judge reply into verdicts, tolerating a truncated array.

    Strict json.loads first (the happy path); if that fails — almost always an
    output-token cut mid-string — fall back to regex-salvaging every COMPLETE
    index/supported pair. A partial judge result still drops the claims it did
    manage to reject, instead of the whole pass failing open."""
    try:
        return _Verdicts.model_validate(json.loads(content or "")).verdicts
    except (json.JSONDecodeError, pydantic.ValidationError):
        salvaged = [
            ClaimVerdict(index=int(i), supported=(s == "true"))
            for i, s in _VERDICT_RE.findall(content or "")
        ]
        if not salvaged:
            raise  # nothing usable — let the caller fail open
        logger.warning("groundedness judge JSON truncated — salvaged %d verdict(s)", len(salvaged))
        return salvaged


def _page_texts(urls: set[str]) -> dict[str, str]:
    """url → that page's stored text, capped. The judge reads the SAME evidence
    the agent had — not a re-crawl (deterministic, no network, no drift)."""
    texts: dict[str, list[str]] = {}
    for c in store.all_chunks():
        if c["url"] in urls:
            if c["url"] not in texts:
                # CTA/heading METADATA is evidence the agent saw via read_page
                # (it's stripped from body text as boilerplate) — the judge must
                # see it too, or it drops TRUE claims about signup buttons
                # (caught live: judged a real 'Get Started Now' CTA unsupported).
                header = []
                if c.get("ctas"):
                    header.append(f"[primary actions on this page: {c['ctas']}]")
                if c.get("headings"):
                    header.append(f"[sections: {c['headings']}]")
                texts[c["url"]] = header
            texts[c["url"]].append(c["text"])
    return {u: "\n".join(parts)[:_PAGE_EXCERPT_CHARS] for u, parts in texts.items()}


def verify_groundedness(report: FirstImpressionReport) -> FirstImpressionReport:
    """Drop observations whose cited page does not support the claim.

    Called by: report.apply_guards(), after citation verification (so every
    source_url here is a real page). One LLM call, fail-open on errors.
    """
    if not settings.groundedness_judge:
        return report

    # Collect (field, obs) pairs with a stable index the judge echoes back.
    indexed: list[tuple[str, object]] = [
        (field, obs) for field in _JUDGED_FIELDS for obs in getattr(report, field)
    ]
    if not indexed:
        return report

    pages = _page_texts({obs.source_url for _, obs in indexed})
    claims_block = "\n".join(
        f"[{i}] claim: {obs.claim!r} | evidence quoted: {obs.evidence!r} | cited page: {obs.source_url}"
        for i, (_, obs) in enumerate(indexed)
    )
    pages_block = "\n\n".join(f"=== SOURCE PAGE {u} ===\n{t}" for u, t in pages.items())
    prompt = f"{_JUDGE_INSTRUCTION}\n\nCLAIMS:\n{claims_block}\n\n{pages_block}"

    try:
        # Pool with the finalized chain (prefer = settings.pool_prefer, GLM-led).
        message = llm_pool.chat(
            [
                {
                    "role": "system",
                    "content": 'Reply ONLY with JSON: {"verdicts": [{"index": int, '
                    '"supported": bool}, ...]} — one entry per claim, no other keys.',
                },
                {"role": "user", "content": prompt},
            ],
            prefer=settings.pool_prefer,
            response_format={"type": "json_object"},
            label="groundedness-judge",
        )
        verdicts = _parse_verdicts(message.content or "")
    except Exception as exc:  # fail-open: judge is a bonus layer, not a gate
        logger.warning("groundedness judge skipped (fail-open): %s", exc)
        return report

    unsupported = {
        v.index for v in verdicts if not v.supported and 0 <= v.index < len(indexed)
    }
    if unsupported:
        logger.warning(
            "groundedness judge dropped %d claim(s): %s",
            len(unsupported),
            [indexed[i][1].claim for i in sorted(unsupported)],
        )
        for field in _JUDGED_FIELDS:
            kept = [
                obs
                for i, (f, obs) in enumerate(indexed)
                if f == field and i not in unsupported
            ]
            setattr(report, field, kept)
    return report
