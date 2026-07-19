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

# Per cited page. Raised from 3000 → 8000 (2026-07-18): on the NVIDIA chain the
# judge has ample context, and a true claim grounded DEEPER than the old 3000
# cut was being judged "unsupported" and wrongly dropped. More page text = the
# judge sees what it's checking against.
_PAGE_EXCERPT_CHARS = 8000
_JUDGED_FIELDS = (
    "what_the_product_is",
    "likely_new_user_journey",
    "friction_points",
    "standout_strengths",
    # improvement_opportunities are opinion, but each cites a real page premise
    # (`observed`) — fact-check that premise too, so advice can't rest on an
    # unsupported observation. (Its item type has .suggestion/.observed instead
    # of .claim/.evidence — _claim_evidence normalizes both.)
    "improvement_opportunities",
)

_JUDGE_INSTRUCTION = """\
You are a strict fact-checker. For every numbered claim below, decide whether
the SOURCE PAGE TEXT actually supports it. A claim is supported only if the
page text states or clearly implies it — plausible-sounding is NOT enough.
Claims about something MISSING/unclear are supported when the page text indeed
does not show it. Return a verdict for EVERY claim, in order."""

# Persona impressions and unanswered questions carry no citations, so
# "supported" is the wrong test — instead ask whether ANY page text
# CONTRADICTS them. Caught live (2026-07-19, trynarrative.com): personas said
# "no SOC 2 mentioned" and a question asked about SOC 2 while the homepage
# said "SOC 2 Type II audit (in progress)" — a false negative the founder
# would spot instantly. Contradicted statements are DROPPED.
_STMT_INSTRUCTION = """\
Additionally, for every numbered STATEMENT below (uncited persona impressions
and open questions), decide whether the source page text CONTRADICTS it. A
statement that something is missing/absent/"not mentioned"/unanswered is
contradicted when the page text shows it present IN ANY FORM — a partial or
in-progress mention counts (e.g. "no SOC 2 mentioned" IS contradicted by
"SOC 2 audit in progress"; "no free trial" IS contradicted by "start your
free trial"; "no screenshots" IS contradicted by an [images/videos ...]
metadata line listing product images). Statements about genuine absences stay
uncontradicted. When truly uncertain, it is NOT contradicted. Return a verdict
for EVERY statement, in order."""


class ClaimVerdict(pydantic.BaseModel):
    index: int  # the claim's number in the prompt
    supported: bool
    reason: str = ""  # optional — we never read it; kept so old payloads validate


class StatementVerdict(pydantic.BaseModel):
    index: int  # the statement's number in the prompt
    contradicted: bool
    reason: str = ""


class _Verdicts(pydantic.BaseModel):
    verdicts: list[ClaimVerdict]
    statement_verdicts: list[StatementVerdict] = []


# Salvage complete "index"+"supported" pairs from a response body — used when
# the model's JSON is truncated (a completion-token cap can cut the array
# mid-object). We only ever need index + supported, so counting the COMPLETE
# pairs recovers every finished verdict and safely ignores the cut-off tail.
_VERDICT_RE = re.compile(r'"index"\s*:\s*(\d+)\s*,\s*"supported"\s*:\s*(true|false)')
_STMT_RE = re.compile(r'"index"\s*:\s*(\d+)\s*,\s*"contradicted"\s*:\s*(true|false)')


def _parse_verdicts(content: str) -> tuple[list[ClaimVerdict], list[StatementVerdict]]:
    """Parse the judge reply into (claim, statement) verdicts, tolerating a
    truncated array.

    Strict json.loads first (the happy path); if that fails — almost always an
    output-token cut mid-string — fall back to regex-salvaging every COMPLETE
    pair. A partial judge result still drops the claims it did manage to
    reject, instead of the whole pass failing open."""
    try:
        parsed = _Verdicts.model_validate(json.loads(content or ""))
        return parsed.verdicts, parsed.statement_verdicts
    except (json.JSONDecodeError, pydantic.ValidationError):
        claims = [
            ClaimVerdict(index=int(i), supported=(s == "true"))
            for i, s in _VERDICT_RE.findall(content or "")
        ]
        stmts = [
            StatementVerdict(index=int(i), contradicted=(s == "true"))
            for i, s in _STMT_RE.findall(content or "")
        ]
        if not claims and not stmts:
            raise  # nothing usable — let the caller fail open
        logger.warning(
            "groundedness judge JSON truncated — salvaged %d claim / %d statement verdict(s)",
            len(claims), len(stmts),
        )
        return claims, stmts


def _claim_evidence(obs: object) -> tuple[str, str]:
    """Normalize the (claim, evidence) pair across the two judged item types:
    an Observation carries .claim/.evidence; an ImprovementOpportunity carries
    .suggestion/.observed (its `observed` IS the grounded premise to check)."""
    claim = getattr(obs, "claim", None) or getattr(obs, "suggestion", "")
    evidence = getattr(obs, "evidence", None) or getattr(obs, "observed", "")
    return claim, evidence


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
                # Same rationale as CTAs: visuals are stripped from body text,
                # so the judge must see they exist or it upholds false
                # "no screenshot/video" claims (vortexify.ai, 2026-07-19).
                if c.get("images"):
                    header.append(f"[images/videos on this page: {c['images']}]")
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

    # Uncited statements: persona impressions + unanswered questions. These are
    # contradiction-checked against ALL stored pages (they cite nothing, so
    # "which page supports this" does not apply — but "which page disproves
    # this" does).
    statements: list[tuple[str, int | None, int, str]] = []  # (kind, p_idx, item_idx, text)
    for p_idx, persona in enumerate(report.persona_panel):
        for item_idx, txt in enumerate(persona.what_resonated):
            statements.append(("resonated", p_idx, item_idx, txt))
        for item_idx, txt in enumerate(persona.friction):
            statements.append(("friction", p_idx, item_idx, txt))
    for q_idx, q in enumerate(report.unanswered_questions):
        statements.append(("question", None, q_idx, q))

    all_urls = {obs.source_url for _, obs in indexed}
    all_urls.update(c["url"] for c in store.all_chunks())
    pages = _page_texts(all_urls)
    claims_block = "\n".join(
        f"[{i}] claim: {c!r} | evidence quoted: {e!r} | cited page: {obs.source_url}"
        for i, (_, obs) in enumerate(indexed)
        for c, e in (_claim_evidence(obs),)
    )
    stmts_block = "\n".join(f"[{i}] {txt!r}" for i, (_, _, _, txt) in enumerate(statements))
    pages_block = "\n\n".join(f"=== SOURCE PAGE {u} ===\n{t}" for u, t in pages.items())
    prompt = (
        f"{_JUDGE_INSTRUCTION}\n\nCLAIMS:\n{claims_block}\n\n"
        f"{_STMT_INSTRUCTION}\n\nSTATEMENTS:\n{stmts_block}\n\n{pages_block}"
    )

    try:
        # Pool with the finalized chain (prefer = settings.pool_prefer, GLM-led).
        message = llm_pool.chat(
            [
                {
                    "role": "system",
                    "content": 'Reply ONLY with JSON: {"verdicts": [{"index": int, '
                    '"supported": bool}, ...], "statement_verdicts": [{"index": int, '
                    '"contradicted": bool}, ...]} — one verdicts entry per claim, one '
                    "statement_verdicts entry per statement, no other keys.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            # Enough tokens for one small verdict per claim so the JSON array is
            # not cut mid-stream (truncated tails default to "kept" — see
            # _parse_verdicts — so avoiding the cut avoids keeping unverified
            # claims). The NVIDIA chain has the headroom.
            # 8000 (was 4000): failover models (e.g. Nemotron) spend "thinking"
            # tokens from this same budget before the JSON — 4000 could truncate
            # verdicts on long claim lists, and truncated tails default to KEPT.
            max_tokens=8000,
            # A fact-checker must be repeatable: at default temperature the
            # same claims got different verdicts run-to-run (observed live).
            temperature=0.0,
            label="groundedness-judge",
        )
        verdicts, stmt_verdicts = _parse_verdicts(message.content or "")
    except Exception as exc:  # fail-open: judge is a bonus layer, not a gate
        # But SURFACE it — a silent skip lets an unverified report look verified.
        # The reader must know the automated fact-check did not run this time.
        logger.warning("groundedness judge skipped (fail-open): %s", exc)
        report.scope_note = (
            report.scope_note.rstrip(".")
            + ". NOTE: the automated groundedness fact-check could not run for this "
            "report (the judge model was unavailable), so claims are cited but not "
            "double-verified against their source pages — treat them accordingly."
        )
        return report

    unsupported = {
        v.index for v in verdicts if not v.supported and 0 <= v.index < len(indexed)
    }
    if unsupported:
        logger.warning(
            "groundedness judge dropped %d claim(s): %s",
            len(unsupported),
            [_claim_evidence(indexed[i][1])[0] for i in sorted(unsupported)],
        )
        for field in _JUDGED_FIELDS:
            kept = [
                obs
                for i, (f, obs) in enumerate(indexed)
                if f == field and i not in unsupported
            ]
            setattr(report, field, kept)

    # Drop contradicted persona impressions / questions (false negatives like
    # "no SOC 2 mentioned" when the page says otherwise).
    contradicted = {
        v.index for v in stmt_verdicts if v.contradicted and 0 <= v.index < len(statements)
    }
    if contradicted:
        logger.warning(
            "groundedness judge dropped %d contradicted statement(s): %s",
            len(contradicted),
            [statements[i][3] for i in sorted(contradicted)],
        )
        drop = {(k, p, j) for i in contradicted for k, p, j, _ in (statements[i],)}
        for p_idx, persona in enumerate(report.persona_panel):
            persona.what_resonated = [
                t for j, t in enumerate(persona.what_resonated)
                if ("resonated", p_idx, j) not in drop
            ]
            persona.friction = [
                t for j, t in enumerate(persona.friction)
                if ("friction", p_idx, j) not in drop
            ]
        report.unanswered_questions = [
            q for j, q in enumerate(report.unanswered_questions)
            if ("question", None, j) not in drop
        ]
    return report
