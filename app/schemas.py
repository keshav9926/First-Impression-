"""
===============================================================================
FILE: app/schemas.py
ORIGIN      : FastAPI routes (app.main) / Synthesis LLM (app.agent.report)
PURPOSE     : Strict Pydantic data validation schemas for HTTP API payloads & JSON reports
DESTINATION : FastAPI serialization layer / OpenAPI docs / Frontend UI
===============================================================================
"""

from pydantic import BaseModel, Field, HttpUrl


class IngestRequest(BaseModel):
    """
    POST /ingest Request Payload

    ORIGIN      : HTTP client / Web UI
    PURPOSE     : Validates website target URL and crawl depth parameters.
    DESTINATION : Handed to app.main.ingest() -> app.ingestion.crawler.crawl_site()
    """

    url: HttpUrl
    max_pages: int = Field(default=15, ge=1, le=300)


class IngestResponse(BaseModel):
    """What POST /ingest returns — a summary of what was crawled and stored.

    Built at the end of main.ingest(); skipped_by_robots comes from
    fetcher.CrawlResult so rule-#1 enforcement is visible to the caller.
    """

    pages_fetched: int
    chunks_stored: int
    skipped_by_robots: int
    # True when the site's text/HTML ratio says it is JS-rendered (SPA/Framer/
    # Webflow) and a static crawl captured only a fraction of the real content.
    # The caller should treat any downstream "the site doesn't mention X" with
    # suspicion — it may be OUR blindness, not the site's gap.
    extraction_warning: bool = False
    # Lines stripped by the prompt-injection guard (sanitize.py) — nonzero
    # means the site contained instruction-shaped text aimed at the LLM.
    injection_lines_removed: int = 0
    # Vision (Phase 9) visibility: how many product images were found across the
    # crawl and how many the vision model actually captioned. Both 0 means either
    # vision is off/unconfigured or the pages carried no product imagery.
    images_seen: int = 0
    images_captioned: int = 0
    # Same visibility for demo videos the VLM actually watched. Only downloadable
    # media files count here — third-party player embeds are captioned through
    # their poster frame and show up in the images_* counters instead.
    videos_seen: int = 0
    videos_captioned: int = 0


class AskRequest(BaseModel):
    """Body of POST /ask — the question, and how many chunks to retrieve.

    Validated by FastAPI before main.ask() runs.
    """

    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)  # how many chunks to retrieve


class Source(BaseModel):
    """One retrieved chunk shown to the caller as a citation.

    Nested inside AskResponse. `index` matches the [n] markers Claude
    writes in the answer text (numbering assigned in qa.answer()).
    """

    index: int
    url: str
    snippet: str


class AskResponse(BaseModel):
    """What POST /ask returns — the grounded answer plus its citations.

    Built at the end of main.ask(): `answer` from qa.answer(),
    `sources` from the same hits that were given to Claude.
    """

    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Phase 3: the structured First Impression report.
#
# This schema is doing TWO jobs at once:
#   1. It is the Gemini `response_schema` — the model must generate JSON that
#      fits it, so an Observation without a source_url is literally impossible
#      to produce. That is hard rule #2 (grounded output only) made STRUCTURAL,
#      not just requested in a prompt.
#   2. It is the FastAPI response_model for POST /report — so the /docs page
#      documents the exact report shape and the response is validated on the
#      way out.
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """One grounded finding. Every field is required — an observation cannot
    exist without the evidence and the page a founder can check it against."""

    claim: str  # e.g. "A prospective user cannot find pricing from the docs"
    evidence: str  # short quote/paraphrase from the site that supports the claim
    source_url: str  # where on the public site this was observed


class ImprovementOpportunity(BaseModel):
    """A friendly, OPTIONAL suggestion — explicitly our inference, NOT a grounded
    observation. Kept in its own type (and its own report field) so advice can
    never be mistaken for a cited fact: the wall between "what a visitor
    experiences" (Observation, defensible) and "what we'd gently suggest"
    (opinion) is what keeps the report both useful and trustworthy.

    Still grounded, though: `observed` names the real first-impression experience
    the suggestion responds to, and `source_url` is the page it came from — so a
    founder can trace the WHY even though the WHAT-TO-DO is our opinion. The
    source_url is citation-verified exactly like an Observation's."""

    observed: str  # the real first-impression experience this responds to
    suggestion: str  # the gentle, constructive idea — an invitation, not a verdict
    source_url: str  # the page the observation came from (verified against the store)


class PersonaImpression(BaseModel):
    """One persona's read of the SAME shared evidence (Phase 4 panel).
    Produced by a persona node in agent/panel.py — opinion derived from
    evidence, so fields are plain strings (no per-claim source_url; the
    evidence itself was already grounded + citation-verified upstream)."""

    persona: str  # e.g. "Technical Evaluator"
    what_resonated: list[str]  # what worked for THIS persona
    friction: list[str]  # where THIS persona hesitates/bounces
    would_sign_up: bool  # the persona's gut verdict after the first visit
    reason: str  # one-sentence why behind the verdict


class FirstImpressionReport(BaseModel):
    """The deliverable: a structured, observational read of a company's public
    product surface as experienced by a prospective new user. Generated by the
    agent in agent/report.py; every list holds cited Observations."""

    company: str
    what_the_product_is: list[Observation]
    likely_new_user_journey: list[Observation]  # what the public surface teaches, in order
    friction_points: list[Observation]  # unclear / missing / hard-to-find — described, not graded
    standout_strengths: list[Observation]  # what genuinely works well
    unanswered_questions: list[str]  # what a prospect CANNOT learn before signing up
    # Friendly, constructive suggestions — separate from the cited Observations
    # above so opinion never contaminates fact. Defaults to [] so the model may
    # honestly return none rather than invent filler.
    improvement_opportunities: list[ImprovementOpportunity] = Field(default_factory=list)
    # Phase 4 panel: attached PROGRAMMATICALLY by agent/panel.py (never asked
    # of the synthesis LLM — the impressions are already validated objects).
    persona_panel: list[PersonaImpression] = Field(default_factory=list)
    scope_note: str  # honest "public surface only, no login" disclaimer


class ReportResponse(BaseModel):
    """What POST /report returns — the report plus a little transparency about
    HOW the agent produced it (which pages it read, how many tool calls)."""

    report: FirstImpressionReport
    steps_taken: int
    pages_examined: list[str]
    tool_calls: list[str]  # human-readable trace of the agent's actions
