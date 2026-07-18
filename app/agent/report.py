# app/agent/report.py — the report entry point + guards.
#
# generate_report() returns (FirstImpressionReport, steps_log, pages_examined).
# The report pipeline is NVIDIA-only (2026-07-18): both paths run on the pool
# (agent/groq_driver.py over settings.pool_prefer, the GLM-led NVIDIA chain).
#   panel=False → groq_driver.generate()  (single-agent explore → synthesize)
#   panel=True  → agent/panel.py run_panel() (three personas over one explore)
#
# EXPLORE (free-form ReAct) then SYNTHESIZE (schema-constrained JSON) is the
# shape in both. apply_guards() then runs the shared safety pass (citation
# verification → groundedness judge → thin-extraction caveat).
#
# CALL FLOW:
#   main.py report() → generate_report() → groq_driver.generate() / run_panel()

from app import observability
from app.agent import grounding, groq_driver, judge
from app.rag import store
from app.schemas import FirstImpressionReport

# Below this much stored text a report cannot be grounded — refuse rather than
# hallucinate. A robots-blocked or dead crawl stores ~0 chars, yet a synthesis
# LLM will still invent a plausible, fully-fabricated report from empty
# evidence (observed directly). /ask already 409s on an empty store; /report
# must refuse too. 200 chars ≈ a couple of sentences — anything real clears it;
# genuinely-thin-but-present sites are handled separately by the thin caveat.
_MIN_EVIDENCE_CHARS = 200


class InsufficientEvidenceError(ValueError):
    """Raised when the store holds too little to ground a report. Subclasses
    ValueError so older callers still catch it; main.py maps it to HTTP 409
    (a state problem — 'ingest first' — not a 502 synthesis failure)."""


def apply_guards(report: FirstImpressionReport) -> FirstImpressionReport:
    """The post-synthesis safety pass, shared by every path (both single-agent
    drivers AND the Phase 4 panel):
    - citation verification: the synthesis LLM GENERATES source_urls; drop any
      observation/suggestion citing a non-ingested page (rule #2 structural).
    - thin-extraction caveat appended IN CODE (not trusted to the LLM): if the
      crawl captured only a fraction of a JS-rendered site, every reader must
      see that "not found" may mean "not read".
    - groundedness judge (Phase 5): one LLM pass that drops claims the cited
      page does not actually support (citations only prove the URL exists).
    """
    all_chunks = store.all_chunks()
    valid_urls = sorted({c["url"] for c in all_chunks})
    report, _dropped = grounding.enforce_citations(report, valid_urls)
    report = judge.verify_groundedness(report)

    if any(c.get("extraction_warning") for c in all_chunks):
        report.scope_note = (
            report.scope_note.rstrip(".")
            + ". IMPORTANT: this site appears to be JavaScript-rendered and the "
            "crawler captured only a small fraction of its content — statements "
            "about missing or unaddressed topics may reflect the crawler's "
            "limitation, not the site."
        )
    return report


def generate_report(panel: bool = False) -> tuple[FirstImpressionReport, list[dict], list[str]]:
    """Produce the structured report using the configured agent provider.

    Called by: main.py report(). Returns (report, steps_log, pages_examined).
    May raise provider rate-limit errors — the endpoint maps them to HTTP codes.
    panel=True runs the Phase 4 LangGraph persona panel (explore once → three
    personas in parallel → merged report with persona_panel attached).

    Refuses (ValueError) when the store holds too little to ground a report —
    the agent would otherwise "explore" nothing and the synthesis LLM would
    fabricate a report from empty evidence. main.py maps this to HTTP 409.
    """
    chunks = store.all_chunks()
    total_chars = sum(len(c.get("text") or "") for c in chunks)
    if not chunks or total_chars < _MIN_EVIDENCE_CHARS:
        raise InsufficientEvidenceError(
            f"Not enough ingested content to ground a report ({len(chunks)} chunks, "
            f"{total_chars} chars). Ingest a crawlable public site first — a grounded "
            "report cannot be produced from empty evidence, and must not be hallucinated."
        )

    # Phase 8: trace the whole run to Langfuse (no-op unless configured). Every
    # LLM call below — explore, personas, judge, synthesis — nests under this
    # span automatically via llm_pool.chat's record_generation.
    pages = sorted({c.get("url", "") for c in chunks if c.get("url")})
    with observability.report_trace(panel=panel, chunks=len(chunks), pages=len(pages)):
        observability.update_trace_io(input={"pages": pages, "panel": panel})

        if panel:
            from app.agent.panel import run_panel  # local: langgraph import stays optional

            report, steps_log, pages_examined = run_panel()
        else:
            # Single-agent path: explore + synthesize on the NVIDIA pool
            # (groq_driver is the OpenAI-compat driver; despite the legacy name
            # it runs on settings.pool_prefer, the GLM-led NVIDIA chain).
            report, steps_log, pages_examined = groq_driver.generate()

        if not panel:
            # Single-agent path: the synthesis LLM may have fabricated a panel
            # from the schema — real impressions come only from the panel graph.
            report.persona_panel = []
        final = apply_guards(report), steps_log, pages_examined
        observability.update_trace_io(
            output={"company": report.company, "pages_examined": pages_examined}
        )
        return final
