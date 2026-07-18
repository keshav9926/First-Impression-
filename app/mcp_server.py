# app/mcp_server.py — Phase 7: First Impression as an MCP server.
#
# Exposes the SAME pipeline the FastAPI app serves — crawl → ReAct report, and
# grounded Q&A — behind the Model Context Protocol, so any MCP client (Claude
# Desktop, Claude Code, an IDE) can call the analyzer as a tool. It is a second
# front door onto app/, NOT a reimplementation: every tool below delegates to
# the exact functions main.py's HTTP endpoints call (_ingest_site,
# generate_report, pipeline.retrieve + qa.answer), so the two can never drift.
#
# TRANSPORT: stdio (the standard for local MCP clients). The protocol owns
# stdout, so NOTHING here may print to it — logging goes to stderr, and the
# tools return structured dicts rather than writing anything.
#
# RUN:
#     uv run python -m app.mcp_server
#
# REGISTER (e.g. Claude Desktop / Claude Code MCP config):
#     "first-impression": {
#         "command": "uv",
#         "args": ["run", "python", "-m", "app.mcp_server"],
#         "cwd": "<absolute path to this repo>"
#     }
#
# The heavy dependencies (Voyage, the LLM pool, ChromaDB) are the same ones the
# API needs; a report call blocks for ~2-4 min under free-tier pacing, exactly
# as POST /report does.

import logging

from mcp.server.fastmcp import FastMCP

from app.agent.report import InsufficientEvidenceError, generate_report
from app.config import settings
from app.ingestion.robots import is_allowed
from app.main import _ingest_site
from app.rag import pipeline, qa, store

logger = logging.getLogger("first_impression.mcp")

mcp = FastMCP("first-impression")


def _report_keys_missing() -> str | None:
    """Return a human-readable reason if the server lacks a key the report path
    needs, else None. Mirrors main.py's _require_keys guards so an MCP client
    gets the same clear 'set this key' message a curl user would."""
    if not settings.voyage_api_key:
        return "VOYAGE_API_KEY is not set on the server — embeddings are required."
    if not settings.nvidia_api_key:
        return "NVIDIA_API_KEY is not set on the server — the report pipeline runs on the NVIDIA pool."
    return None


@mcp.tool()
def analyze_first_impression(url: str, max_pages: int = 15, panel: bool = True) -> dict:
    """Analyze a startup's public site and return a grounded First Impression report.

    Crawls the URL (robots.txt-respecting, public pages only), builds a hybrid
    semantic + keyword index, runs a ReAct agent that explores the content, and
    synthesizes a structured report in which every claim cites a source page.
    This is the end-to-end analyzer as a single tool call.

    Args:
        url: The company's public site or docs URL (e.g. "https://example.com").
        max_pages: Max pages to crawl (default 15, hard-capped server-side).
        panel: When true (default), also run the three-persona panel
            (technical evaluator / business buyer / first-time user) so the
            report shows who bounces where.

    Returns a dict with `status` and, on success, the full report plus
    `pages_examined`, `steps_taken`, and `tool_calls`. On failure returns
    `status: "error"` with a `reason` — never a fabricated report.
    """
    missing = _report_keys_missing()
    if missing:
        return {"status": "error", "reason": missing}

    max_pages = max(1, min(max_pages, settings.max_pages_hard_limit))

    try:
        if not is_allowed(url):
            return {
                "status": "error",
                "reason": "robots.txt disallows fetching this URL (public-data rule).",
            }
        _ingest_site(url, max_pages)
        report_obj, steps_log, pages_examined = generate_report(panel=panel)
    except InsufficientEvidenceError as exc:
        # Robots-blocked or dead crawl → too thin to ground. Refuse, don't invent.
        return {"status": "insufficient_evidence", "reason": str(exc)}
    except Exception as exc:  # rate limits, malformed synthesis, network — surface, don't crash
        logger.exception("analyze_first_impression failed for %s", url)
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    return {
        "status": "ok",
        "report": report_obj.model_dump(),
        "pages_examined": pages_examined,
        "steps_taken": len(steps_log),
        "tool_calls": [f"{s['tool']}({s['args']})" for s in steps_log],
    }


@mcp.tool()
def ask_ingested(question: str, top_k: int = 5) -> dict:
    """Answer a question about the most recently analyzed site, with citations.

    Uses the hybrid retrieval funnel (dense + BM25 → RRF → rerank → relevance
    gate) over whatever `analyze_first_impression` last ingested. If nothing
    clears the relevance bar the tool refuses rather than guess.

    Args:
        question: A natural-language question about the ingested site.
        top_k: How many ranked chunks to consider (default 5).

    Returns `status` plus, on success, `answer` and `sources` (numbered
    url/snippet pairs matching the [n] citations in the answer text).
    """
    if not settings.voyage_api_key:
        return {"status": "error", "reason": "VOYAGE_API_KEY is not set on the server."}
    if store.count() == 0:
        return {
            "status": "error",
            "reason": "Nothing ingested yet — call analyze_first_impression first.",
        }
    try:
        ranked = pipeline.retrieve(question, top_k=top_k)
    except Exception as exc:
        # FAIL CLOSED: if we can't verify relevance, refuse rather than degrade.
        logger.exception("ask_ingested retrieval failed")
        return {"status": "error", "reason": f"retrieval failed: {type(exc).__name__}: {exc}"}

    relevant = [hit for hit in ranked if hit["relevance"] >= settings.min_relevance]
    if not relevant:
        return {
            "status": "no_relevant_content",
            "answer": "The ingested content does not appear to contain information "
            "relevant to this question, so no grounded answer can be given.",
            "sources": [],
        }

    answer_text = qa.answer(question, relevant)
    sources = [
        {"index": i + 1, "url": hit["url"], "snippet": hit["text"][:200]}
        for i, hit in enumerate(relevant)
    ]
    return {"status": "ok", "answer": answer_text, "sources": sources}


@mcp.tool()
def ingestion_status() -> dict:
    """Report whether a site is currently ingested and available for `ask_ingested`.

    Returns `chunks_stored` (0 means nothing ingested yet) and the distinct
    source `pages` currently in the store."""
    try:
        chunks = store.all_chunks()
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    pages = sorted({c.get("url", "") for c in chunks if c.get("url")})
    return {"status": "ok", "chunks_stored": len(chunks), "pages": pages}


def main() -> None:
    """Entry point: serve the tools over stdio for a local MCP client."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
