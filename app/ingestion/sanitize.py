"""
===============================================================================
FILE: app/ingestion/sanitize.py
ORIGIN      : app.main (ingest_site)
PURPOSE     : Prompt-injection guard stripping malicious instruction lines from crawled text
DESTINATION : app.ingestion.chunker (Provides clean, sanitized text)
===============================================================================
"""

import logging
import re

logger = logging.getLogger("first_impression")

# Lines matching any of these are dropped. Case-insensitive. Deliberately
# NARROW: each pattern is instruction-shaped language that has no business in
# legitimate product copy — a broad net would eat real content (false
# positives cost accuracy, our #1 goal).
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any |your |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
        r"disregard (all |any |your |the )?(previous|prior|above|earlier|system)",
        r"forget (all |any |your |the )?(previous|prior|above|earlier) instructions",
        r"you are now\b",
        r"new (system )?instructions?:",
        # Bare "system prompt" stripped legit copy on AI-product sites (docs
        # that MENTION prompts). Require an instruction verb around it.
        r"(ignore|disregard|forget|reveal|override|leak|print|repeat).{0,30}system prompt",
        r"\bact as\b.{0,40}\b(instead|now)\b",
        r"do not (mention|reveal|tell).{0,40}(user|instructions|prompt)",
        r"(respond|reply|answer) (only )?with\b.{0,60}(regardless|no matter)",
        r"rate (this|the) (product|site|company) as\b",
        r"describe (this|the) (product|site|company) as\b.{0,40}(best|perfect|excellent)",
    )
]


def sanitize_text(text: str) -> tuple[str, list[str]]:
    """Strip instruction-shaped lines from untrusted page text.

    Called by: main.py ingest(), once per crawled page (BEFORE chunking, so
    poisoned lines never get embedded/stored/retrieved).
    Returns (clean_text, removed_lines). Removals are logged with content —
    a site trying to manipulate the report is itself a finding worth seeing.
    """
    kept: list[str] = []
    removed: list[str] = []
    for line in text.split("\n"):
        if any(p.search(line) for p in _INJECTION_PATTERNS):
            removed.append(line.strip())
        else:
            kept.append(line)
    if removed:
        logger.warning("prompt-injection guard removed %d line(s): %r", len(removed), removed)
    return "\n".join(kept), removed
