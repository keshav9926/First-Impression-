"""
===============================================================================
FILE: app/agent/tools.py
ORIGIN      : app.agent.react (run_react_loop)
PURPOSE     : ReAct Agent tool registry definitions (list_pages, read_page, search_content)
DESTINATION : app.rag.store / app.rag.pipeline (Queries underlying RAG indices)
===============================================================================
"""

import json
import logging
from urllib.parse import urlparse

from google.genai import types

from app.config import settings
from app.rag import pipeline, store

logger = logging.getLogger("first_impression")

# Tool outputs are BOUNDED so the ReAct history can't grow past a provider's
# per-request context/token budget (the whole conversation is resent every
# step). If a page is truncated, the agent is told to use search_content to dig
# into specifics instead.
# Raised 4000 → 15000 (2026-08-01) on measurement: vortexify's 14 non-docs pages
# total 36,092 chars — the agent can read EVERY one of them in full for ~9K
# tokens. At 4000 it saw only the first 23% of the pages that mattered, and on
# a real page that opening slice is nav and hero copy. The one page this cap
# could not have covered (/docs, 61,920 chars) is now split into section pages
# upstream, so nothing reaching here needs truncating. Groq's ~12K tokens/minute
# free tier was the original reason for 4000; the pool is NVIDIA-only now.
READ_PAGE_MAX_CHARS = 15000
SEARCH_TOP_K = 3
SEARCH_SNIPPET_CHARS = 1200
# min_relevance is a single threshold tuned on ONE site — a real topic that
# scores just under it would otherwise read as a hard "not covered", which the
# agent turns into a FALSE "unanswered question". If the best match lands within
# this margin below the bar, we report it as UNCERTAIN instead of absent.
SEARCH_NEAR_MISS_MARGIN = 0.10


def _list_pages() -> str:
    """Observation for list_pages(): the distinct pages available to analyze.

    If ingestion flagged thin extraction (JS-rendered site — static crawl saw
    only a fraction of the real content), the agent is warned HERE, on its
    very first tool call, so it never converts our blindness into confident
    "the site doesn't mention X" findings."""
    chunks = store.all_chunks()
    urls = sorted({c["url"] for c in chunks})
    if not urls:
        return "No pages have been ingested."
    warning = ""
    if any(c.get("extraction_warning") for c in chunks):
        warning = (
            "WARNING: this site appears to be JavaScript-rendered — the crawler "
            "captured only a small fraction of its real content. Content that "
            "seems missing may simply be unread. Do NOT report 'the site does "
            "not mention X' as a friction point or unanswered question; only "
            "describe what you POSITIVELY observed.\n\n"
        )
    # A page too big to read in one call is stored as one page per heading
    # section (url#anchor). Those are VIEWS of their parent, not separate
    # pages — listed flat, a 15-page site reads as 50 pages, which is a false
    # impression of exactly the thing this report describes. So: nest them, and
    # state both counts.
    sections_by_page: dict[str, list[str]] = {}
    for u in urls:
        base, _, fragment = u.partition("#")
        sections_by_page.setdefault(base, [])
        if fragment:
            sections_by_page[base].append(u)
    lines = []
    for base in sorted(sections_by_page):
        lines.append(f"- {base}")
        for section in sections_by_page[base]:
            lines.append(f"    - {section}")
    total_sections = sum(len(s) for s in sections_by_page.values())
    header = f"Pages available to analyze ({len(sections_by_page)} pages"
    header += (
        f"; the indented entries are sections of the page above them — "
        f"{total_sections} of them, split out because that page was too long to "
        f"read in one call. Read them like any other page):\n"
        if total_sections
        else "):\n"
    )
    return warning + header + "\n".join(lines)


def resolve_page_url(url: str, all_chunks: list[dict] | None = None) -> str | None:
    """The INGESTED page a read_page(url) argument actually refers to, or None
    if there is no such page.

    Models often pass the wrong string instead of the exact URL list_pages
    returned — a bare slug ("pricing"), or a hallucinated placeholder domain
    ("https://example.com/pricing"). Recover instead of wasting a step by
    matching on the LAST path segment against stored URLs. Only an UNAMBIGUOUS
    match is accepted; ambiguous → None, and the caller asks for the exact URL.

    Shared by _read_page and by pages_from_steps, which reports how much of the
    site was really read: a model that guesses /about, /blog and /careers on a
    site with none of them must not have those counted as pages examined.
    """
    all_chunks = store.all_chunks() if all_chunks is None else all_chunks
    available = sorted({c["url"] for c in all_chunks})
    if url in available:
        return url
    # Reduce whatever was passed to its last path segment: bare slug stays
    # itself; "https://example.com/pricing" → "pricing"; root/"home" → "".
    slug = urlparse(url).path.strip("/").split("/")[-1].lower() if "/" in url else url.strip().lower()
    if slug in ("", "home", "index"):
        # Root page = a URL with an empty path (scheme://host/).
        candidates = [u for u in available if not urlparse(u).path.strip("/")]
    else:
        candidates = [u for u in available if u.rstrip("/").lower().endswith("/" + slug)]
    return candidates[0] if len(candidates) == 1 else None


def _read_page(url: str) -> str:
    """Observation for read_page(url): the full readable text of one page.

    store.all_chunks() returns chunks in reading order (store.py sorts them),
    so concatenating a page's chunks reconstructs the page as a user reads it.
    """
    all_chunks = store.all_chunks()
    resolved = resolve_page_url(url, all_chunks)
    url = resolved or url
    page_chunks = [c for c in all_chunks if c["url"] == url]

    if not page_chunks:
        available = sorted({c["url"] for c in all_chunks})
        # This is a failed GUESS by the model, and it must not become evidence.
        # On vortexify (2026-08-02) the agent invented /about, /capabilities,
        # /how-it-works and /launch from footer LABELS, got this message six
        # times, and reported "several primary footer links lead to no
        # substantive page" as a friction point — a claim about the site built
        # entirely out of its own wrong urls. Nothing here says anything about
        # the site, so say so explicitly.
        return (
            f"No page found at {url!r} — this url was never part of this site's "
            "crawl, so it is almost certainly one you guessed rather than one "
            "that exists. This is NOT evidence about the site: do not report it "
            "as a broken link, a dead page, or missing content. Only the urls "
            "below exist; a page absent from that list was never fetched, which "
            "is not the same as the site not having it.\n"
            f"Use the EXACT url from list_pages. "
            "Available pages are:\n"
            + "\n".join(f"- {u}" for u in available)
        )
    body = "\n\n".join(c["text"] for c in page_chunks)
    if len(body) > READ_PAGE_MAX_CHARS:
        # The model is told (below) to use search_content for the rest, but the
        # cut is otherwise invisible to us — log it so we can see when a page is
        # too big for a single read to represent faithfully.
        logger.info(
            "read_page truncated %s: %d chars → %d", url, len(body), READ_PAGE_MAX_CHARS
        )
        # Section map: the cut hides everything past READ_PAGE_MAX_CHARS, and
        # the model can't search for content it never learned EXISTS (the
        # unknown-unknown). The page's own headings (~150 tokens) reveal the
        # full shape, so the model can search_content into any section it
        # never saw. Only shown when truncating — a fully-visible page needs
        # no map.
        headings = page_chunks[0].get("headings", "")
        section_map = f"Sections on this page: {headings}\n\n" if headings else ""
        body = (
            section_map
            + body[:READ_PAGE_MAX_CHARS]
            + "\n\n[... page truncated — it continues beyond what is shown. "
            "Use search_content('<section or topic>') to read any section "
            "listed above that you have not seen ...]"
        )
    # Primary actions (Sign up / Try free / Book a demo): these live in the
    # header/footer that trafilatura strips as boilerplate, so they are absent
    # from `body` — but they are the #1 signal for the "can I get started?"
    # question. Surfaced on EVERY read (not truncation-only): a missing signup
    # CTA is a real finding; a present one must not read as missing.
    ctas = page_chunks[0].get("ctas", "")
    cta_line = f"Primary actions available on this page: {ctas}\n\n" if ctas else ""
    # Visual evidence: the text extractor cannot read images, so without this
    # line models "observe" that no screenshots/videos exist on pages full of
    # them (caught live: vortexify.ai dashboard shots reported as missing).
    images = page_chunks[0].get("images", "")
    img_line = (
        f"Visuals on this page (labels + vision-model descriptions where read): {images}\n\n"
        if images
        else "No substantive images detected on this page.\n\n"
    )
    return f"Text of {url}:\n\n{cta_line}{img_line}{body}"


def _search_content(query: str) -> str:
    """Observation for search_content(query): the most relevant chunks, cited.

    Uses the SAME hybrid funnel as /ask (rag/pipeline.retrieve) and the SAME
    relevance gate — so if nothing clears the bar, the agent is told plainly
    that the site does not cover this, which is itself a useful finding
    (it becomes an 'unanswered question' in the report)."""
    hits = pipeline.retrieve(query, top_k=SEARCH_TOP_K)
    relevant = [h for h in hits if h["relevance"] >= settings.min_relevance]
    if not relevant:
        top = hits[0]["relevance"] if hits else 0.0
        if top >= settings.min_relevance - SEARCH_NEAR_MISS_MARGIN:
            # Borderline: don't assert the site ignores this — say so honestly
            # so it doesn't become a false "unanswered question".
            return (
                f"No STRONGLY relevant content for {query!r} (best match {top:.2f}, "
                f"just under the {settings.min_relevance:.2f} bar). The site may "
                "touch on this weakly — treat as uncertain, not a confirmed gap."
            )
        return f"No content relevant to {query!r} was found in the ingested pages."
    return "\n\n".join(
        f"[relevance {h['relevance']:.2f}] (from {h['url']})\n{h['text'][:SEARCH_SNIPPET_CHARS]}"
        for h in relevant
    )


# --- Repeat-call guard, shared by both drivers (react.py + groq_driver.py) ---
# The store is frozen during a run and every tool is deterministic, so calling
# the same tool with the same args twice can only waste a step and re-add the
# same tokens to the resent history. Instead of re-executing, the agent gets a
# short reminder — the original observation is already in its history.

def repeat_call_reminder(name: str, args: dict, seen: set) -> str | None:
    """Return a reminder string if (name, args) was already executed this run,
    else record it in `seen` and return None (meaning: go ahead and execute).

    Called by: react.py loop and groq_driver.py loop, before execute_tool().
    `seen` is created fresh per run by the caller — module state would leak
    between requests.
    """
    key = (name, json.dumps(args, sort_keys=True))
    if key in seen:
        return (
            f"You already called {name} with these exact arguments — the result "
            "is unchanged and already in your context. Call a DIFFERENT tool or "
            "different arguments, or stop and write the report."
        )
    seen.add(key)
    return None


# --- Dispatcher: map a model-chosen tool name + args to the right impl ---

def execute_tool(name: str, args: dict) -> str:
    """Run the tool the model asked for; return its observation string.

    Called by: react.py, for every function_call the model emits.
    Unknown names return an error string (not an exception) so a model
    hallucinating a tool name gets corrected instead of crashing the loop.
    """
    if name == "list_pages":
        return _list_pages()
    if name == "read_page":
        return _read_page(args.get("url", ""))
    if name == "search_content":
        return _search_content(args.get("query", ""))
    return f"Unknown tool {name!r}. Available: list_pages, read_page, search_content."


# --- Neutral tool metadata (name, description, JSON-schema params) ---
# Both provider schemas below are built from these, so the tool surface is
# defined ONCE and can't drift between Gemini and Groq.
_TOOLS = [
    {
        "name": "list_pages",
        "description": (
            "List every public page available to analyze. Call this FIRST to "
            "see what the site contains before deciding what to read."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_page",
        "description": (
            "Read the full text of one page, exactly as a prospective user "
            "would. Use the URLs returned by list_pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The exact page URL to read."}
            },
            "required": ["url"],
        },
    },
    {
        "name": "search_content",
        "description": (
            "Search across ALL pages for content on a specific topic (e.g. "
            "'onboarding steps', 'pricing', 'customer support', 'security'). "
            "Use this to check whether the site covers something a new user "
            "would look for. If it returns nothing, the site likely does not "
            "address that topic — a useful finding in itself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain language."}
            },
            "required": ["query"],
        },
    },
]


# --- Groq / OpenAI-compatible tool schema (built from _TOOLS) ---
OPENAI_TOOLS = [
    {"type": "function", "function": t} for t in _TOOLS
]


# --- Gemini function-declaration schema (built from _TOOLS) ---

_GEMINI_TYPE = {"string": types.Type.STRING}


def _gemini_schema(params: dict) -> types.Schema:
    """Convert a JSON-schema params dict into a Gemini types.Schema."""
    properties = {
        name: types.Schema(type=_GEMINI_TYPE[spec["type"]], description=spec.get("description"))
        for name, spec in params.get("properties", {}).items()
    }
    return types.Schema(
        type=types.Type.OBJECT,
        properties=properties,
        required=params.get("required", []),
    )


FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name=t["name"], description=t["description"], parameters=_gemini_schema(t["parameters"])
    )
    for t in _TOOLS
]
