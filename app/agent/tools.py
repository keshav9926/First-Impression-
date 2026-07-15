# app/agent/tools.py — the agent's three tools + their Gemini schemas.
#
# These are thin wrappers over machinery we already built — the agent doesn't
# get new powers, it gets the ABILITY TO CHOOSE which existing capability to
# use next. That choice, made by the model each step, is what makes this an
# agent rather than a pipeline.
#
#   list_pages()      → distinct URLs in the store   (survey the territory)
#   read_page(url)    → full text of one page        (read like a user)
#   search_content(q) → the Phase 2 hybrid funnel    (targeted questions)
#
# CALL FLOW:
#   react.py loop → execute_tool(name, args) → one of the _impl functions
#   Each returns a plain STRING — that string becomes the "observation" fed
#   back to the model as a function response. Tools never raise for "not
#   found"; they return a helpful message so the model can recover and re-plan
#   (a raised exception would just kill the loop).
#
# FUNCTION_DECLARATIONS is the schema list handed to Gemini so it knows what
# tools exist and how to call them (name, description, parameters).

from urllib.parse import urlparse

from google.genai import types

from app.config import settings
from app.rag import pipeline, store

# Tool outputs are BOUNDED so the ReAct history can't grow past a provider's
# per-request context/token budget (Groq free tier is only ~12K tokens/minute,
# and the whole conversation is resent every step). Bounded reads also make the
# agent cheaper and faster on every provider. If a page is truncated, the agent
# is told to use search_content to dig into specifics instead.
READ_PAGE_MAX_CHARS = 4000
SEARCH_TOP_K = 3
SEARCH_SNIPPET_CHARS = 1200


def _list_pages() -> str:
    """Observation for list_pages(): the distinct pages available to analyze."""
    urls = sorted({c["url"] for c in store.all_chunks()})
    if not urls:
        return "No pages have been ingested."
    return "Pages available to analyze:\n" + "\n".join(f"- {u}" for u in urls)


def _read_page(url: str) -> str:
    """Observation for read_page(url): the full readable text of one page.

    store.all_chunks() returns chunks in reading order (store.py sorts them),
    so concatenating a page's chunks reconstructs the page as a user reads it.
    """
    all_chunks = store.all_chunks()
    page_chunks = [c for c in all_chunks if c["url"] == url]

    # Models often pass a bare slug ("pricing", "home") instead of the exact
    # URL list_pages returned. Recover instead of wasting a step: match a URL
    # whose path ends with the slug ("home"/"" → the root URL). Only accept an
    # UNAMBIGUOUS match; if a slug hits several pages, ask for the exact URL.
    if not page_chunks:
        available = sorted({c["url"] for c in all_chunks})
        slug = url.strip().strip("/").lower()
        if slug in ("", "home", "index"):
            # Root page = a URL with an empty path (scheme://host/).
            candidates = [u for u in available if not urlparse(u).path.strip("/")]
        else:
            candidates = [u for u in available if u.rstrip("/").lower().endswith("/" + slug)]
        if len(candidates) == 1:
            url = candidates[0]
            page_chunks = [c for c in all_chunks if c["url"] == url]

    if not page_chunks:
        available = sorted({c["url"] for c in all_chunks})
        return (
            f"No page found at {url!r}. Use the EXACT url from list_pages. "
            "Available pages are:\n"
            + "\n".join(f"- {u}" for u in available)
        )
    body = "\n\n".join(c["text"] for c in page_chunks)
    if len(body) > READ_PAGE_MAX_CHARS:
        body = (
            body[:READ_PAGE_MAX_CHARS]
            + "\n\n[... page truncated — use search_content to find specific "
            "details on this page ...]"
        )
    return f"Text of {url}:\n\n{body}"


def _search_content(query: str) -> str:
    """Observation for search_content(query): the most relevant chunks, cited.

    Uses the SAME hybrid funnel as /ask (rag/pipeline.retrieve) and the SAME
    relevance gate — so if nothing clears the bar, the agent is told plainly
    that the site does not cover this, which is itself a useful finding
    (it becomes an 'unanswered question' in the report)."""
    hits = pipeline.retrieve(query, top_k=SEARCH_TOP_K)
    relevant = [h for h in hits if h["relevance"] >= settings.min_relevance]
    if not relevant:
        return f"No content relevant to {query!r} was found in the ingested pages."
    return "\n\n".join(
        f"[relevance {h['relevance']:.2f}] (from {h['url']})\n{h['text'][:SEARCH_SNIPPET_CHARS]}"
        for h in relevant
    )


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
