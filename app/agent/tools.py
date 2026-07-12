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

from google.genai import types

from app.config import settings
from app.rag import pipeline, store


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
    if not page_chunks:
        available = sorted({c["url"] for c in all_chunks})
        return (
            f"No page found at {url!r}. Available pages are:\n"
            + "\n".join(f"- {u}" for u in available)
        )
    body = "\n\n".join(c["text"] for c in page_chunks)
    return f"Full text of {url}:\n\n{body}"


def _search_content(query: str) -> str:
    """Observation for search_content(query): the most relevant chunks, cited.

    Uses the SAME hybrid funnel as /ask (rag/pipeline.retrieve) and the SAME
    relevance gate — so if nothing clears the bar, the agent is told plainly
    that the site does not cover this, which is itself a useful finding
    (it becomes an 'unanswered question' in the report)."""
    hits = pipeline.retrieve(query, top_k=5)
    relevant = [h for h in hits if h["relevance"] >= settings.min_relevance]
    if not relevant:
        return f"No content relevant to {query!r} was found in the ingested pages."
    return "\n\n".join(
        f"[relevance {h['relevance']:.2f}] (from {h['url']})\n{h['text']}" for h in relevant
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


# --- Schemas handed to Gemini so it knows the tool surface ---

FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="list_pages",
        description=(
            "List every public page available to analyze. Call this FIRST to "
            "see what the site contains before deciding what to read."
        ),
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="read_page",
        description=(
            "Read the full text of one page, exactly as a prospective user "
            "would. Use the URLs returned by list_pages."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "url": types.Schema(
                    type=types.Type.STRING,
                    description="The exact page URL to read (from list_pages).",
                )
            },
            required=["url"],
        ),
    ),
    types.FunctionDeclaration(
        name="search_content",
        description=(
            "Search across ALL pages for content on a specific topic (e.g. "
            "'onboarding steps', 'pricing', 'customer support', 'security'). "
            "Use this to check whether the site covers something a new user "
            "would look for. If it returns nothing, the site likely does not "
            "address that topic — a useful finding in itself."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="What to look for, in plain language.",
                )
            },
            required=["query"],
        ),
    ),
]
