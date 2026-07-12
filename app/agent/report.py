# app/agent/report.py — orchestrates the two phases that produce the report.
#
#   Phase A — EXPLORE (free-form ReAct): the agent uses list_pages / read_page
#             / search_content to gather everything it needs. Open-ended.
#   Phase B — SYNTHESIZE (constrained): one final call over the SAME history,
#             this time forced to output JSON matching FirstImpressionReport.
#             No tools here — just turn the gathered evidence into the schema.
#
# Splitting it this way keeps each phase good at one thing: exploration wants
# freedom (decide what to read), synthesis wants structure (every claim cited).
# We verified live that Gemini can reuse a tool-call history for a schema-
# constrained call (scratchpad/gemini_smoke2.py) before building this.
#
# CALL FLOW:
#   main.py report() → generate_report()
#       ├── run_react_loop(...)   Phase A (agent/react.py + tools.py)
#       └── generate_content(response_schema=FirstImpressionReport)  Phase B
#   Returns (report, steps_log, pages_examined) to the endpoint.
#
# GEMINI-SPECIFIC: this agent uses Gemini function calling. The endpoint guards
# that llm_provider == "gemini"; a Claude tool-use variant could live alongside
# this later without changing the endpoint.

from google import genai
from google.genai import types

from app.agent import tools
from app.agent.llm import generate_with_retry
from app.agent.react import run_react_loop
from app.config import settings
from app.schemas import FirstImpressionReport

MAX_STEPS = 12  # generous for a startup site (survey + read a handful + a few searches)

EXPLORE_SYSTEM = """\
You are a product analyst. You examine a company's PUBLIC website exactly as a \
prospective new user would — someone deciding whether to sign up, who has NOT \
logged in and cannot see anything behind a signup wall.

Your goal is to understand the FIRST IMPRESSION the public site gives a new user:
- what the product actually is and who it is for,
- the journey a new user is guided through (what they learn, in what order),
- friction points: things that are unclear, missing, or hard to find,
- genuine strengths: what the site communicates well,
- and what a prospective user still CANNOT learn before signing up.

How to work:
1. Call list_pages first to see what exists.
2. read_page the important pages (home, product/feature pages, pricing, docs).
3. Use search_content to check for things a new user looks for but you haven't \
seen — e.g. "getting started steps", "pricing", "customer support", "security", \
"integrations". If a search returns nothing, the site likely doesn't cover it — \
note that; it's a real finding.

Rules:
- Ground every eventual claim in what you actually read. Do not invent or assume.
- Be OBSERVATIONAL, never judgmental. Describe what a user would experience \
("a new user may not find pricing without submitting a form"), do not grade or \
attack ("the pricing is bad").
- Distinguish normal troubleshooting/reference docs from genuine new-user \
friction — a documented error message is not itself a product shortcoming.

When you have gathered enough to write the report, stop calling tools and say so."""

SYNTHESIZE_INSTRUCTION = """\
Now produce the First Impression report from what you gathered.

For every Observation, include: the claim, a short piece of evidence (a brief \
quote or paraphrase of what the site actually says), and the source_url where \
you observed it. An observation with no supporting evidence must be omitted.

- friction_points: describe the experience gap observationally (unclear/missing/\
hard-to-find), not as criticism. Do NOT list normal troubleshooting docs as \
friction.
- unanswered_questions: concrete things a prospective user CANNOT learn from the \
public site before signing up.
- scope_note: one honest sentence stating this analysis covers only the public, \
pre-signup surface (no authenticated/in-product experience)."""


def _client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def generate_report() -> tuple[FirstImpressionReport, list[dict], list[str]]:
    """Run explore → synthesize and return the structured report.

    Returns (report, steps_log, pages_examined). May raise voyageai.error.*
    from the search_content tool — the endpoint maps those to HTTP codes.
    """
    client = _client()

    # ---- Phase A: explore ----
    explore_config = types.GenerateContentConfig(
        system_instruction=EXPLORE_SYSTEM,
        tools=[types.Tool(function_declarations=tools.FUNCTION_DECLARATIONS)],
        # Manual calling: WE run the tools and log each step (see react.py).
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text="Analyze this company's public site and prepare its report.")],
        )
    ]
    contents, steps_log = run_react_loop(
        client, settings.gemini_agent_model, contents, explore_config, MAX_STEPS
    )

    # ---- Phase B: synthesize into the schema ----
    contents.append(types.Content(role="user", parts=[types.Part(text=SYNTHESIZE_INSTRUCTION)]))
    synthesis = generate_with_retry(
        client,
        settings.gemini_agent_model,
        contents,
        types.GenerateContentConfig(
            system_instruction=EXPLORE_SYSTEM,
            response_mime_type="application/json",
            response_schema=FirstImpressionReport,
        ),
    )
    report: FirstImpressionReport = synthesis.parsed

    # Which pages did the agent actually READ (for the transparency trace).
    pages_examined = sorted(
        {s["args"]["url"] for s in steps_log if s["tool"] == "read_page" and "url" in s["args"]}
    )
    return report, steps_log, pages_examined
