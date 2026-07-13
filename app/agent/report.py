# app/agent/report.py — the report entry point + provider dispatch.
#
# generate_report() picks a driver based on settings.agent_provider and returns
# the same tuple regardless of which LLM produced it:
#     (FirstImpressionReport, steps_log, pages_examined)
#
#   "gemini" → _generate_gemini() below (Gemini function calling)
#   "groq"   → groq_driver.generate() (Groq / Llama tool calling)
#
# Both drivers share the instructions (agent/prompts.py), the tools
# (agent/tools.py), and the report schema (schemas.FirstImpressionReport).
# Only the API dialect differs — that's the whole point of the split.
#
# EXPLORE (free-form ReAct) then SYNTHESIZE (schema-constrained) is the shape
# in both drivers. We verified live that Gemini can reuse a tool-call history
# for a schema-constrained call (scratchpad smoke tests) before building this.
#
# CALL FLOW:
#   main.py report() → generate_report() → the selected driver

from google import genai
from google.genai import types

from app.agent import groq_driver, prompts, tools
from app.agent.llm import generate_with_retry
from app.agent.react import run_react_loop
from app.config import settings
from app.schemas import FirstImpressionReport


def _generate_gemini() -> tuple[FirstImpressionReport, list[dict], list[str]]:
    """Run explore → synthesize on Gemini."""
    client = genai.Client(api_key=settings.gemini_api_key)

    # ---- Phase A: explore ----
    explore_config = types.GenerateContentConfig(
        system_instruction=prompts.EXPLORE_SYSTEM,
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
        client, settings.gemini_agent_model, contents, explore_config, prompts.MAX_STEPS
    )

    # ---- Phase B: synthesize into the schema ----
    contents.append(
        types.Content(role="user", parts=[types.Part(text=prompts.SYNTHESIZE_INSTRUCTION)])
    )
    synthesis = generate_with_retry(
        client,
        settings.gemini_agent_model,
        contents,
        types.GenerateContentConfig(
            system_instruction=prompts.EXPLORE_SYSTEM,
            response_mime_type="application/json",
            response_schema=FirstImpressionReport,
        ),
    )
    report: FirstImpressionReport = synthesis.parsed

    pages_examined = sorted(
        {s["args"]["url"] for s in steps_log if s["tool"] == "read_page" and "url" in s["args"]}
    )
    return report, steps_log, pages_examined


def generate_report() -> tuple[FirstImpressionReport, list[dict], list[str]]:
    """Produce the structured report using the configured agent provider.

    Called by: main.py report(). Returns (report, steps_log, pages_examined).
    May raise provider rate-limit errors — the endpoint maps them to HTTP codes.
    """
    if settings.agent_provider == "groq":
        return groq_driver.generate()
    return _generate_gemini()
