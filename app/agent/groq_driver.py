# app/agent/groq_driver.py — the report agent, driven by Groq (Llama).
#
# TWO distinct phases with DIFFERENT providers:
#   Phase A — EXPLORE  : Groq's OpenAI-compatible tool-calling loop.
#                         Groq's generous free-tier RPM handles the burst well.
#   Phase B — SYNTHESIZE (final eval): Gemini — native response_schema gives us
#                         type-safe structured output; Gemini is also the model
#                         the user wants for the final "judgment" call.
#
# Groq is on the OpenAI-compatible chat.completions API, so the message shapes
# differ from Gemini's (role "assistant" with tool_calls; role "tool" results),
# which is exactly why each provider needs its own driver rather than a shared
# loop. What stays shared is everything that matters: the instructions, the
# tool behavior, and the FirstImpressionReport schema.
#
# CALL FLOW:
#   report.py generate_report() → generate()   (when agent_provider == "groq")

import json
import re
import time

import groq
from google import genai
from google.genai import types as genai_types

from app.agent import prompts, tools
from app.config import settings
from app.schemas import FirstImpressionReport

_MAX_RATE_RETRIES = 6
# Groq free tier is ~12K tokens/minute, and the whole history is resent each
# step. Cap at 5 to match the Gemini driver's MAX_STEPS (list_pages + read
# key pages + 1-2 targeted searches is sufficient for most startup sites).
_GROQ_MAX_STEPS = 5


def _retry_after_seconds(exc: groq.RateLimitError) -> float:
    """Groq returns a Retry-After header and/or 'try again in 1.2s' text."""
    header = exc.response.headers.get("retry-after") if exc.response is not None else None
    if header:
        try:
            return float(header) + 0.5
        except ValueError:
            pass
    match = re.search(r"try again in ([0-9.]+)s", str(exc))
    return float(match.group(1)) + 0.5 if match else 2.0


def _complete(client: groq.Groq, **kwargs):
    """chat.completions.create with retry on free-tier 429s (honor Retry-After)."""
    for attempt in range(_MAX_RATE_RETRIES):
        try:
            return client.chat.completions.create(model=settings.groq_model, **kwargs)
        except groq.RateLimitError as exc:
            if attempt == _MAX_RATE_RETRIES - 1:
                raise
            time.sleep(min(_retry_after_seconds(exc), 60))


def generate() -> tuple[FirstImpressionReport, list[dict], list[str]]:
    """Run explore (Groq) → synthesize (Gemini). Returns (report, steps_log, pages).

    Phase A: Groq drives the ReAct tool-calling loop — its generous free-tier
    RPM handles the burst of calls without exhausting a daily quota.
    Phase B: Gemini produces the final structured report via response_schema —
    native type-safe schema enforcement, and the "final eval" model the user
    configured for quality-critical output.
    """
    groq_client = groq.Groq(api_key=settings.groq_api_key)

    # ---- Phase A: explore (Groq tool-calling loop) ----
    messages: list = [
        {"role": "system", "content": prompts.EXPLORE_SYSTEM},
        {"role": "user", "content": "Analyze this company's public site and prepare its report."},
    ]
    steps_log: list[dict] = []

    for _ in range(_GROQ_MAX_STEPS):
        response = _complete(
            groq_client, messages=messages, tools=tools.OPENAI_TOOLS, tool_choice="auto"
        )
        message = response.choices[0].message

        if not message.tool_calls:
            # Model produced text instead of a tool call = done exploring.
            messages.append({"role": "assistant", "content": message.content or ""})
            break

        # Record the assistant's tool-call turn (must precede the tool results).
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        # Execute each call; return one tool message per call.
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            observation = tools.execute_tool(tc.function.name, args)
            steps_log.append({"tool": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})

    # ---- Phase B: synthesize into the schema (Gemini — final eval) ----
    # Groq explored; Gemini produces the definitive structured report.
    # We distill the Groq conversation into a flat context block that Gemini's
    # stateless API can consume, then use response_schema for native enforcement
    # (no JSON mode tricks, no Pydantic retry loop needed).
    context_parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            continue  # already in EXPLORE_SYSTEM prompt below
        content = msg.get("content") or ""
        if role == "assistant" and msg.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in msg["tool_calls"]
            )
            context_parts.append(f"[agent called tools: {calls}]")
        elif role == "tool":
            context_parts.append(f"[tool result]: {content}")
        elif content:
            context_parts.append(f"[{role}]: {content}")

    gemini_client = genai.Client(api_key=settings.gemini_api_key)
    synthesis_prompt = (
        "Below is the raw exploration log from a ReAct agent that examined the "
        "company's public website using Groq.\n\n"
        + "\n".join(context_parts)
        + "\n\n"
        + prompts.SYNTHESIZE_INSTRUCTION
    )
    synthesis_response = gemini_client.models.generate_content(
        model=settings.gemini_agent_model,
        contents=synthesis_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=prompts.EXPLORE_SYSTEM,
            response_mime_type="application/json",
            response_schema=FirstImpressionReport,
        ),
    )
    report: FirstImpressionReport = synthesis_response.parsed

    pages_examined = sorted(
        {s["args"]["url"] for s in steps_log if s["tool"] == "read_page" and "url" in s["args"]}
    )
    return report, steps_log, pages_examined
