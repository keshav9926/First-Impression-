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
from app.agent.llm import generate_with_retry
from app.config import settings
from app.schemas import FirstImpressionReport

_MAX_RATE_RETRIES = 6
# Llama occasionally emits malformed tool-call syntax that Groq rejects with a
# 400 "tool_use_failed" — a STOCHASTIC generation glitch, not a bug in our
# request. Groq samples with temperature, so simply re-asking usually produces
# valid syntax. Retry a few times before giving up.
_MAX_TOOL_FORMAT_RETRIES = 3
# Step cap lives in config (settings.agent_max_steps) — ONE value shared with
# the Gemini driver so the two can never drift apart.


def _is_tool_format_error(exc: groq.BadRequestError) -> bool:
    """True for the transient 'tool_use_failed' 400 (retriable), False for any
    other 400 (a real bad request — must NOT be retried, that would just mask a
    genuine bug behind three identical failures)."""
    return "tool_use_failed" in str(exc)


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
    """chat.completions.create, retrying two distinct transient failures:
      - 429 rate limits (honor Retry-After),
      - 400 'tool_use_failed' (Llama emitted malformed tool-call syntax).
    Any other error propagates immediately — a real bug shouldn't be retried.
    """
    rate_attempts = 0
    format_attempts = 0
    while True:
        try:
            return client.chat.completions.create(model=settings.groq_model, **kwargs)
        except groq.RateLimitError as exc:
            rate_attempts += 1
            if rate_attempts >= _MAX_RATE_RETRIES:
                raise
            time.sleep(min(_retry_after_seconds(exc), 60))
        except groq.BadRequestError as exc:
            format_attempts += 1
            if not _is_tool_format_error(exc) or format_attempts >= _MAX_TOOL_FORMAT_RETRIES:
                raise
            # No sleep: it's a sampling glitch, not a rate issue — just re-ask.


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
    seen_calls: set = set()  # repeat-call guard (see tools.repeat_call_reminder)

    for _ in range(settings.agent_max_steps):
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
            # `or {}` twice: arguments may be "" (falsy) OR the string "null"
            # (json.loads → None) — both must become an empty dict, or
            # args.get(...) in execute_tool would crash on None.
            args = (json.loads(tc.function.arguments) if tc.function.arguments else {}) or {}
            # Repeat-call guard: identical (tool, args) → short reminder
            # instead of re-executing (result already in history).
            observation = tools.repeat_call_reminder(
                tc.function.name, args, seen_calls
            ) or tools.execute_tool(tc.function.name, args)
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
    # generate_with_retry: the whole exploration is already paid for by the
    # time we get here — a transient Gemini free-tier 429 must not throw all
    # of that work away when waiting a few seconds would save it.
    synthesis_response = generate_with_retry(
        gemini_client,
        settings.gemini_agent_model,
        synthesis_prompt,
        genai_types.GenerateContentConfig(
            system_instruction=prompts.EXPLORE_SYSTEM,
            response_mime_type="application/json",
            response_schema=FirstImpressionReport,
        ),
    )
    report: FirstImpressionReport | None = synthesis_response.parsed
    if report is None:
        # Schema-constrained output failed to parse (safety block / malformed
        # JSON) — fail with a clear message, not an AttributeError downstream.
        raise ValueError("Synthesis returned no parseable report — retry the request.")

    pages_examined = sorted(
        {s["args"]["url"] for s in steps_log if s["tool"] == "read_page" and "url" in s["args"]}
    )
    return report, steps_log, pages_examined
