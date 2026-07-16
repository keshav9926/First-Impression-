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

from google import genai
from google.genai import types as genai_types

from app import events
from app.agent import llm_pool, prompts, tools
from app.agent.llm import generate_with_retry
from app.config import settings
from app.schemas import FirstImpressionReport

# Retry/failover machinery lives in llm_pool.chat() — one place for both
# providers (Groq + Cerebras). Step cap lives in settings.agent_max_steps.


def explore() -> tuple[list, list[dict]]:
    """Phase A: the Groq ReAct tool-calling loop. Returns (messages, steps_log).

    Called by: generate() below AND agent/panel.py (Phase 4) — the panel
    reuses this exact exploration so evidence is gathered ONCE per report.
    Runs on llm_pool (prefer Groq, fail over to Cerebras on daily quota).
    """
    messages: list = [
        {"role": "system", "content": prompts.EXPLORE_SYSTEM},
        {"role": "user", "content": "Analyze this company's public site and prepare its report."},
    ]
    steps_log: list[dict] = []
    seen_calls: set = set()  # repeat-call guard (see tools.repeat_call_reminder)

    for _ in range(settings.agent_max_steps):
        message = llm_pool.chat(
            messages, prefer="groq", tools=tools.OPENAI_TOOLS, tool_choice="auto"
        )

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
            events.emit("tool", name=tc.function.name, args=args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})

    return messages, steps_log


def flatten_context(messages: list) -> str:
    """Distill the Groq tool-call conversation into one flat evidence block
    that a stateless API call (Gemini synthesis, persona nodes) can consume.

    Called by: synthesize() below and agent/panel.py (personas read this)."""
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
    return "\n".join(context_parts)


def synthesize(context: str, extra_context: str = "") -> FirstImpressionReport:
    """Phase B: Gemini turns the evidence into the schema-constrained report.

    Called by: generate() below and agent/panel.py (which passes the persona
    panel's findings as extra_context so the final report can reflect them).
    generate_with_retry: the whole exploration is already paid for by now — a
    transient Gemini free-tier 429 must not throw that work away.
    """
    gemini_client = genai.Client(api_key=settings.gemini_api_key)
    synthesis_prompt = (
        "Below is the raw exploration log from a ReAct agent that examined the "
        "company's public website using Groq.\n\n"
        + context
        + (f"\n\n{extra_context}" if extra_context else "")
        + "\n\n"
        + prompts.SYNTHESIZE_INSTRUCTION
    )
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
    return report


def pages_from_steps(steps_log: list[dict]) -> list[str]:
    """Distinct urls the agent actually read, from the steps log."""
    return sorted(
        {s["args"]["url"] for s in steps_log if s["tool"] == "read_page" and "url" in s["args"]}
    )


def generate() -> tuple[FirstImpressionReport, list[dict], list[str]]:
    """Explore (Groq) → synthesize (Gemini). Returns (report, steps_log, pages).

    Called by: report.generate_report() when agent_provider == "groq".
    Composed from the reusable pieces above (which agent/panel.py also uses).
    """
    messages, steps_log = explore()
    report = synthesize(flatten_context(messages))
    return report, steps_log, pages_from_steps(steps_log)
