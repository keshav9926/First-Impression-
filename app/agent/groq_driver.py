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

import pydantic
from google import genai
from google.genai import types as genai_types

from app import events
from app.agent import llm_pool, prompts, tools
from app.agent.llm import generate_with_retry
from app.config import settings
from app.schemas import FirstImpressionReport

# The report JSON shape, for providers WITHOUT native response_schema (the NVIDIA
# chain over OpenAI-compat). Mirrors FirstImpressionReport minus persona_panel,
# which is attached programmatically by the panel graph, never asked of the LLM.
_REPORT_JSON_SHAPE = (
    '{"company": str, '
    '"what_the_product_is": [{"claim": str, "evidence": str, "source_url": str}], '
    '"likely_new_user_journey": [{"claim": str, "evidence": str, "source_url": str}], '
    '"friction_points": [{"claim": str, "evidence": str, "source_url": str}], '
    '"standout_strengths": [{"claim": str, "evidence": str, "source_url": str}], '
    '"unanswered_questions": [str], '
    '"improvement_opportunities": [{"observed": str, "suggestion": str, "source_url": str}], '
    '"scope_note": str}'
)


def _synthesize_via_pool(synthesis_prompt: str) -> FirstImpressionReport | None:
    """Synthesize via the finalized NVIDIA chain (GLM → DeepSeek-Pro → Nemotron →
    Mistral) over OpenAI-compat JSON mode. Returns a validated report, or None if
    the whole chain produced nothing parseable (caller then falls back to Gemini).
    """
    system = (
        prompts.EXPLORE_SYSTEM
        + "\n\nReply ONLY with JSON of this exact shape (no other keys, no prose):\n"
        + _REPORT_JSON_SHAPE
    )
    try:
        message = llm_pool.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": synthesis_prompt}],
            prefer=settings.pool_prefer,
            response_format={"type": "json_object"},
        )
    except Exception:
        return None  # whole NVIDIA chain unavailable → Gemini fallback
    raw = message.content or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)  # salvage a truncated/wrapped object
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    data.pop("persona_panel", None)  # attached programmatically, never from the LLM
    try:
        return FirstImpressionReport.model_validate(data)
    except pydantic.ValidationError:
        return None

# Retry/failover machinery lives in llm_pool.chat() — one place for both
# providers (Groq + Cerebras). Step cap lives in settings.agent_max_steps.


def explore() -> tuple[list, list[dict]]:
    """Phase A: the Groq ReAct tool-calling loop. Returns (messages, steps_log).

    Called by: generate() below AND agent/panel.py (Phase 4) — the panel
    reuses this exact exploration so evidence is gathered ONCE per report.
    Runs on llm_pool (the finalized GLM-led chain; DeepSeek-Pro auto-skipped
    here because explore passes tools=).
    """
    messages: list = [
        {"role": "system", "content": prompts.EXPLORE_SYSTEM},
        {"role": "user", "content": "Analyze this company's public site and prepare its report."},
    ]
    steps_log: list[dict] = []
    seen_calls: set = set()  # repeat-call guard (see tools.repeat_call_reminder)

    for _ in range(settings.agent_max_steps):
        message = llm_pool.chat(
            messages,
            prefer=settings.pool_prefer,
            tools=tools.OPENAI_TOOLS,
            tool_choice="auto",
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
    """Phase B: turn the evidence into the schema-constrained report.

    Called by: generate() below and agent/panel.py (which passes the persona
    panel's findings as extra_context so the final report can reflect them).

    Primary: the finalized NVIDIA quality chain (GLM-5.2 → DeepSeek-V4-Pro →
    Nemotron-3-Ultra → Mistral-Medium-3.5) via the pool. Fallback: native Gemini
    with response_schema (a different key = rate-limit insurance, and Gemini's
    schema-constrained decoding is a reliable last resort if the NVIDIA chain
    returns nothing parseable).
    """
    synthesis_prompt = (
        "Below is the raw exploration log from a ReAct agent that examined the "
        "company's public website.\n\n"
        + context
        + (f"\n\n{extra_context}" if extra_context else "")
        + "\n\n"
        + prompts.SYNTHESIZE_INSTRUCTION
    )

    # Primary path: the finalized NVIDIA chain.
    report = _synthesize_via_pool(synthesis_prompt)
    if report is not None:
        return report

    # Fallback path: native Gemini (schema-constrained).
    synthesis_config = genai_types.GenerateContentConfig(
        system_instruction=prompts.EXPLORE_SYSTEM,
        response_mime_type="application/json",
        response_schema=FirstImpressionReport,
    )
    # Synthesis chain of (api_key, model): Gemini 3 Flash on account 1, then
    # account 2 (each ~20 RPD → ~40 premium syntheses/day combined). By choice
    # there is NO lower-tier fallback: once both accounts' 3-flash quota is
    # spent the report HARD-FAILS, so exhaustion is visible (add a 3rd key)
    # rather than silently degrading synthesis quality. Transient 503s are still
    # retried per candidate inside generate_with_retry.
    k1, k2 = settings.gemini_api_key, settings.gemini_secondacc_api_key
    candidates = [
        (k1, settings.gemini_agent_model),   # 3-flash · account 1
        (k2, settings.gemini_agent_model),   # 3-flash · account 2
    ]
    last_exc: Exception = RuntimeError("no Gemini key configured for synthesis")
    for api_key, model in candidates:
        if not api_key:
            continue
        try:
            synthesis_response = generate_with_retry(
                genai.Client(api_key=api_key), model, synthesis_prompt, synthesis_config
            )
            # Count this native (non-pool) Gemini call so get_usage() reflects
            # total quota consumption — 1 request/report when healthy.
            llm_pool.record(f"gemini-native:{model}")
        except Exception as exc:  # exhausted retries on this candidate → next
            llm_pool.record(f"gemini-native:{model}", "error")
            last_exc = exc
            continue
        report: FirstImpressionReport | None = synthesis_response.parsed
        if report is not None:
            return report
        # Schema-constrained output failed to parse (safety block / malformed
        # JSON) — try the next candidate rather than dying here.
        last_exc = ValueError("Synthesis returned no parseable report — retry the request.")
    raise last_exc


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
