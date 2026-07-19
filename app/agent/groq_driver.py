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

from app import events, observability
from app.agent import llm_pool, prompts, tools
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
            label="synthesize",
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


# --- history bounding (the explore-loop resend pain point) -----------------
# The ReAct loop resends the ENTIRE conversation every turn. Left unbounded, by
# turn 7 that means re-sending every read_page observation (~4000 chars each) 7
# times — the dominant cost of a slow report. Fix: keep only the most recent
# observations verbatim (the model just acted on those); collapse older tool
# results to a short head + a pointer to search_content. Cumulative and in-place
# — a page read 3 turns ago no longer costs its full text on every later turn.
# Structure is untouched (only `tool` message CONTENT shrinks), so the API's
# tool_call/tool_result pairing stays valid.
_KEEP_FULL_OBS = 2
_TRIM_OBS_TO = 600


def _trim_history(messages: list) -> None:
    tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    older = tool_positions[:-_KEEP_FULL_OBS] if len(tool_positions) > _KEEP_FULL_OBS else []
    for i in older:
        content = messages[i].get("content") or ""
        if len(content) > _TRIM_OBS_TO:
            messages[i]["content"] = (
                content[:_TRIM_OBS_TO]
                + " […earlier observation truncated to save context; "
                "use search_content to re-retrieve specifics if needed]"
            )


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

    # Trace the ReAct loop as an `agent` observation — its generations
    # (explore-step) and the retriever/tool calls it triggers nest under it.
    with observability.span("explore", as_type="agent"):
        for _ in range(settings.agent_max_steps):
            _trim_history(messages)  # bound the resent context before each call
            message = llm_pool.chat(
                messages,
                prefer=settings.pool_prefer,
                tools=tools.OPENAI_TOOLS,
                tool_choice="auto",
                label="explore-step",
            )

            if not message.tool_calls:
                # Model produced text instead of a tool call = done exploring.
                messages.append({"role": "assistant", "content": message.content or ""})
                break

            # Record the assistant's tool-call turn (must precede the results).
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
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
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": observation}
                )

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

    Runs on the finalized NVIDIA quality chain (GLM-5.2 → DeepSeek-V4-Pro →
    Nemotron-3-Ultra → Mistral-Medium-3.5) via the pool, over OpenAI-compat JSON
    mode. If the WHOLE chain returns nothing parseable, we HARD-FAIL (ValueError)
    rather than silently degrading — a visible failure the caller maps to a 502,
    which is recoverable by retry, beats shipping a half-baked report.
    """
    synthesis_prompt = (
        "Below is the raw exploration log from a ReAct agent that examined the "
        "company's public website.\n\n"
        + context
        + (f"\n\n{extra_context}" if extra_context else "")
        + "\n\n"
        + prompts.SYNTHESIZE_INSTRUCTION
    )

    report = _synthesize_via_pool(synthesis_prompt)
    if report is not None:
        return report
    raise ValueError(
        "Synthesis produced no parseable report from the NVIDIA chain — retry the request."
    )


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
