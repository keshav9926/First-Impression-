# app/agent/groq_driver.py — the report agent, driven by Groq (Llama).
#
# Same TWO phases as the Gemini driver, same prompts (agent/prompts.py), same
# tools (agent/tools.execute_tool) — only the API dialect differs:
#   Phase A — EXPLORE: Groq's OpenAI-compatible tool-calling loop.
#   Phase B — SYNTHESIZE: JSON mode + Pydantic validation into the report.
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

from app.agent import prompts, tools
from app.config import settings
from app.schemas import FirstImpressionReport

_MAX_RATE_RETRIES = 6
# Groq free tier is ~12K tokens/minute, and the whole history is resent each
# step, so keep the exploration shorter than the Gemini cap to bound growth.
_GROQ_MAX_STEPS = 8

# A compact, human-readable shape for synthesis. We send THIS instead of the
# full FirstImpressionReport JSON schema (which is ~1K tokens) to save context;
# Pydantic still validates the result, so correctness doesn't depend on it.
_SHAPE_HINT = """\
{
  "company": "string",
  "what_the_product_is":       [{"claim": "string", "evidence": "string", "source_url": "string"}],
  "likely_new_user_journey":   [{"claim": "string", "evidence": "string", "source_url": "string"}],
  "friction_points":           [{"claim": "string", "evidence": "string", "source_url": "string"}],
  "standout_strengths":        [{"claim": "string", "evidence": "string", "source_url": "string"}],
  "unanswered_questions":      ["string"],
  "scope_note": "string"
}"""


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
    """Run explore → synthesize on Groq. Returns (report, steps_log, pages)."""
    client = groq.Groq(api_key=settings.groq_api_key)

    # ---- Phase A: explore (tool-calling loop) ----
    messages: list = [
        {"role": "system", "content": prompts.EXPLORE_SYSTEM},
        {"role": "user", "content": "Analyze this company's public site and prepare its report."},
    ]
    steps_log: list[dict] = []

    for _ in range(_GROQ_MAX_STEPS):
        response = _complete(
            client, messages=messages, tools=tools.OPENAI_TOOLS, tool_choice="auto"
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

    # ---- Phase B: synthesize into the schema (JSON mode + Pydantic) ----
    # Groq's JSON mode guarantees valid JSON but not schema-conformance, so we
    # hand it the shape and VALIDATE with Pydantic (retrying once on a bad
    # shape). "json" must appear in the prompt for Groq's JSON mode.
    messages.append(
        {
            "role": "user",
            "content": (
                f"{prompts.SYNTHESIZE_INSTRUCTION}\n\n"
                f"Return ONLY a JSON object with exactly this shape:\n{_SHAPE_HINT}"
            ),
        }
    )

    report: FirstImpressionReport | None = None
    for attempt in range(2):
        response = _complete(client, messages=messages, response_format={"type": "json_object"})
        raw = response.choices[0].message.content or "{}"
        try:
            report = FirstImpressionReport.model_validate_json(raw)
            break
        except ValueError as exc:
            if attempt == 1:
                raise
            # Tell the model exactly what was wrong and let it fix the shape.
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"That did not match the schema: {exc}. Fix it."}
            )

    pages_examined = sorted(
        {s["args"]["url"] for s in steps_log if s["tool"] == "read_page" and "url" in s["args"]}
    )
    return report, steps_log, pages_examined
