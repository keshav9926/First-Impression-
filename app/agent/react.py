# app/agent/react.py — the hand-rolled ReAct loop (Reason + Act).
#
# This is the heart of "agentic": instead of a fixed pipeline, the model
# DECIDES each step what to do next, we DO it, the model OBSERVES the result,
# and it decides again — until it stops asking for tools.
#
#   ┌───────────────────────────────────────────────┐
#   │  model reasons → emits function_call(s)        │  (Reason + Act)
#   │  we run execute_tool() → observation string    │  (Act)
#   │  observation appended to history               │  (Observe)
#   │  loop, until model emits TEXT instead of a call │
#   └───────────────────────────────────────────────┘
#
# We use Gemini's MANUAL function calling (automatic calling disabled) so the
# loop is visible and logged — the whole point is to SEE the ReAct cycle, not
# hide it behind SDK magic. Built by hand on purpose; Phase 4 brings LangGraph
# in only when MULTI-agent orchestration genuinely needs it.
#
# CALL FLOW:
#   report.py generate_report() → run_react_loop(client, model, contents, config, max_steps)
#     └── per step: client.generate_content → tools.execute_tool → append observation
#   Returns the full conversation `contents` (which report.py then reuses for
#   the schema-constrained synthesis call) plus a log of what the agent did.

from google.genai import types

from app.agent import tools
from app.agent.llm import generate_with_retry


def run_react_loop(
    client,
    model: str,
    contents: list,
    config: types.GenerateContentConfig,
    max_steps: int,
) -> tuple[list, list[dict]]:
    """Drive the reason→act→observe loop until the model stops calling tools.

    Args:
        client: a google-genai Client (real, or a fake in tests).
        contents: the running conversation (starts with the system+task turn);
                  MUTATED and returned so the caller can reuse the full history.
        config: the GenerateContentConfig carrying the tool declarations.
        max_steps: hard cap so a misbehaving model can't loop forever.

    Returns:
        (contents, steps_log) where steps_log is a list of
        {"tool": name, "args": {...}} in the order the agent acted.
    """
    steps_log: list[dict] = []

    for _ in range(max_steps):
        response = generate_with_retry(client, model, contents, config)

        # Safety block or empty response → stop exploring, synthesize with
        # whatever we have.
        if not response.candidates:
            break

        function_calls = response.function_calls
        if not function_calls:
            # The model produced TEXT instead of a tool call = it's done
            # gathering. Keep that final turn in history and exit.
            contents.append(response.candidates[0].content)
            break

        # Record the model's tool-call turn in history (required before we can
        # answer it).
        contents.append(response.candidates[0].content)

        # Execute every requested call (Gemini may request several at once) and
        # return all observations in a single turn.
        response_parts = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            observation = tools.execute_tool(fc.name, args)
            steps_log.append({"tool": fc.name, "args": args})
            response_parts.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": observation}
                )
            )
        contents.append(types.Content(role="user", parts=response_parts))

    return contents, steps_log
