"""
===============================================================================
FILE: app/agent/react.py
ORIGIN      : app.agent.report (generate_report)
PURPOSE     : Reason+Act (ReAct) iterative exploration loop for Gemini models
DESTINATION : app.agent.tools (Executes tool calls & feeds back observations)
===============================================================================
"""

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
    seen_calls: set = set()  # repeat-call guard (see tools.repeat_call_reminder)

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
            # Repeat-call guard: identical (tool, args) → short reminder
            # instead of re-executing (result already in history).
            observation = tools.repeat_call_reminder(
                fc.name, args, seen_calls
            ) or tools.execute_tool(fc.name, args)
            steps_log.append({"tool": fc.name, "args": args})
            response_parts.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": observation}
                )
            )
        contents.append(types.Content(role="user", parts=response_parts))

    return contents, steps_log
