# app/agent/panel.py — Phase 4: the persona panel, orchestrated with LangGraph.
#
# TOPOLOGY (why a graph library is finally justified — real fan-out/fan-in):
#
#   explore (ONCE, Groq ReAct — reuses groq_driver.explore)
#      ├── persona: Technical Evaluator ┐
#      ├── persona: Business Buyer      ├─ PARALLEL, same shared evidence
#      └── persona: First-Time End User ┘
#            └── merge: Gemini synthesis (evidence + panel findings)
#                  → persona_panel attached PROGRAMMATICALLY (validated objects,
#                    never asked of the synthesis LLM)
#
# Exploration is the expensive part (tool calls, Voyage searches) — it runs
# ONCE; judgment (cheap synthesis-only calls) fans out per persona. Personas
# run on Groq (JSON mode + Pydantic validation, one retry) because the fan-out
# is a burst — Groq's RPM absorbs it; Gemini's ~20/day quota could not.
#
# CALL FLOW:
#   main.py report(panel=True) → run_panel()
#     └── LangGraph: _explore_node → _persona_node×3 → _merge_node
#   Guards (citations, thin-extraction caveat) are applied by report.py's
#   apply_guards — same as the single-agent path.

import json
import operator
from typing import Annotated, TypedDict

import pydantic

from app import events, observability
from app.agent import groq_driver, llm_pool, personas
from app.config import settings
from app.schemas import FirstImpressionReport, PersonaImpression

_PERSONA_RETRIES = 2  # JSON-mode replies occasionally malformed — one re-ask


class PanelState(TypedDict):
    """The graph's shared state. `impressions` uses an operator.add reducer so
    the three parallel persona nodes APPEND without clobbering each other."""

    evidence: str
    steps_log: list
    impressions: Annotated[list[PersonaImpression], operator.add]
    report: FirstImpressionReport | None


def _explore_node(state: PanelState) -> dict:
    """Run the (single) Groq ReAct exploration; flatten it into the shared
    evidence block every persona will judge."""
    messages, steps_log = groq_driver.explore()
    return {"evidence": groq_driver.flatten_context(messages), "steps_log": steps_log}


def _judge_as(persona: dict, evidence: str) -> PersonaImpression:
    """One persona's verdict over the shared evidence: JSON-mode reply via the
    provider pool (prefer = settings.pool_prefer, the GLM-led NVIDIA chain),
    validated into PersonaImpression (retried once on malformed JSON)."""
    last_error: Exception = ValueError("no attempt made")
    for _ in range(_PERSONA_RETRIES):
        message = llm_pool.chat(
            [
                {"role": "system", "content": personas.persona_system_prompt(persona)},
                {"role": "user", "content": f"EVIDENCE:\n\n{evidence}"},
            ],
            prefer=settings.pool_prefer,
            response_format={"type": "json_object"},
            label="persona-judge",
        )
        try:
            return PersonaImpression.model_validate(json.loads(message.content or ""))
        except (json.JSONDecodeError, pydantic.ValidationError) as exc:
            last_error = exc  # malformed reply — re-ask (sampling glitch)
    raise ValueError(f"Persona {persona['key']} returned unusable JSON: {last_error}")


def _make_persona_node(persona: dict):
    """Build one graph node for one persona (closure carries the definition)."""

    def node(state: PanelState) -> dict:
        # Each persona is a distinct subagent → an `agent` observation, named per
        # persona so it's a distinguishable node in Langfuse's Agent Graph. Its
        # judge generation nests under it.
        name = persona.get("name") or persona["key"]
        with observability.span(f"persona:{name}", as_type="agent") as obs:
            impression = _judge_as(persona, state["evidence"])
            if obs:
                obs.update(
                    output={
                        "would_sign_up": impression.would_sign_up,
                        "reason": impression.reason,
                    }
                )
        events.emit(
            "persona",
            persona=impression.persona,
            would_sign_up=impression.would_sign_up,
            reason=impression.reason,
        )
        return {"impressions": [impression]}

    return node


def _order_by_strength(impressions: list[PersonaImpression]) -> list[PersonaImpression]:
    """Lead with credit, WITHOUT faking the verdict.

    A founder-facing report should open with what genuinely worked — but it must
    never fabricate a "would sign up" that no persona actually gave (that would
    make the report's core signal a lie, the opposite of trustworthy). So we
    leave every would_sign_up exactly as the persona judged it and instead ORDER
    the panel so the strongest positive signal is surfaced first: a yes before a
    no, then most-resonated-minus-friction, then most-resonated.

    The honest softening lives in the prompts: personas.py leans generous and
    frames friction as "what would raise confidence," and the synthesis
    instruction keeps the tone observational. A rare unanimous "no" still reaches
    the founder — framed kindly, not hidden behind a manufactured yes."""
    return sorted(
        impressions,
        key=lambda i: (
            i.would_sign_up,
            len(i.what_resonated) - len(i.friction),
            len(i.what_resonated),
        ),
        reverse=True,
    )


def _merge_node(state: PanelState) -> dict:
    """Fan-in: synthesize the final report from evidence + the panel's findings,
    then attach the validated impressions programmatically."""
    impressions = _order_by_strength(state["impressions"])
    panel_context = "PERSONA PANEL FINDINGS (three visitors judged the same evidence):\n" + "\n".join(
        f"- {i.persona}: would sign up: {i.would_sign_up} — {i.reason} "
        f"| resonated: {'; '.join(i.what_resonated)} | friction: {'; '.join(i.friction)}"
        for i in impressions
    )
    report = groq_driver.synthesize(state["evidence"], extra_context=panel_context)
    # Panel attached from the validated objects — the LLM's own persona_panel
    # output (if any) is overwritten; opinion enters the report exactly once.
    report.persona_panel = list(impressions)
    return {"report": report}


def _build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PanelState)
    graph.add_node("explore", _explore_node)
    graph.add_node("merge", _merge_node)
    for p in personas.PERSONAS:
        graph.add_node(p["key"], _make_persona_node(p))
        graph.add_edge("explore", p["key"])  # fan-out
        graph.add_edge(p["key"], "merge")  # fan-in barrier
    graph.add_edge(START, "explore")
    graph.add_edge("merge", END)
    return graph.compile()


def run_panel() -> tuple[FirstImpressionReport, list[dict], list[str]]:
    """Run the full panel graph. Returns (report, steps_log, pages_examined) —
    the same contract as the single-agent drivers, so report.py can apply the
    same guards afterwards."""
    state = _build_graph().invoke(
        {"evidence": "", "steps_log": [], "impressions": [], "report": None}
    )
    report: FirstImpressionReport = state["report"]
    steps_log = state["steps_log"]
    return report, steps_log, groq_driver.pages_from_steps(steps_log)
