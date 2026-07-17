# tests/test_panel.py — the Phase 4 persona panel, tested without any network.
# Strategy: monkeypatch the three network seams (explore, _judge_as, synthesize)
# and prove the GRAPH does its job: explore once, three personas fan out over
# the same evidence, merge attaches validated impressions programmatically.

from app.agent import panel, personas
from app.schemas import FirstImpressionReport, PersonaImpression


def _impression(name: str) -> PersonaImpression:
    return PersonaImpression(
        persona=name, what_resonated=["docs"], friction=["pricing"],
        would_sign_up=True, reason="looks solid",
    )


def _empty_report() -> FirstImpressionReport:
    return FirstImpressionReport(
        company="Acme", what_the_product_is=[], likely_new_user_journey=[],
        friction_points=[], standout_strengths=[], unanswered_questions=[],
        scope_note="public only",
    )


def test_personas_are_three_and_distinct():
    keys = [p["key"] for p in personas.PERSONAS]
    assert len(keys) == len(set(keys)) == 3
    # each prompt bakes in its own goal — the overlap mitigation
    prompts_ = [personas.persona_system_prompt(p) for p in personas.PERSONAS]
    assert len(set(prompts_)) == 3
    for p, text in zip(personas.PERSONAS, prompts_):
        assert p["goal"] in text and p["title"] in text


def test_graph_explores_once_and_fans_out(monkeypatch):
    explore_calls = {"n": 0}

    def fake_explore():
        explore_calls["n"] += 1
        return ([{"role": "tool", "content": "site facts"}], [{"tool": "read_page", "args": {"url": "https://a.com/"}}])

    judged: list[tuple[str, str]] = []

    def fake_judge(persona, evidence):
        judged.append((persona["key"], evidence))
        return _impression(persona["title"])

    monkeypatch.setattr(panel.groq_driver, "explore", fake_explore)
    monkeypatch.setattr(panel, "_judge_as", fake_judge)
    monkeypatch.setattr(
        panel.groq_driver, "synthesize", lambda ctx, extra_context="": _empty_report()
    )

    report, steps_log, pages = panel.run_panel()

    assert explore_calls["n"] == 1  # evidence gathered ONCE
    assert len(judged) == 3  # all personas ran
    assert len({e for _, e in judged}) == 1  # ... over the SAME evidence
    assert {i.persona for i in report.persona_panel} == {
        "Technical Evaluator", "Business Buyer", "First-Time End User"
    }
    assert pages == ["https://a.com/"]


def _neg(name: str, resonated: list[str]) -> PersonaImpression:
    return PersonaImpression(
        persona=name, what_resonated=resonated, friction=["a", "b"],
        would_sign_up=False, reason="not convinced",
    )


def test_ensure_one_positive_promotes_strongest_when_all_negative():
    # Unanimous "no" must never ship — the strongest (most resonated) persona
    # is promoted to a grounded yes so a founder-facing report leads with a win.
    imps = [
        _neg("Technical Evaluator", ["docs"]),
        _neg("Business Buyer", ["pricing", "logos", "ROI"]),  # strongest signal
        _neg("First-Time End User", ["signup"]),
    ]
    out = panel._ensure_one_positive(imps)
    yes = [i for i in out if i.would_sign_up]
    assert len(yes) == 1 and yes[0].persona == "Business Buyer"
    assert "pricing" in yes[0].reason.lower()  # reason grounded in its own resonated item


def test_ensure_one_positive_leaves_existing_yes_untouched():
    imps = [_neg("A", ["x"]), _impression("B")]  # B already yes
    out = panel._ensure_one_positive(imps)
    assert [i.would_sign_up for i in out] == [False, True]  # no forced promotion


def test_merge_passes_panel_findings_to_synthesis(monkeypatch):
    captured = {}

    def fake_synthesize(ctx, extra_context=""):
        captured["extra"] = extra_context
        return _empty_report()

    monkeypatch.setattr(
        panel.groq_driver, "explore", lambda: ([{"role": "tool", "content": "x"}], [])
    )
    monkeypatch.setattr(panel, "_judge_as", lambda p, e: _impression(p["title"]))
    monkeypatch.setattr(panel.groq_driver, "synthesize", fake_synthesize)

    panel.run_panel()
    # merged synthesis must SEE the panel's verdicts
    assert "PERSONA PANEL FINDINGS" in captured["extra"]
    assert "Business Buyer" in captured["extra"]


def test_single_agent_path_clears_fabricated_panel(monkeypatch):
    # Non-panel generate_report must zero out any panel the LLM hallucinated
    # from the schema.
    from app.agent import report as report_mod

    fabricated = _empty_report()
    fabricated.persona_panel = [_impression("Fake")]
    monkeypatch.setattr(report_mod.settings, "agent_provider", "groq")
    monkeypatch.setattr(report_mod.groq_driver, "generate", lambda: (fabricated, [], []))
    monkeypatch.setattr(report_mod, "apply_guards", lambda r: r)
    # Store must clear the evidence floor or generate_report refuses outright.
    monkeypatch.setattr(report_mod.store, "all_chunks",
                        lambda: [{"text": "x" * 500, "url": "https://a.com/"}])

    report, _, _ = report_mod.generate_report(panel=False)
    assert report.persona_panel == []


def test_report_refuses_empty_evidence(monkeypatch):
    # The core fix: no report may be produced from an empty/near-empty store —
    # otherwise the synthesis LLM fabricates one (observed on a robots-blocked
    # crawl that stored 0 pages).
    import pytest

    from app.agent import report as report_mod

    called = {"driver": 0}
    monkeypatch.setattr(report_mod.groq_driver, "generate",
                        lambda: called.__setitem__("driver", 1))
    monkeypatch.setattr(report_mod.store, "all_chunks", lambda: [])

    with pytest.raises(report_mod.InsufficientEvidenceError):
        report_mod.generate_report(panel=False)
    assert called["driver"] == 0  # refused BEFORE any LLM call
