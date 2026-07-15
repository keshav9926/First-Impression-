# tests/test_agent.py — the analysis agent, tested without any network.
#
# Two targets:
#   1. tools.py   — monkeypatch store.all_chunks() to fake ingested content,
#                   then check each tool returns the right observation string.
#   2. react.py   — drive run_react_loop with a FAKE Gemini client that emits
#                   scripted tool calls then text. This proves the ReAct loop
#                   mechanics (execute → append observation → stop on text)
#                   with zero API calls — the loop is the heart of the agent,
#                   so it's the thing most worth pinning down.

from types import SimpleNamespace

from google.genai import types

from app.agent import grounding, react, tools
from app.schemas import FirstImpressionReport, ImprovementOpportunity, Observation

FAKE_CHUNKS = [
    {"id": "chunk-0", "text": "Acme builds widgets for small teams.", "url": "https://acme.com/"},
    {"id": "chunk-1", "text": "Widgets sync every night automatically.", "url": "https://acme.com/"},
    {"id": "chunk-2", "text": "Pricing starts at $20 per month.", "url": "https://acme.com/pricing"},
]


# ----- tools.py -----


def test_list_pages_returns_distinct_urls(monkeypatch):
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("list_pages", {})
    assert "https://acme.com/" in out
    assert "https://acme.com/pricing" in out
    # distinct — the homepage (2 chunks) is listed once
    assert out.count("https://acme.com/pricing") == 1


def test_read_page_concatenates_a_pages_chunks_in_order(monkeypatch):
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("read_page", {"url": "https://acme.com/"})
    # both homepage chunks present, pricing chunk absent
    assert "builds widgets" in out
    assert "sync every night" in out
    assert "$20 per month" not in out
    # order preserved (chunk-0 before chunk-1)
    assert out.index("builds widgets") < out.index("sync every night")


def test_read_page_unknown_url_lists_whats_available(monkeypatch):
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("read_page", {"url": "https://acme.com/nope"})
    assert "No page found" in out
    assert "https://acme.com/pricing" in out  # helps the model recover


def test_read_page_recovers_from_a_bare_slug(monkeypatch):
    # Models often pass "pricing" instead of the exact URL. An unambiguous
    # slug should resolve to the real page instead of wasting a step.
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("read_page", {"url": "pricing"})
    assert "$20 per month" in out
    assert "No page found" not in out


def test_read_page_bare_home_resolves_to_root(monkeypatch):
    # "home"/"index"/"" → the URL with an empty path (the site root).
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("read_page", {"url": "home"})
    assert "builds widgets" in out
    assert "$20 per month" not in out  # root, not the pricing page


def test_read_page_ambiguous_slug_still_asks_for_exact_url(monkeypatch):
    # If a slug matches more than one page, don't guess — ask for the exact URL.
    chunks = [
        {"id": "a", "text": "US pricing.", "url": "https://acme.com/us/pricing"},
        {"id": "b", "text": "EU pricing.", "url": "https://acme.com/eu/pricing"},
    ]
    monkeypatch.setattr(tools.store, "all_chunks", lambda: chunks)
    out = tools.execute_tool("read_page", {"url": "pricing"})
    assert "No page found" in out


def test_unknown_tool_returns_error_not_exception(monkeypatch):
    out = tools.execute_tool("teleport", {"to": "mars"})
    assert "Unknown tool" in out  # a string, not a raised exception


def test_search_content_borderline_is_uncertain_not_a_gap(monkeypatch):
    # Best match sits just under the bar → uncertain, NOT a confirmed absence.
    monkeypatch.setattr(tools.settings, "min_relevance", 0.45)
    monkeypatch.setattr(
        tools.pipeline, "retrieve", lambda q, top_k: [{"relevance": 0.40, "url": "u", "text": "t"}]
    )
    out = tools.execute_tool("search_content", {"query": "security"})
    assert "uncertain" in out.lower()
    assert "was found" not in out  # not the hard "nothing found" wording


def test_search_content_far_below_bar_is_a_hard_miss(monkeypatch):
    # Best match far under the bar → the site really doesn't cover it.
    monkeypatch.setattr(tools.settings, "min_relevance", 0.45)
    monkeypatch.setattr(
        tools.pipeline, "retrieve", lambda q, top_k: [{"relevance": 0.10, "url": "u", "text": "t"}]
    )
    out = tools.execute_tool("search_content", {"query": "security"})
    assert "No content relevant" in out


# ----- grounding.py: citation verification (rule #2 made structural) -----


def _report_with_urls(*urls):
    """A minimal report whose what_the_product_is cites the given source urls."""
    return FirstImpressionReport(
        company="Acme",
        what_the_product_is=[
            Observation(claim=f"claim {i}", evidence="e", source_url=u)
            for i, u in enumerate(urls)
        ],
        likely_new_user_journey=[],
        friction_points=[],
        standout_strengths=[],
        unanswered_questions=["untouched"],
        scope_note="public only",
    )


def test_enforce_citations_drops_hallucinated_urls():
    report = _report_with_urls("https://acme.com/", "https://acme.com/ghost")
    report, dropped = grounding.enforce_citations(report, ["https://acme.com/"])
    urls = [o.source_url for o in report.what_the_product_is]
    assert urls == ["https://acme.com/"]  # real kept, ghost dropped
    assert len(dropped) == 1 and dropped[0]["source_url"] == "https://acme.com/ghost"
    assert report.unanswered_questions == ["untouched"]  # non-cited field untouched


def test_enforce_citations_tolerates_trailing_slash_and_case():
    # Synthesis may emit a trailing slash / different case than the store.
    report = _report_with_urls("https://Acme.com/Docs/")
    report, dropped = grounding.enforce_citations(report, ["https://acme.com/docs"])
    assert len(report.what_the_product_is) == 1 and not dropped


def test_enforce_citations_also_verifies_improvement_suggestions():
    # A friendly suggestion pinned to a hallucinated page must be dropped too —
    # advice is grounded to a real page or it does not ship.
    report = _report_with_urls("https://acme.com/")
    report.improvement_opportunities = [
        ImprovementOpportunity(
            observed="pricing needs a form", suggestion="show public pricing",
            source_url="https://acme.com/pricing",  # real
        ),
        ImprovementOpportunity(
            observed="made up", suggestion="do a thing",
            source_url="https://acme.com/ghost",  # hallucinated
        ),
    ]
    report, dropped = grounding.enforce_citations(
        report, ["https://acme.com/", "https://acme.com/pricing"]
    )
    kept = [o.source_url for o in report.improvement_opportunities]
    assert kept == ["https://acme.com/pricing"]
    assert any(d["source_url"] == "https://acme.com/ghost" for d in dropped)


# ----- react.py: the loop, driven by a fake LLM -----


class _FakeModels:
    """Scripted generate_content: returns queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


def _tool_call_response(name, args):
    """A fake response whose candidate content holds a function_call part."""
    part = types.Part(function_call=types.FunctionCall(name=name, args=args))
    content = types.Content(role="model", parts=[part])
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
        function_calls=[types.FunctionCall(name=name, args=args)],
    )


def _text_response(text):
    """A fake response with no function calls = the model is done."""
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], function_calls=[])


def test_loop_runs_tools_then_stops_on_text(monkeypatch):
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)

    # Scripted agent: call list_pages, then read_page, then answer with text.
    client = _FakeClient(
        [
            _tool_call_response("list_pages", {}),
            _tool_call_response("read_page", {"url": "https://acme.com/"}),
            _text_response("I have enough to write the report."),
        ]
    )

    contents, steps_log = react.run_react_loop(
        client, model="fake", contents=[], config=None, max_steps=10
    )

    # It executed exactly the two tools, in order, then stopped.
    assert [s["tool"] for s in steps_log] == ["list_pages", "read_page"]
    assert steps_log[1]["args"] == {"url": "https://acme.com/"}
    assert client.models.calls == 3  # two tool turns + the final text turn


def test_repeat_call_reminder_blocks_second_identical_call():
    seen: set = set()
    # First call: allowed (returns None), recorded in `seen`.
    assert tools.repeat_call_reminder("read_page", {"url": "https://a.com/"}, seen) is None
    # Identical repeat: blocked with a reminder string.
    out = tools.repeat_call_reminder("read_page", {"url": "https://a.com/"}, seen)
    assert out is not None and "already called" in out
    # Different args: allowed again.
    assert tools.repeat_call_reminder("read_page", {"url": "https://b.com/"}, seen) is None


def test_loop_reminds_instead_of_reexecuting_a_repeat(monkeypatch):
    # The store must only be read ONCE for two identical read_page calls —
    # the second gets the reminder, not a re-execution.
    calls = {"n": 0}

    def counting_chunks():
        calls["n"] += 1
        return FAKE_CHUNKS

    monkeypatch.setattr(tools.store, "all_chunks", counting_chunks)
    client = _FakeClient(
        [
            _tool_call_response("read_page", {"url": "https://acme.com/"}),
            _tool_call_response("read_page", {"url": "https://acme.com/"}),  # repeat
            _text_response("done"),
        ]
    )
    contents, steps_log = react.run_react_loop(
        client, model="fake", contents=[], config=None, max_steps=10
    )
    assert calls["n"] == 1  # second call never hit the store
    assert len(steps_log) == 2  # both attempts logged for transparency


def test_loop_respects_max_steps(monkeypatch):
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    # A model that NEVER stops calling tools must still be bounded.
    never_stops = [_tool_call_response("list_pages", {}) for _ in range(100)]
    client = _FakeClient(never_stops)

    _, steps_log = react.run_react_loop(
        client, model="fake", contents=[], config=None, max_steps=4
    )
    assert len(steps_log) == 4  # capped, did not run away
