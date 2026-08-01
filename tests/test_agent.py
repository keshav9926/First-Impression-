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


def test_list_pages_warns_on_thin_extraction(monkeypatch):
    # JS-rendered site → flag on chunks → the agent's FIRST observation warns
    # it not to convert crawler blindness into "the site doesn't mention X".
    flagged = [{**c, "extraction_warning": True} for c in FAKE_CHUNKS]
    monkeypatch.setattr(tools.store, "all_chunks", lambda: flagged)
    out = tools.execute_tool("list_pages", {})
    assert "WARNING" in out and "JavaScript-rendered" in out
    assert "https://acme.com/pricing" in out  # pages still listed


def test_list_pages_no_warning_on_normal_site(monkeypatch):
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("list_pages", {})
    assert "WARNING" not in out


def test_read_page_truncation_shows_section_map(monkeypatch):
    # A long page gets cut at READ_PAGE_MAX_CHARS — the section map must reveal
    # what exists beyond the cut so the model can search into it.
    long_chunks = [
        {
            "id": "chunk-0",
            "text": "word " * 4000,  # 20K chars → past READ_PAGE_MAX_CHARS (15K)
            "url": "https://acme.com/docs",
            "headings": "Setup · Deployment · Access & roles",
        }
    ]
    monkeypatch.setattr(tools.store, "all_chunks", lambda: long_chunks)
    out = tools.execute_tool("read_page", {"url": "https://acme.com/docs"})
    assert "Sections on this page: Setup · Deployment · Access & roles" in out
    assert "page truncated" in out


def test_read_page_short_page_skips_the_map(monkeypatch):
    # Fully-visible page → no truncation → no map (no wasted tokens).
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("read_page", {"url": "https://acme.com/"})
    assert "Sections on this page" not in out


def test_is_thin_extraction_calibrated_on_real_sites():
    from app.ingestion.fetcher import _is_thin_extraction

    # Real measurements (2026-07-15):
    assert _is_thin_extraction(368, 388_217) is True  # trynarrative (Framer/JS)
    assert _is_thin_extraction(2200, 229_954) is False  # vortexify (server-rendered)


def test_is_thin_extraction_is_failsafe_on_text_only():
    # NVIDIA-era rule (2026-07-18): thin is decided by the single robust signal —
    # too little seed text — and the fragile ratio condition is gone. Escalating
    # to render is cheap and safe, so we err toward True on low text.
    from app.ingestion.fetcher import _is_thin_extraction

    # Plenty of text → NOT thin, whatever the ratio (bloated HTML is fine).
    assert _is_thin_extraction(5000, 2_000_000) is False
    # Little text → thin now, REGARDLESS of ratio (was False under the old
    # both-signals rule; this is the fix for partly-rendered SPAs slipping by).
    assert _is_thin_extraction(300, 6_000) is True
    assert _is_thin_extraction(300, 400_000) is True
    # Just under / over the 1200-char bar.
    assert _is_thin_extraction(1199, 500_000) is True
    assert _is_thin_extraction(1200, 500_000) is False
    # No HTML at all → not thin (dead page, not a JS shell).
    assert _is_thin_extraction(0, 0) is False


def test_read_page_surfaces_ctas(monkeypatch):
    # CTAs (stripped from body by boilerplate removal) must appear on read —
    # the fix for the false "no signup button" persona verdict.
    chunks = [{**FAKE_CHUNKS[0], "ctas": "Try for free · Book a demo · Sign in"}]
    monkeypatch.setattr(tools.store, "all_chunks", lambda: chunks)
    out = tools.execute_tool("read_page", {"url": "https://acme.com/"})
    assert "Primary actions available on this page: Try for free" in out


def test_read_page_no_cta_line_when_absent(monkeypatch):
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("read_page", {"url": "https://acme.com/"})
    assert "Primary actions" not in out


def test_extract_ctas_matches_signup_family():
    from app.ingestion.fetcher import _extract_ctas

    html = """
    <header>
      <a href="/signup"><span>Try for free</span></a>
      <button>Book a Demo</button>
      <a href="/login">Sign in</a>
      <a href="/blog">Read our blog</a>   <!-- not a CTA -->
      <a href="/x">Try for free</a>        <!-- dup, deduped -->
    </header>
    """
    ctas = _extract_ctas(html)
    assert ctas == ["Try for free", "Book a Demo", "Sign in"]


def test_split_into_sections_anchors_on_body_not_nav():
    """A docs page lists every heading in its sidebar BEFORE any content, so
    anchoring on a heading's first occurrence yields 25-char 'sections'. We
    anchor on the last, and cite the page's own element ids."""
    from app.ingestion.fetcher import Page, _split_into_sections

    body_a = "Data flows in from your warehouse on a schedule. " * 20
    body_b = "Each run is queued, retried on failure, and logged. " * 20
    body_c = "Permissions gate who can see which dashboard. " * 20
    text = (
        "Documentation\nConnectors\nSync jobs\nAccess roles\n"  # sidebar nav
        f"Connectors\n{body_a}\nSync jobs\n{body_b}\nAccess roles\n{body_c}"
    )
    html = """
    <h1>Connectors</h1><h2>Sync jobs</h2><h2>Access roles</h2>
    <div id="connectors"></div><div id="core-sync-jobs"></div><div id="access-roles"></div>
    """
    page = Page(url="https://acme.com/docs", text=text, headings=[], ctas=["Sign up"])
    out = _split_into_sections(page, html)

    assert [p.url for p in out] == [
        "https://acme.com/docs",
        "https://acme.com/docs#connectors",
        "https://acme.com/docs#core-sync-jobs",  # id suffix-matched to "Sync jobs"
        "https://acme.com/docs#access-roles",
    ]
    # Sections carry the BODY text, not the sidebar entry.
    assert body_a.strip() in out[1].text
    assert len(out[1].text) > 500
    # The parent keeps the opening and the page-level visuals/CTAs.
    assert out[0].text.startswith("Documentation")
    assert all(p.ctas == ["Sign up"] for p in out)
    # Every character of the body survives somewhere — splitting must not drop text.
    assert body_c.strip() in out[3].text


def test_split_into_sections_declines_when_unstructured():
    from app.ingestion.fetcher import Page, _split_into_sections

    page = Page(url="https://acme.com/x", text="flat text " * 500, headings=[], ctas=[])
    assert _split_into_sections(page, "<p>no headings here</p>") == [page]


def test_pages_examined_counts_only_real_pages_read(monkeypatch):
    """Two ways this count lied. Sections: reading three sections of /docs is
    reading ONE page. Ghosts: a model guessing /about on a site without one had
    the ATTEMPT counted — 39 pages examined on a site with 18 (2026-08-02)."""
    from app.agent.groq_driver import pages_from_steps

    chunks = [
        {"id": "chunk-0", "text": "a", "url": "https://acme.com/docs"},
        {"id": "chunk-1", "text": "b", "url": "https://acme.com/docs#connectors"},
        {"id": "chunk-2", "text": "c", "url": "https://acme.com/pricing"},
    ]
    monkeypatch.setattr(tools.store, "all_chunks", lambda: chunks)
    steps = [
        {"tool": "read_page", "args": {"url": "https://acme.com/docs"}},
        {"tool": "read_page", "args": {"url": "https://acme.com/docs#connectors"}},
        {"tool": "read_page", "args": {"url": "https://acme.com/about"}},    # ghost
        {"tool": "read_page", "args": {"url": "https://acme.com/careers"}},  # ghost
        {"tool": "read_page", "args": {"url": "pricing"}},   # bare slug — recovered
        {"tool": "search_content", "args": {"query": "onboarding"}},
    ]
    assert pages_from_steps(steps) == ["https://acme.com/docs", "https://acme.com/pricing"]


def test_citation_to_a_section_anchor_is_grounded():
    """Sections cite url#anchor; that must verify against the stored parent page
    rather than being dropped as a hallucinated url."""
    from app.agent import grounding

    assert grounding._normalize("https://acme.com/docs#connectors") == grounding._normalize(
        "https://acme.com/docs"
    )


def test_list_pages_nests_sections_under_their_parent(monkeypatch):
    chunks = [
        {"id": "chunk-0", "text": "a", "url": "https://acme.com/"},
        {"id": "chunk-1", "text": "b", "url": "https://acme.com/docs"},
        {"id": "chunk-2", "text": "c", "url": "https://acme.com/docs#connectors"},
        {"id": "chunk-3", "text": "d", "url": "https://acme.com/docs#sync-jobs"},
    ]
    monkeypatch.setattr(tools.store, "all_chunks", lambda: chunks)
    out = tools.execute_tool("list_pages", {})
    # 2 real pages, not 4 — the site's shape is the report's subject.
    assert "(2 pages" in out
    assert "    - https://acme.com/docs#connectors" in out


def test_extract_images_finds_embedded_and_self_hosted_videos():
    """Demo videos are the visual the report cares most about, and marketing
    sites ship them as player iframes / lazy data-* mounts, not <video> tags."""
    from app.ingestion.fetcher import _extract_images

    html = """
    <img src="/dash.png" alt="Dashboard overview" width="1200" height="800">
    <iframe title="Product demo" src="https://www.youtube.com/embed/dQw4w9WgXcQ?rel=0"></iframe>
    <div data-video-url="https://player.vimeo.com/video/76979871"></div>
    <video poster="/poster.jpg">
      <source src="/media/hero.webm">
      <source src="/media/hero.mp4" type="video/mp4">
    </video>
    <a href="https://www.loom.com/share/abc123">Watch the walkthrough</a>
    <a href="https://youtube.com/@acme">Our channel</a>   <!-- social, not a demo -->
    """
    labels, image_urls, video_urls = _extract_images(html, base_url="https://acme.com/")

    videos = [v for v in labels if v.startswith("[video")]
    assert videos == [
        '[video: YouTube embed — "Product demo"]',
        "[video: Vimeo embed]",
        "[video: hero.mp4]",   # one element = one video, mp4 preferred over webm
        "[video: Loom embed]",
    ]
    # Only the downloadable file is watchable; embeds fall back to poster frames.
    assert video_urls == ["https://acme.com/media/hero.mp4"]
    assert "https://acme.com/poster.jpg" in image_urls
    assert "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg" in image_urls


def test_extract_headings_pulls_h1_to_h3_in_order():
    from app.ingestion.fetcher import _extract_headings

    html = """
    <html><body>
      <h1>Docs</h1>
      <p>intro text</p>
      <h2>Setup</h2><p>...</p>
      <h3>Install</h3>
      <h2>Deployment</h2>
      <h2>Setup</h2>          <!-- duplicate: kept once -->
      <h4>too deep</h4>       <!-- h4 ignored -->
    </body></html>
    """
    assert _extract_headings(html) == ["Docs", "Setup", "Install", "Deployment"]


def test_read_page_recovers_from_a_bare_slug(monkeypatch):
    # Models often pass "pricing" instead of the exact URL. An unambiguous
    # slug should resolve to the real page instead of wasting a step.
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("read_page", {"url": "pricing"})
    assert "$20 per month" in out
    assert "No page found" not in out


def test_read_page_recovers_from_hallucinated_domain(monkeypatch):
    # Models sometimes invent a placeholder domain ("example.com") instead of
    # the real URL — recover by matching the last path segment.
    monkeypatch.setattr(tools.store, "all_chunks", lambda: FAKE_CHUNKS)
    out = tools.execute_tool("read_page", {"url": "https://example.com/pricing"})
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
