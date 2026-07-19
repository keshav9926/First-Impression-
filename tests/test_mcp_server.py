# tests/test_mcp_server.py — Phase 7: the MCP server front door.
#
# The tools are thin wrappers over the same pipeline the API tests already
# cover, so these tests verify ONLY the wrapper contract: key guards, the
# robots/insufficient-evidence refusals, the happy-path shape, and the
# relevance gate — all with the heavy pipeline monkeypatched (no network).

import pytest

from app import mcp_server
from app.agent.report import InsufficientEvidenceError
from app.schemas import FirstImpressionReport, Observation


@pytest.fixture(autouse=True)
def _keys_present(monkeypatch):
    """Default every test to a fully-configured server (Voyage + the NVIDIA
    pool key). Guard tests blank a key explicitly."""
    monkeypatch.setattr(mcp_server.settings, "voyage_api_key", "vk")
    monkeypatch.setattr(mcp_server.settings, "nvidia_api_key", "nk")


def _fake_report() -> FirstImpressionReport:
    obs = Observation(claim="c", evidence="e", source_url="https://x.co")
    return FirstImpressionReport(
        company="Acme",
        what_the_product_is=[obs],
        likely_new_user_journey=[obs],
        friction_points=[],
        standout_strengths=[obs],
        unanswered_questions=["q?"],
        scope_note="public surface only",
    )


# ----- analyze_first_impression -----


def test_analyze_errors_when_voyage_key_missing(monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "voyage_api_key", "")
    out = mcp_server.analyze_first_impression("https://acme.co")
    assert out["status"] == "error"
    assert "VOYAGE_API_KEY" in out["reason"]


def test_analyze_refuses_robots_blocked(monkeypatch):
    monkeypatch.setattr(mcp_server, "is_allowed", lambda url: False)
    out = mcp_server.analyze_first_impression("https://acme.co")
    assert out["status"] == "error"
    assert "robots.txt" in out["reason"]


def test_analyze_reports_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(mcp_server, "is_allowed", lambda url: True)
    monkeypatch.setattr(mcp_server, "_ingest_site", lambda url, mp: None)

    def _boom(panel, mode="normal"):
        raise InsufficientEvidenceError("nothing to ground")

    monkeypatch.setattr(mcp_server, "generate_report", _boom)
    out = mcp_server.analyze_first_impression("https://acme.co")
    assert out["status"] == "insufficient_evidence"
    assert "nothing to ground" in out["reason"]


def test_analyze_happy_path(monkeypatch):
    monkeypatch.setattr(mcp_server, "is_allowed", lambda url: True)
    monkeypatch.setattr(mcp_server, "_ingest_site", lambda url, mp: None)
    steps = [{"tool": "read_page", "args": {"url": "https://acme.co"}}]
    monkeypatch.setattr(
        mcp_server, "generate_report", lambda panel, mode="normal": (_fake_report(), steps, ["https://acme.co"])
    )
    out = mcp_server.analyze_first_impression("https://acme.co")
    assert out["status"] == "ok"
    assert out["report"]["company"] == "Acme"
    assert out["steps_taken"] == 1
    assert out["pages_examined"] == ["https://acme.co"]
    assert out["tool_calls"] == ["read_page({'url': 'https://acme.co'})"]


def test_analyze_clamps_max_pages(monkeypatch):
    seen = {}
    monkeypatch.setattr(mcp_server, "is_allowed", lambda url: True)
    monkeypatch.setattr(mcp_server, "_ingest_site", lambda url, mp: seen.update(mp=mp))
    monkeypatch.setattr(mcp_server, "generate_report", lambda panel, mode="normal": (_fake_report(), [], []))
    mcp_server.analyze_first_impression("https://acme.co", max_pages=9999)
    assert seen["mp"] == mcp_server.settings.max_pages_hard_limit


# ----- ask_ingested -----


def test_ask_errors_when_store_empty(monkeypatch):
    monkeypatch.setattr(mcp_server.store, "count", lambda: 0)
    out = mcp_server.ask_ingested("what is this?")
    assert out["status"] == "error"
    assert "Nothing ingested" in out["reason"]


def test_ask_refuses_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr(mcp_server.store, "count", lambda: 5)
    monkeypatch.setattr(mcp_server.settings, "min_relevance", 0.45)
    # Everything scores below the bar → honest refusal, no qa.answer call.
    monkeypatch.setattr(
        mcp_server.pipeline, "retrieve", lambda q, top_k: [{"relevance": 0.1, "url": "u", "text": "t"}]
    )
    out = mcp_server.ask_ingested("what is this?")
    assert out["status"] == "no_relevant_content"
    assert out["sources"] == []


def test_ask_happy_path(monkeypatch):
    monkeypatch.setattr(mcp_server.store, "count", lambda: 5)
    monkeypatch.setattr(mcp_server.settings, "min_relevance", 0.45)
    monkeypatch.setattr(
        mcp_server.pipeline,
        "retrieve",
        lambda q, top_k: [{"relevance": 0.9, "url": "https://acme.co", "text": "Acme builds widgets."}],
    )
    monkeypatch.setattr(mcp_server.qa, "answer", lambda q, hits: "Acme builds widgets [1].")
    out = mcp_server.ask_ingested("what is this?")
    assert out["status"] == "ok"
    assert out["answer"] == "Acme builds widgets [1]."
    assert out["sources"][0] == {"index": 1, "url": "https://acme.co", "snippet": "Acme builds widgets."}


# ----- ingestion_status -----


def test_ingestion_status_reports_pages(monkeypatch):
    monkeypatch.setattr(
        mcp_server.store,
        "all_chunks",
        lambda: [{"url": "https://acme.co/a"}, {"url": "https://acme.co/b"}, {"url": "https://acme.co/a"}],
    )
    out = mcp_server.ingestion_status()
    assert out["status"] == "ok"
    assert out["chunks_stored"] == 3
    assert out["pages"] == ["https://acme.co/a", "https://acme.co/b"]
