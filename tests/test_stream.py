# tests/test_stream.py — event bus + SSE streaming, no network.

import json

from fastapi.testclient import TestClient

from app import events, main


def test_emit_is_noop_without_collector():
    events.emit("tool", name="x")  # must not raise, no listener


def test_collector_captures_emitted_events():
    with events.collector() as q:
        events.emit("crawl.page", url="https://a.com/", chars=10)
        events.emit("tool", name="read_page")
    assert q.get()["type"] == "crawl.page"
    assert q.get()["type"] == "tool"


def test_dashboard_served():
    client = TestClient(main.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "First Impression" in r.text
    assert "text/html" in r.headers["content-type"]


def test_analyze_stream_emits_events_then_report(monkeypatch):
    # Replace the pipeline seams with instant fakes that emit through the bus.
    monkeypatch.setattr(main.settings, "voyage_api_key", "test-key")

    def fake_ingest(url, max_pages):
        events.emit("crawl.page", url=url, chars=42)
        from app.schemas import IngestResponse

        return IngestResponse(pages_fetched=1, chunks_stored=1, skipped_by_robots=0)

    def fake_report(panel=False, mode="normal"):
        from app.schemas import FirstImpressionReport

        events.emit("tool", name="list_pages", args={})
        rep = FirstImpressionReport(
            company="Acme", what_the_product_is=[], likely_new_user_journey=[],
            friction_points=[], standout_strengths=[], unanswered_questions=[],
            scope_note="public only",
        )
        return rep, [{"tool": "list_pages", "args": {}}], []

    monkeypatch.setattr(main, "_ingest_site", fake_ingest)
    monkeypatch.setattr(main, "generate_report", fake_report)

    client = TestClient(main.app)
    with client.stream("GET", "/analyze/stream?url=https://acme.com/&panel=false") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        types = []
        for raw in r.iter_lines():
            if raw and raw.startswith("data: "):
                types.append(json.loads(raw[6:])["type"])
                if types[-1] == "report.done":
                    break
    assert "crawl.page" in types
    assert "tool" in types
    assert "report.done" in types
