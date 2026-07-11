# tests/test_health.py — verifies the app boots and the health endpoint works.
# TestClient runs the FastAPI app in-process (no server, no port), sends a
# real HTTP request through the full request/response cycle, and lets us
# assert on the result.

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "First Impression"
