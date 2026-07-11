# tests/test_health.py — verifies the app boots and the health endpoint works.
#
# WHAT IT EXERCISES: importing app.main (so any import-time error in ANY
# module fails this test) and the full request→response cycle for GET /health.
#
# HOW: TestClient runs the FastAPI app IN-PROCESS — no server, no port.
# It hands a fake-but-complete HTTP request straight to the app object,
# so routing, the endpoint function, and JSON serialization all really run;
# only the network layer (Uvicorn) is skipped.

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    """GET /health → 200 with the expected body (see main.py health())."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "First Impression"
