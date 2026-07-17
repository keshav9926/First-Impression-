# tests/conftest.py — suite-wide guards.
#
# The real .env (loaded by pydantic at import) may hold live LANGFUSE_* keys.
# Without this, any test that runs the pipeline would spin up a real Langfuse
# client and ship traces to the cloud mid-test. Blank the keys and reset the
# observability module's cached client before EVERY test, so tracing is off by
# default; the observability tests opt back in with an injected fake client.

import pytest

from app import observability


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch):
    monkeypatch.setattr(observability.settings, "langfuse_public_key", "")
    monkeypatch.setattr(observability.settings, "langfuse_secret_key", "")
    monkeypatch.setattr(observability.settings, "langfuse_base_url", "")
    observability._client = None
    observability._init_done = False
    yield
    observability._client = None
    observability._init_done = False
