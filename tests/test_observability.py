# tests/test_observability.py — Phase 8: the Langfuse tracing facade.
#
# No network and no Langfuse account: a fake client stands in for the SDK. The
# contract under test is the one the rest of the app relies on — tracing is a
# hard no-op without keys, and once enabled it NEVER raises into a report.

import pytest

from app import observability


@pytest.fixture(autouse=True)
def _reset_module_state():
    """observability caches its client + init flag at module scope. Reset around
    each test so one test's fake client can't leak into the next."""
    observability._client = None
    observability._init_done = False
    yield
    observability._client = None
    observability._init_done = False


class _FakeSpan:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self.log.append("enter")
        return self

    def __exit__(self, *exc):
        self.log.append("exit")
        return False

    def update(self, **kw):
        self.log.append(("update", kw))

    def end(self):
        self.log.append("end")


class _FakeClient:
    def __init__(self):
        self.log = []

    def start_as_current_observation(self, **kw):
        self.log.append(("span", kw))
        return _FakeSpan(self.log)

    def start_observation(self, **kw):
        self.log.append(("obs", kw))
        return _FakeSpan(self.log)

    def update_current_span(self, **kw):
        self.log.append(("update_span", kw))

    def flush(self):
        self.log.append("flush")


def _enable(monkeypatch):
    """Bypass real init and install a fake client — simulates configured keys."""
    fake = _FakeClient()
    monkeypatch.setattr(observability, "_client", fake)
    monkeypatch.setattr(observability, "_init_done", True)
    return fake


# ----- disabled (no keys) = hard no-op -----


def test_disabled_without_keys(monkeypatch):
    monkeypatch.setattr(observability.settings, "langfuse_public_key", "")
    monkeypatch.setattr(observability.settings, "langfuse_secret_key", "")
    assert observability.enabled() is False


def test_noop_functions_do_nothing_when_disabled(monkeypatch):
    monkeypatch.setattr(observability.settings, "langfuse_public_key", "")
    monkeypatch.setattr(observability.settings, "langfuse_secret_key", "")
    with observability.report_trace(panel=True) as t:
        assert t is None
        observability.update_trace_io(input={"x": 1}, output={"y": 2})
        observability.record_generation(name="c", model="m", input=[], output="o")
    # nothing raised, nothing sent — that's the whole assertion


# ----- enabled = spans created + flushed -----


def test_report_trace_creates_span_and_flushes(monkeypatch):
    fake = _enable(monkeypatch)
    with observability.report_trace(panel=False, chunks=3):
        pass
    kinds = [e[0] if isinstance(e, tuple) else e for e in fake.log]
    assert "span" in kinds
    assert "flush" in kinds  # short-lived process must ship its spans


def test_record_generation_logs_usage(monkeypatch):
    fake = _enable(monkeypatch)

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    observability.record_generation(
        name="chat:glm", model="z-ai/glm-5.2", input=[{"role": "user"}], output="hi", usage=_Usage()
    )
    obs = [e for e in fake.log if isinstance(e, tuple) and e[0] == "obs"]
    updates = [e for e in fake.log if isinstance(e, tuple) and e[0] == "update"]
    assert obs and obs[0][1]["as_type"] == "generation"
    assert updates[0][1]["usage_details"] == {"input": 10, "output": 5, "total": 15}


def test_usage_dict_tolerates_none():
    assert observability._usage_dict(None) is None


# ----- observation typing contract (best-practice: correct as_type) -----


def test_span_forwards_type_and_io(monkeypatch):
    """span() must forward as_type + input so retrieval is typed `retriever` and
    personas `agent` (drives the Agent Graph), and let the caller set output."""
    fake = _enable(monkeypatch)
    with observability.span("retrieve-context", as_type="retriever", input="q") as s:
        s.update(output=[{"url": "/", "relevance": 0.9}])
    obs = [e for e in fake.log if isinstance(e, tuple) and e[0] == "span"]
    assert obs[0][1]["as_type"] == "retriever"
    assert obs[0][1]["input"] == "q"
    updates = [e for e in fake.log if isinstance(e, tuple) and e[0] == "update"]
    assert updates[0][1]["output"] == [{"url": "/", "relevance": 0.9}]


def test_agent_span_type(monkeypatch):
    fake = _enable(monkeypatch)
    with observability.span("persona:Technical Evaluator", as_type="agent"):
        pass
    obs = [e for e in fake.log if isinstance(e, tuple) and e[0] == "span"]
    assert obs[0][1]["as_type"] == "agent"
    assert obs[0][1]["name"] == "persona:Technical Evaluator"


def test_span_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(observability.settings, "langfuse_public_key", "")
    monkeypatch.setattr(observability.settings, "langfuse_secret_key", "")
    with observability.span("x", as_type="agent") as s:
        assert s is None  # nothing created, caller's `if s:` guard short-circuits


def test_base_url_beats_host(monkeypatch):
    """LANGFUSE_BASE_URL (skill convention) must win over LANGFUSE_HOST."""
    monkeypatch.setattr(observability.settings, "langfuse_host", "https://cloud.langfuse.com")
    monkeypatch.setattr(observability.settings, "langfuse_base_url", "https://us.cloud.langfuse.com")
    assert observability._resolve_host() == "https://us.cloud.langfuse.com"


# ----- never raises into a report -----


def test_record_generation_swallows_sdk_errors(monkeypatch):
    class _Boom:
        def start_observation(self, **kw):
            raise RuntimeError("langfuse down")

    monkeypatch.setattr(observability, "_client", _Boom())
    monkeypatch.setattr(observability, "_init_done", True)
    # Must NOT propagate — observability can't break a report.
    observability.record_generation(name="c", model="m", input=[], output="o")
