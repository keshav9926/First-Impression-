# app/observability.py — Phase 8: optional Langfuse tracing.
#
# Same philosophy as events.py: a HARD no-op unless configured. If both
# LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, each report run becomes
# one Langfuse trace — a span tree with every LLM call nested as a generation
# (model, prompt/completion, token usage, latency) — so "why did this report
# take 3 minutes / which model actually answered / how many tokens" stops being
# a mystery. Absent keys, every function here returns instantly and nothing is
# imported from Langfuse at call time, so tests and key-less runs pay nothing.
#
# DESIGN RULES:
#   1. Observability must NEVER break a report. Every Langfuse call is wrapped;
#      an SDK/network error is swallowed (logged once), not raised to the user.
#   2. ONE insertion point per concern: report_trace() wraps a whole run
#      (report.generate_report), record_generation() logs one LLM call from the
#      single choke point (llm_pool.chat + the native Gemini synthesis). New
#      call sites need no plumbing — a generation created inside an active
#      report_trace nests under it automatically via OpenTelemetry context.
#
# CALL FLOW:
#   report.generate_report()      → with report_trace(...) as trace: ...
#   llm_pool.chat() (on success)  → record_generation(name, model, in, out, usage)
#   groq_driver.synthesize()      → record_generation(...) for the native Gemini call

import logging
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger("first_impression.observability")

# Lazily-initialized Langfuse client. None = tracing disabled (no keys, or the
# SDK failed to start). Guarded by _init() so import of this module is cheap and
# never touches the network.
_client = None
_init_done = False


def _init() -> None:
    """Create the Langfuse client once, iff both keys are configured. Any failure
    leaves _client None (tracing disabled) — observability must not break boot."""
    global _client, _init_done
    if _init_done:
        return
    _init_done = True
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return  # not configured → stay a no-op
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_host)
    except Exception:  # bad host, SDK import error, etc. — disable, don't crash
        logger.exception("Langfuse init failed — tracing disabled")
        _client = None


def enabled() -> bool:
    """True when traces will actually be sent. Cheap; safe to call anywhere."""
    _init()
    return _client is not None


@contextmanager
def report_trace(name: str = "first-impression-report", **metadata):
    """Wrap a whole report run in one Langfuse span. LLM generations recorded
    inside (via record_generation) nest under it automatically. Flushes on exit
    so a short-lived request/CLI process actually ships its spans. No-op (yields
    None) when tracing is disabled."""
    if not enabled():
        yield None
        return
    span = None
    try:
        span = _client.start_as_current_observation(
            name=name, as_type="span", metadata=metadata or None
        )
        with span as active:
            yield active
    except Exception:
        logger.exception("report_trace failed — continuing without tracing")
        yield None
    finally:
        try:
            _client.flush()
        except Exception:
            logger.exception("Langfuse flush failed")


def update_trace_io(*, output=None, input=None) -> None:
    """Attach the run's input/output to the active trace span (e.g. the ingested
    URL and the finished report's headline). No-op if disabled."""
    if not enabled():
        return
    try:
        if input is not None:
            _client.update_current_span(input=input)
        if output is not None:
            _client.update_current_span(output=output)
    except Exception:
        logger.exception("update_trace_io failed")


def record_generation(
    *, name: str, model: str, input, output, usage=None, metadata=None
) -> None:
    """Log one completed LLM call as a leaf generation under the active trace.
    `usage` is an OpenAI-style usage object (prompt_tokens/completion_tokens/
    total_tokens) or None. Never raises — observability can't break a report."""
    if not enabled():
        return
    try:
        gen = _client.start_observation(
            name=name, as_type="generation", model=model, input=input, metadata=metadata
        )
        gen.update(output=output, usage_details=_usage_dict(usage))
        gen.end()
    except Exception:
        logger.exception("record_generation failed for %s", model)


def _usage_dict(usage) -> dict | None:
    """OpenAI-style usage object → Langfuse usage_details, tolerant of missing
    fields and of providers that omit usage entirely."""
    if usage is None:
        return None
    try:
        return {
            "input": getattr(usage, "prompt_tokens", None) or 0,
            "output": getattr(usage, "completion_tokens", None) or 0,
            "total": getattr(usage, "total_tokens", None) or 0,
        }
    except Exception:
        return None
