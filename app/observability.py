"""
===============================================================================
FILE: app/observability.py
ORIGIN      : FastAPI routes / Agent report workflow / RAG execution
PURPOSE     : Telemetry and tracing integration via Langfuse (spans & generations)
DESTINATION : Langfuse Cloud / Local logs
===============================================================================
"""

import logging
import os
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger("first_impression.observability")

_client = None
_init_done = False


def _resolve_host() -> str:
    """LANGFUSE_BASE_URL (skill convention) wins over LANGFUSE_HOST (SDK
    convention); default is the EU cloud."""
    return settings.langfuse_base_url or settings.langfuse_host


def _init() -> None:
    """Create the Langfuse client once, iff both keys are set. Registers it as
    the SDK default so the OpenAI drop-in shares it. Any failure leaves tracing
    disabled — observability must not break boot."""
    global _client, _init_done
    if _init_done:
        return
    _init_done = True
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return  # not configured → stay a no-op
    try:
        host = _resolve_host()
        # Mirror the resolved host into the env the drop-in/get_client() read, so
        # every code path agrees on the region even if only BASE_URL was set.
        os.environ.setdefault("LANGFUSE_HOST", host)
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=host,
        )
        logger.info("Langfuse tracing enabled (host=%s)", host)
    except Exception:  # bad host, SDK import error, etc. — disable, don't crash
        logger.exception("Langfuse init failed — tracing disabled")
        _client = None


def enabled() -> bool:
    """True when traces will actually be sent. Cheap; safe to call anywhere."""
    _init()
    return _client is not None


def openai_client_class():
    """The OpenAI class to instantiate for OpenAI-compatible providers: the
    Langfuse drop-in (auto-instruments generations) when tracing is on, else the
    plain openai.OpenAI. Lazy import so key-less runs never import langfuse."""
    if enabled():
        try:
            from langfuse.openai import OpenAI

            return OpenAI
        except Exception:
            logger.exception("langfuse.openai unavailable — using plain OpenAI client")
    import openai

    return openai.OpenAI


@contextmanager
def report_trace(name: str = "analyze-first-impression", **metadata):
    """Wrap a whole report run in one root span; LLM generations recorded inside
    (drop-in or manual) nest under it. Flushes on exit so a short-lived request/
    CLI process ships its spans. No-op (yields None) when tracing is disabled."""
    if not enabled():
        yield None
        return
    # Guard ONLY Langfuse's own machinery. The caller's body must never be
    # caught here: swallowing it and yielding again turns every real pipeline
    # error (a 400, a quota 429) into "generator didn't stop after throw()" —
    # which masked real failures and broke main.py's error mapping.
    try:
        cm = _client.start_as_current_observation(
            name=name, as_type="span", metadata=metadata or None
        )
        active = cm.__enter__()
    except Exception:
        logger.exception("report_trace failed — continuing without tracing")
        yield None
        return
    try:
        yield active  # body exceptions propagate to the caller untouched
    except BaseException as exc:
        try:
            cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            logger.exception("Langfuse span exit failed")
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.exception("Langfuse span exit failed")
    finally:
        try:
            _client.flush()
        except Exception:
            logger.exception("Langfuse flush failed")


@contextmanager
def span(name: str, as_type: str = "span", input=None, metadata=None):
    """A nested observation of a specific type — `agent` for a subagent,
    `retriever` for a lookup, `tool` for a tool call, `span` for a plain step.
    Becomes the current observation so generations inside nest under it. Yields
    the observation (or None) so the caller can set .update(output=...). Never
    raises."""
    if not enabled():
        yield None
        return
    # Same body-vs-machinery separation as report_trace: never swallow the
    # caller's exception (that masks real errors); only guard Langfuse itself.
    try:
        cm = _client.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=metadata
        )
        obs = cm.__enter__()
    except Exception:
        logger.exception("span %s failed — continuing", name)
        yield None
        return
    try:
        yield obs
    except BaseException as exc:
        try:
            cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            logger.exception("span %s exit failed", name)
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.exception("span %s exit failed", name)


def update_trace_io(*, input=None, output=None) -> None:
    """Attach meaningful input/output to the CURRENT span (best practice: the
    root's I/O becomes the trace's I/O — set it to what a reviewer needs, not raw
    args). No-op if disabled."""
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
    """Manually log one completed LLM call as a leaf generation under the active
    trace. Used ONLY for providers the OpenAI drop-in cannot patch (Groq; the
    native google-genai synthesis) — OpenAI-compatible calls are auto-traced by
    the drop-in. Never raises."""
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
