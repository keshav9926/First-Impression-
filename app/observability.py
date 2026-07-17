# app/observability.py — Phase 8: Langfuse tracing, following the official
# Langfuse instrumentation best-practices (github.com/langfuse/skills).
#
# Same philosophy as events.py: a HARD no-op unless configured. With both
# LANGFUSE_* keys set, each report run is ONE trace — a span tree whose LLM
# calls are captured as proper `generation` observations (model, tokens, cost,
# latency) by the Langfuse OpenAI drop-in, with the persona subagents typed as
# `agent` and retrieval typed as `retriever` so the trace reads correctly and
# drives Langfuse's Agent Graph.
#
# WHY the drop-in over manual logging: Langfuse's baseline guidance is "prefer
# framework integrations over manual instrumentation — they capture more context
# with less code." Our OpenAI-compatible providers (NVIDIA GLM/DeepSeek/Nemotron/
# Mistral + Gemini) all go through openai.OpenAI, so swapping that class for
# langfuse.openai.OpenAI auto-instruments every generation. Only Groq (its own
# SDK, deep fallback) is logged manually.
#
# ONE Langfuse client: we construct it here, which also registers it as the SDK
# default singleton the drop-in resolves via get_client() — so the drop-in's
# generations nest under the spans we open here (shared OpenTelemetry context).
#
# DESIGN RULES:
#   1. Observability must NEVER break a report — every Langfuse call is wrapped.
#   2. Import Langfuse only AFTER config is loaded, and construct the client
#      BEFORE any langfuse.openai client is built (drop-in patch order).
#
# CALL FLOW:
#   report.generate_report()  → with report_trace(...) as trace: ...
#   llm_pool._client()        → openai_client_class() (drop-in when enabled)
#   llm_pool.chat()           → passes name=/metadata= so each generation is named
#   panel persona node        → with span(name, as_type="agent"): ...
#   rag.pipeline.retrieve()   → with span("retrieve-context", as_type="retriever")

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
    try:
        with _client.start_as_current_observation(
            name=name, as_type="span", metadata=metadata or None
        ) as active:
            yield active
    except Exception:
        logger.exception("report_trace failed — continuing without tracing")
        yield None
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
    try:
        with _client.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=metadata
        ) as obs:
            yield obs
    except Exception:
        logger.exception("span %s failed — continuing", name)
        yield None


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
