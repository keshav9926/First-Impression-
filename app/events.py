"""
===============================================================================
FILE: app/events.py
ORIGIN      : Pipeline components (app.ingestion, app.agent)
PURPOSE     : In-process async event bus for streaming live progress to the Web UI
DESTINATION : app.main streaming endpoints / SSE Clients
===============================================================================
"""

import contextvars
import queue
from contextlib import contextmanager

# Holds the active sink (a callable) or None. Per-context, so concurrent
# requests each see their own collector without cross-talk.
_sink: contextvars.ContextVar = contextvars.ContextVar("event_sink", default=None)


def emit(event_type: str, **data) -> None:
    """Publish one progress event. No-op when nobody is collecting.

    Called by: pipeline components (crawl, explore loop, panel, ingest).
    `event_type` is a short string the UI switches on ("crawl.page", "tool",
    "persona", "report.done", ...); data is arbitrary JSON-serializable kwargs.
    """
    sink = _sink.get()
    if sink is not None:
        sink({"type": event_type, **data})


@contextmanager
def collector(q: queue.Queue | None = None):
    """Activate collection for the current context; yield the Queue of events.

    Called by: the SSE endpoint's worker thread. Pass a shared queue so the
    consuming generator (running on a DIFFERENT thread) can drain it. On exit,
    restores the previous sink so nothing leaks between requests.
    """
    q = q or queue.Queue()
    token = _sink.set(q.put)
    try:
        yield q
    finally:
        _sink.reset(token)
