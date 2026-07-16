# app/events.py — a tiny in-process event bus for streaming progress to the UI.
#
# WHY: the pipeline (crawl → ingest → explore → panel → synthesize) runs for
# 2-4 minutes and, until now, told the caller NOTHING until it finished. The
# dashboard needs to show each step live. Rather than thread an `emit` argument
# through a dozen function signatures, components call events.emit(...) freely;
# a contextvar decides whether anyone is listening.
#
# DEFAULT = NO-OP: if no collector is active (the plain /report path, tests,
# the CLI), emit() does nothing and costs almost nothing. Only the SSE endpoint
# activates a collector, so instrumentation is invisible everywhere else.
#
# THREADING: the streaming endpoint runs the sync pipeline in ONE worker thread
# and calls collector() INSIDE that thread, so the contextvar is visible to
# every pipeline call on that thread. Events are pushed to a thread-safe Queue
# the SSE generator drains.
#
# CALL FLOW:
#   fetcher/groq_driver/panel → events.emit("tool", name=...)   (producers)
#   main.analyze_stream → with events.collector() as q: ...      (consumer)

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
