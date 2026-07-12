# app/agent/llm.py — one wrapper around Gemini generate_content that survives
# free-tier rate limits.
#
# WHY THIS EXISTS: the ReAct agent makes many Gemini calls in quick succession
# (one per reasoning step + the final synthesis). Gemini's free tier allows
# only ~5 requests/minute, so an un-paced agent 429s within seconds — exactly
# the Voyage lesson again, one layer up.
#
# THE FIX — honor the server's Retry-After: on a 429 the API tells us when a
# quota slot frees ("Please retry in 1.9s"). We parse that and sleep exactly
# that long, so the agent runs at FULL speed when quota is available and paces
# itself to the limit only when it must. This beats a fixed sleep-between-calls
# (which is always slow) and is the general pattern for any rate-limited API:
# respect Retry-After, don't guess.
#
# CALL FLOW:
#   react.py loop      → generate_with_retry(...)   (every exploration step)
#   report.py synthesis→ generate_with_retry(...)   (the final schema call)

import re
import time

from google.genai import errors

MAX_RETRIES = 8
_MAX_SERVER_WAIT = 120  # honor a server retry hint up to this (per-minute windows)
_FALLBACK_MAX_SLEEP = 30  # cap for the exponential fallback when no hint is given


def _retry_delay_seconds(exc: Exception) -> float:
    """Pull the server-suggested wait out of a 429 error message, e.g.
    'Please retry in 1.926s' → 2.9 (with a 1s cushion). 0.0 if not found."""
    match = re.search(r"retry in ([0-9.]+)s", str(exc))
    return float(match.group(1)) + 1.0 if match else 0.0


def generate_with_retry(client, model: str, contents: list, config):
    """client.models.generate_content, but retrying free-tier 429s.

    Non-429 errors propagate immediately (a real bug shouldn't be retried).
    After MAX_RETRIES exhausted 429s, the last error propagates too — which is
    what a truly exhausted DAILY quota looks like (the server keeps asking us
    to wait longer than a per-minute window; retries can't fix "come back
    tomorrow"). We honor the server's own retry hint rather than guessing.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.ClientError as exc:
            is_last = attempt == MAX_RETRIES - 1
            if getattr(exc, "code", None) != 429 or is_last:
                raise
            hint = _retry_delay_seconds(exc)
            # Server hint (per-minute windows) honored up to _MAX_SERVER_WAIT;
            # exponential fallback only when the server gave no number.
            delay = min(hint, _MAX_SERVER_WAIT) if hint else min(2**attempt, _FALLBACK_MAX_SLEEP)
            time.sleep(delay)
