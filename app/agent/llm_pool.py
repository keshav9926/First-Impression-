# app/agent/llm_pool.py — one chat() over TWO free-tier providers (Groq +
# Cerebras), with automatic failover.
#
# WHY: 20 cold-outreach reports/day don't fit inside ONE provider's free daily
# token cap (Groq: 100K TPD — one heavy day of testing exhausted it). Both
# providers speak the OpenAI chat.completions dialect, so a single call-site
# can prefer one and fail over to the other:
#   - per-MINUTE 429  → sleep the server's Retry-After, same provider
#   - per-DAY 429     → sleeping won't help TODAY → switch provider immediately
#   - tool_use_failed 400 → stochastic malformed tool-call syntax → re-ask
#
# WHO PREFERS WHAT (spreads the load BY DESIGN, not just on failure):
#   explore loop (groq_driver) → prefer "groq"     (proven tool-calling)
#   personas + judge           → prefer "cerebras" (JSON verdicts; keeps the
#                                                   explore budget on Groq)
#   synthesis                  → Gemini, NOT here (response_schema quality;
#                                                   1 call/report ≈ 20/day cap)
#
# CALL FLOW:
#   groq_driver.explore() / panel._judge_as() / judge.verify_groundedness()
#     → chat(messages, prefer=..., tools=/response_format=...)

import logging
import re
import time

import groq
import openai

from app.config import settings

logger = logging.getLogger("first_impression")

_MAX_RATE_RETRIES = 6
_MAX_FORMAT_RETRIES = 3
_MAX_SERVER_RETRIES = 3  # transient 5xx (provider hiccup) — retry, then fail over
_MAX_EMPTY_RETRIES = 3  # blank 200 completion (seen on Cerebras) — retry, then fail over

# Daily-exhaustion markers in provider 429 messages: retrying is pointless
# until tomorrow, so switch providers instead of sleeping.
_DAILY_MARKERS = ("per day", "tokens per day", "tpd", "rpd", "daily")


def _client(provider: str):
    """OpenAI-compatible client for a provider. Both SDKs expose the same
    .chat.completions.create surface, which is what makes this pool possible."""
    if provider == "groq":
        return groq.Groq(api_key=settings.groq_api_key)
    return openai.OpenAI(
        base_url="https://api.cerebras.ai/v1", api_key=settings.cerebras_api_key
    )


def _model(provider: str) -> str:
    return settings.groq_model if provider == "groq" else settings.cerebras_model


def _retry_after_seconds(exc: Exception) -> float:
    """Server-suggested wait from a 429 ('try again in 1.2s' / Retry-After)."""
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header) + 0.5
        except ValueError:
            pass
    match = re.search(r"try again in ([0-9.]+)s", str(exc))
    return float(match.group(1)) + 0.5 if match else 0.0  # 0.0 = no hint given


def _is_daily(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _DAILY_MARKERS)


def chat(messages: list, prefer: str = "groq", **kwargs):
    """chat.completions.create with retry + cross-provider failover.

    Returns the response's .choices[0].message (same shape on both SDKs).
    kwargs pass through: tools=, tool_choice=, response_format=.
    Raises the LAST provider's error only when BOTH providers are exhausted.
    """
    order = [prefer, "cerebras" if prefer == "groq" else "groq"]
    # A provider without a key can't serve — skip it instead of auth-erroring.
    order = [p for p in order if getattr(settings, f"{p}_api_key")]
    last_exc: Exception = RuntimeError("no LLM provider configured")

    for provider in order:
        client = _client(provider)
        rate_tries = 0
        format_tries = 0
        server_tries = 0
        empty_tries = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=_model(provider), messages=messages, **kwargs
                )
                message = response.choices[0].message
                # Empty completion guard: some models (seen on Cerebras
                # zai-glm-4.7) intermittently return blank content on a 200.
                # For our text/JSON callers (personas, judge, synthesis) that
                # is useless and, unretried, crashed a persona node → the whole
                # panel. A turn with tool_calls legitimately has empty content
                # (explore), so only treat blank-AND-no-tool-calls as transient.
                if not (message.content or "").strip() and not getattr(message, "tool_calls", None):
                    empty_tries += 1
                    last_exc = RuntimeError(f"{provider} returned an empty completion")
                    if empty_tries >= _MAX_EMPTY_RETRIES:
                        logger.warning("%s kept returning empty — failing over", provider)
                        break
                    time.sleep(min(2**empty_tries, 10))
                    continue
                return message
            except (groq.InternalServerError, openai.InternalServerError) as exc:
                # 5xx = the provider glitched (not our request). These are
                # transient — a single one must not kill a whole report, which
                # is how a persona node crashed the panel. Retry with backoff,
                # then fail over to the other provider.
                last_exc = exc
                server_tries += 1
                if server_tries >= _MAX_SERVER_RETRIES:
                    logger.warning("%s 5xx persisted — failing over", provider)
                    break
                time.sleep(min(2**server_tries, 15))
            except (groq.RateLimitError, openai.RateLimitError) as exc:
                last_exc = exc
                if _is_daily(exc):
                    logger.warning("%s daily quota exhausted — failing over", provider)
                    break  # next provider
                rate_tries += 1
                if rate_tries >= _MAX_RATE_RETRIES:
                    break  # persistent minute-limit → try the other provider
                # Honor a server hint; otherwise exponential backoff. A hintless
                # 429 is usually a transient queue overload ("high traffic, try
                # again soon" — seen on Cerebras): a flat 2s isn't enough to let
                # it clear, so back off 2→4→8→16→30s across the retries.
                hint = _retry_after_seconds(exc)
                time.sleep(min(hint or 2**rate_tries, 60))
            except (groq.BadRequestError, openai.BadRequestError) as exc:
                last_exc = exc
                if "tool_use_failed" not in str(exc):
                    raise  # real bad request — a bug, don't mask it
                format_tries += 1  # stochastic malformed tool-call → re-ask
                if format_tries >= _MAX_FORMAT_RETRIES:
                    break
            except (groq.APIConnectionError, openai.APIConnectionError) as exc:
                last_exc = exc
                break  # network/DNS blip on this provider → try the other
    raise last_exc
