# app/agent/llm_pool.py — one chat() over a multi-provider failover chain.
#
# WHY: no single free tier can carry the agent's bursty, many-call workload, and
# any one model/provider can be down, throttled, or (for a given account) gated.
# Every provider speaks the OpenAI chat.completions dialect, so one call-site can
# prefer one and fail over down the chain:
#   - per-MINUTE 429  → sleep the server's Retry-After, same provider
#   - per-DAY 429     → sleeping won't help TODAY → switch provider immediately
#     (and trip the circuit breaker so we stop re-probing it this run)
#   - 5xx / blank completion → transient → retry then fail over
#   - tool_use_failed 400 → stochastic malformed tool-call syntax → re-ask
#
# THE CHAIN (see _PROVIDERS): the finalized NVIDIA quality models first
# (glm → dspro → nemo → mistral, one nvapi key), then Gemini/Groq as the deep
# fallback on a DIFFERENT key (real rate-limit insurance — the NVIDIA models
# share one account quota). Callers pass prefer=settings.pool_prefer ("glm").
#
# CALL FLOW:
#   groq_driver.explore()/synthesize() / panel._judge_as() / judge.verify_groundedness()
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

# Circuit breaker cooldowns. Once a provider gives up on a call (daily cap, or
# persistent rate/5xx/empty), we stop asking it for a while — otherwise the
# explore loop re-probes a dead provider on EVERY step (measured: 8 wasted 429s
# to Groq + 8 to Cerebras in one report). A daily cap won't clear soon → long
# cooldown; a transient throttle → short.
_DAILY_COOLDOWN = 900.0  # 15 min — daily quota won't reset before then
_TRANSIENT_COOLDOWN = 60.0  # 1 min — throttle/5xx/empty; re-probe soon

# Daily-exhaustion markers in provider 429 messages: retrying is pointless
# until tomorrow, so switch providers instead of sleeping.
_DAILY_MARKERS = ("per day", "tokens per day", "tpd", "rpd", "daily")


# Failover chain, in preference order after the caller's `prefer`. The finalized
# NVIDIA quality chain leads (all one nvapi- key, integrate.api.nvidia.com);
# Gemini/Groq are the DEEP fallback (a DIFFERENT key = real rate-limit insurance,
# since the four NVIDIA models share one account quota).
#   glm → dspro → nemo → mistral  (NVIDIA)  then  gemini → gemini2 → groq
_PROVIDERS = ("glm", "dspro", "nemo", "mistral", "gemini", "gemini2", "groq")
_NVIDIA_PROVIDERS = ("glm", "dspro", "nemo", "mistral")
_GEMINI_PROVIDERS = ("gemini", "gemini2")  # need history-flattening + model override
# DeepSeek-V4-Pro fails tool-calling (verified) — skip it when tools are wanted
# (the explore loop), or it errors instead of serving. Fine for persona/synthesis.
_NO_TOOLS = ("dspro",)
_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"

# Google exposes Gemini behind an OpenAI-compatible surface, so it drops into
# this pool with the same client + kwargs (tools=, response_format=) as the rest.
_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _provider_key(provider: str) -> str:
    """The API key a provider authenticates with (one source of truth, used by
    both _client and the key-presence filter in chat)."""
    if provider == "groq":
        return settings.groq_api_key
    if provider == "gemini":
        return settings.gemini_api_key  # account 1
    if provider == "gemini2":
        return settings.gemini_secondacc_api_key  # account 2
    if provider in _NVIDIA_PROVIDERS:
        return settings.nvidia_api_key  # one key for all four NVIDIA models
    return ""


def _client(provider: str):
    """OpenAI-compatible client for a provider. All speak the same
    .chat.completions.create surface, which is what makes this pool possible."""
    if provider == "groq":
        return groq.Groq(api_key=settings.groq_api_key)
    if provider in _GEMINI_PROVIDERS:
        return openai.OpenAI(base_url=_GEMINI_OPENAI_BASE, api_key=_provider_key(provider))
    if provider in _NVIDIA_PROVIDERS:
        return openai.OpenAI(base_url=_NVIDIA_BASE, api_key=settings.nvidia_api_key)
    raise ValueError(f"unknown provider: {provider}")


def _gemini_safe(messages: list) -> list:
    """Flatten past tool-call turns into plain text for Gemini.

    Gemini-3-generation models (3-flash, 3.1-flash-lite — verified live) demand
    a `thought_signature` on every functionCall part in the HISTORY and 400
    without it. Our pool rebuilds history as plain OpenAI dicts (and mixes
    providers mid-conversation), so that signature cannot exist. Rewriting past
    assistant-tool_calls turns as text and tool results as user text removes
    every functionCall part from history — nothing left to demand a signature —
    while the model can still emit NEW tool calls normally."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in m["tool_calls"]
            )
            text = ((m.get("content") or "") + f"\n[I called tools: {calls}]").strip()
            out.append({"role": "assistant", "content": text})
        elif m.get("role") == "tool":
            out.append({"role": "user", "content": f"[tool result]: {m.get('content') or ''}"})
        else:
            out.append(m)
    return out


_NVIDIA_MODEL_ATTR = {
    "glm": "nvidia_glm_model",
    "dspro": "nvidia_dspro_model",
    "nemo": "nvidia_nemo_model",
    "mistral": "nvidia_mistral_model",
}


def _model(provider: str) -> str:
    if provider == "groq":
        return settings.groq_model
    if provider in _GEMINI_PROVIDERS:
        return settings.gemini_pool_model
    if provider in _NVIDIA_PROVIDERS:
        return getattr(settings, _NVIDIA_MODEL_ATTR[provider])
    raise ValueError(f"unknown provider: {provider}")


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


# --- request accounting -----------------------------------------------------
# Every HTTP call consumes free-tier quota. Without this, "why did my quota
# vanish?" is unanswerable. We tally each call by provider:model and outcome so
# usage is a number, not a mystery. Total calls to a model = sum of its counts.
_usage: dict = {}


def record(label: str, outcome: str = "ok") -> None:
    """Tally one call against `label` (e.g. 'gemini:gemini-2.5-flash'). Public so
    callers OUTSIDE the pool — the native Gemini synthesis — count too, giving a
    complete usage picture."""
    row = _usage.setdefault(
        label,
        {"ok": 0, "rate_429": 0, "server_5xx": 0, "empty": 0, "bad_request": 0, "conn": 0},
    )
    row[outcome] = row.get(outcome, 0) + 1


def _tally(provider: str, model_name: str, outcome: str) -> None:
    record(f"{provider}:{model_name}", outcome)


def get_usage() -> dict:
    """Snapshot of requests sent this process: {'provider:model': {outcome: n}}."""
    return {k: dict(v) for k, v in _usage.items()}


def reset_usage() -> None:
    """Zero the counters AND the circuit breaker — call at the start of a run to
    measure just that run and re-probe every provider fresh."""
    _usage.clear()
    _exhausted.clear()


# --- circuit breaker --------------------------------------------------------
# provider -> unix timestamp until which it's considered down. Set when a
# provider gives up on a call; checked when building the failover order so we
# stop re-probing a known-dead provider on every subsequent call.
_exhausted: dict = {}


def _trip(provider: str, cooldown: float) -> None:
    _exhausted[provider] = time.time() + cooldown


def _live(order: list) -> list:
    """Drop providers still on cooldown. If that empties the list (everything is
    tripped), return the original order so we still ATTEMPT rather than silently
    refuse — the cooldown may be stale."""
    now = time.time()
    live = [p for p in order if _exhausted.get(p, 0.0) <= now]
    return live or order


def chat(messages: list, prefer: str = "groq", gemini_model: str | None = None, **kwargs):
    """chat.completions.create with retry + cross-provider failover.

    Returns the response's .choices[0].message (same shape on both SDKs).
    kwargs pass through: tools=, tool_choice=, response_format=.
    gemini_model overrides settings.gemini_pool_model for the Gemini provider
    only (per-call model experiments; other providers keep their own model).
    Raises the LAST provider's error only when ALL providers are exhausted.
    """
    # Caller's preferred provider first, then the rest of the chain (Gemini last
    # as the safety net). De-duped, and any provider without a key is skipped
    # rather than auth-erroring.
    order = [prefer] + [p for p in _PROVIDERS if p != prefer]
    seen: set = set()
    order = [
        p for p in order
        if _provider_key(p) and not (p in seen or seen.add(p))
    ]
    # Tool-gate: a call passing tools= needs a tool-capable model. Drop providers
    # known to fail tool-calling (DeepSeek-V4-Pro) so the explore loop doesn't
    # fall onto one and error instead of failing over.
    if kwargs.get("tools"):
        order = [p for p in order if p not in _NO_TOOLS]
    order = _live(order)  # skip providers on circuit-breaker cooldown
    last_exc: Exception = RuntimeError("no LLM provider configured")

    for provider in order:
        client = _client(provider)
        # Per-provider model + Gemini history flattening (thought_signature fix).
        if provider in _GEMINI_PROVIDERS:
            model_name = gemini_model or _model(provider)
            body = _gemini_safe(messages)
        else:
            model_name = _model(provider)
            body = messages
        rate_tries = 0
        format_tries = 0
        server_tries = 0
        empty_tries = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=body, **kwargs
                )
                message = response.choices[0].message
                # Empty completion guard: some models (seen on Cerebras
                # zai-glm-4.7) intermittently return blank content on a 200.
                # For our text/JSON callers (personas, judge, synthesis) that
                # is useless and, unretried, crashed a persona node → the whole
                # panel. A turn with tool_calls legitimately has empty content
                # (explore), so only treat blank-AND-no-tool-calls as transient.
                if not (message.content or "").strip() and not getattr(message, "tool_calls", None):
                    _tally(provider, model_name, "empty")
                    empty_tries += 1
                    last_exc = RuntimeError(f"{provider} returned an empty completion")
                    if empty_tries >= _MAX_EMPTY_RETRIES:
                        logger.warning("%s kept returning empty — failing over", provider)
                        _trip(provider, _TRANSIENT_COOLDOWN)
                        break
                    time.sleep(min(2**empty_tries, 10))
                    continue
                _tally(provider, model_name, "ok")
                return message
            except (groq.InternalServerError, openai.InternalServerError) as exc:
                # 5xx = the provider glitched (not our request). These are
                # transient — a single one must not kill a whole report, which
                # is how a persona node crashed the panel. Retry with backoff,
                # then fail over to the other provider.
                _tally(provider, model_name, "server_5xx")
                last_exc = exc
                server_tries += 1
                if server_tries >= _MAX_SERVER_RETRIES:
                    logger.warning("%s 5xx persisted — failing over", provider)
                    _trip(provider, _TRANSIENT_COOLDOWN)
                    break
                time.sleep(min(2**server_tries, 15))
            except (groq.RateLimitError, openai.RateLimitError) as exc:
                _tally(provider, model_name, "rate_429")
                last_exc = exc
                if _is_daily(exc):
                    logger.warning("%s daily quota exhausted — failing over", provider)
                    _trip(provider, _DAILY_COOLDOWN)  # won't clear soon → stop asking
                    break  # next provider
                rate_tries += 1
                hint = _retry_after_seconds(exc)
                # A hintless 429 is a transient overload ("high traffic, try
                # again soon" — seen on Cerebras). If another provider is still
                # available, don't grind through 6 backoffs (~60s) — a couple of
                # quick tries, then fail over to it. Only when this is the LAST
                # provider do we wait out the full retry budget.
                is_last = provider == order[-1]
                limit = _MAX_RATE_RETRIES if (hint or is_last) else 2
                if rate_tries >= limit:
                    _trip(provider, _TRANSIENT_COOLDOWN)  # throttled → deprioritize briefly
                    break  # give up on this provider → next one (or raise)
                time.sleep(min(hint or 2**rate_tries, 60))
            except (groq.BadRequestError, openai.BadRequestError) as exc:
                _tally(provider, model_name, "bad_request")  # count ALL 400s (incl re-raised) for visibility
                last_exc = exc
                if "tool_use_failed" not in str(exc):
                    raise  # real bad request (e.g. Gemini-3 thought_signature) — surface it
                format_tries += 1  # stochastic malformed tool-call → re-ask
                if format_tries >= _MAX_FORMAT_RETRIES:
                    _trip(provider, _TRANSIENT_COOLDOWN)
                    break
            except (groq.APIConnectionError, openai.APIConnectionError) as exc:
                _tally(provider, model_name, "conn")
                last_exc = exc
                _trip(provider, _TRANSIENT_COOLDOWN)
                break  # network/DNS blip on this provider → try the other
    raise last_exc
