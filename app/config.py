# app/config.py — typed application configuration.
#
# WHERE THIS IS USED: every module imports the shared `settings` object below —
#   main.py      → app_name, environment (and the two key checks)
#   robots.py    → crawler_user_agent
#   fetcher.py   → crawler_user_agent, request_delay_seconds
#   embeddings.py→ voyage_api_key, embedding_model
#   store.py     → chroma_dir, collection_name
#   qa.py        → anthropic_api_key, claude_model
#
# HOW IT WORKS: Settings() runs ONCE, when this module is first imported
# (at server startup) — not per request. Pydantic reads environment
# variables (and a local .env file in dev; matching is case-insensitive,
# so ANTHROPIC_API_KEY in .env fills anthropic_api_key here), validates
# them, and freezes the result. Bad config fails the BOOT, not a user
# request three hours later.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Reads variables from a `.env` file if present; real environment
    # variables always take precedence over the file.
    # extra="ignore": .env may hold keys this app doesn't map (e.g. an
    # experimental provider key, or a malformed name) — ignore them instead of
    # failing to boot. Only fields declared below are read.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "First Impression"
    environment: str = "development"  # "development" | "production"

    # --- LLM provider switch — used by rag/qa.py for /ask ---
    # "groq"      : Llama via Groq API (generous free-tier RPM — default for /ask)
    # "gemini"    : Google Gemini (free tier, but ~20 req/day — use sparingly)
    # "anthropic" : Claude (paid)
    # The /report agent ALWAYS uses Groq for exploration (Phase A) and Gemini
    # for the final synthesis (Phase B), regardless of this setting.
    llm_provider: str = "groq"

    # --- LLM (Anthropic) — used by qa.py when llm_provider="anthropic" ---
    # Default "" instead of required: lets /health and tests run without keys.
    # Endpoints that need a key check for it and return a clear 503 if absent
    # (see main.py _require_keys).
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-8"

    # --- LLM (Google Gemini) — used by qa.py when llm_provider="gemini" ---
    # Free-tier key from https://aistudio.google.com (no card required).
    gemini_api_key: str = ""
    # Second Google account's key. The primary key's project has no free-tier
    # access to the 2.0 models (observed limit: 0); a standard AI Studio key
    # (AIza…) does — 2.0-flash at ~1,500 requests/day free. Used for the Gemini
    # pool provider so the high-volume agent traffic lands on the big free quota.
    gemini_secondacc_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"  # single-call workloads (/ask)
    # The Phase 3 agent makes MANY calls per report. On THIS project's Gemini
    # free tier that's a problem (~20 requests/DAY on 2.5-flash; the flash-lite
    # models are unusable here), so the agent can run on Groq instead — see
    # agent_provider below. This is the model used when agent_provider="gemini".
    gemini_agent_model: str = "gemini-3-flash-preview"  # synthesis: newest flash

    # --- LLM (Groq) — used by the report agent when agent_provider="groq" ---
    # Free-tier key from https://console.groq.com (no card). Groq's free limits
    # are generous (tens of requests/minute), which suits the agent's burst of
    # calls far better than Gemini's tiny daily free quota.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"  # tool-calling capable

    # --- LLM (Gemini) as a pool provider (agent/llm_pool.py) ---
    # Google's OpenAI-compatible endpoint lets Gemini serve the pool for
    # explore/personas/judge. 3.1-flash-lite: the one Gemini model live-verified
    # on BOTH keys (2.0-flash: free tier limit 0 on both; 2.5-flash: 404 "not
    # available to new users" on key2, daily-capped on key1). Gemini-3-gen
    # models demand a thought_signature on functionCall parts in multi-turn
    # history — llm_pool._gemini_safe flattens past tool turns to text, so the
    # loop works anyway. Uses gemini_secondacc_api_key (fresh 1,000 RPD quota).
    gemini_pool_model: str = "gemini-3.1-flash-lite"

    # --- LLM (NVIDIA NIM) — the FINALIZED agent chain (agent/llm_pool.py) ---
    # One nvapi- key serves all four via integrate.api.nvidia.com. Chosen by a
    # full-report benchmark on real sites + Artificial Analysis cross-check:
    #   glm      GLM-5.2            — top all-round (AA intel 51, sharpest grounding)
    #   dspro    DeepSeek-V4-Pro    — richest persona/synthesis; CANNOT explore (no tools)
    #   nemo     Nemotron-3-Ultra   — agentic-built, fast (186 t/s), tool-calling
    #   mistral  Mistral-Medium-3.5 — most complete extraction
    # Fallback order glm→dspro→nemo→mistral; the pool skips dspro when tools are
    # requested (explore), since its tool-calling fails. Same key = shared account
    # quota, so Gemini/Groq remain the deep fallback for real rate-limit insurance.
    nvidia_api_key: str = ""
    nvidia_glm_model: str = "z-ai/glm-5.2"
    nvidia_dspro_model: str = "deepseek-ai/deepseek-v4-pro"
    nvidia_nemo_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_mistral_model: str = "mistralai/mistral-medium-3.5-128b"

    # Which provider the agent workloads (explore, personas, judge, synthesis)
    # try FIRST. "glm" = the finalized quality-first NVIDIA chain above.
    pool_prefer: str = "glm"

    # --- Which LLM runs the /report agent: "gemini" or "groq" ---
    # Separate from llm_provider (which drives /ask) because the agent is a
    # high-volume workload with different rate-limit economics. The tool-calling
    # dialects differ per provider, so each has its own driver (agent/*_driver
    # style) behind this switch.
    agent_provider: str = "gemini"

    # Hard cap on the agent's explore loop, shared by BOTH drivers (react.py
    # and groq_driver.py) so the two can never drift apart. 5 is enough for
    # list_pages + read key pages + 1-2 targeted searches, and keeps the
    # resent-history token burst inside Groq's ~12K tokens/minute free tier.
    agent_max_steps: int = 5

    # Phase 5: one extra Gemini call per report that fact-checks every claim
    # against its cited page's stored text; unsupported claims are dropped.
    # Fail-open on judge errors. Disable to save the call (or when Gemini
    # daily quota is exhausted).
    groundedness_judge: bool = True

    # --- Embeddings (Voyage AI) — used by embeddings.py ---
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3.5"

    # --- Retrieval quality (Phase 2) — used by rerank.py + main.py ---
    rerank_model: str = "rerank-2.5-lite"
    # Candidates scoring below this after re-ranking are treated as
    # irrelevant; if NONE clear the bar, /ask answers "no relevant content"
    # without calling the LLM. Tune with evals/run_retrieval_eval.py.
    # 0.45 chosen from the 2026-07-12 eval run on vortexify.ai:
    # lowest answerable top-score 0.480, highest unanswerable 0.424 → midpoint.
    # NOTE: rerank-2.5-lite is 3rd gen. rerank-2.5 (full) is the real upgrade
    # when scores need to be sharper — NOT rerank-2 (that is 2nd gen, older).
    min_relevance: float = 0.45

    # --- Vector store (Chroma) — used by store.py ---
    chroma_dir: str = "./chroma_data"  # where Chroma persists vectors on disk
    collection_name: str = "company_docs"

    # --- Crawler politeness — used by robots.py + fetcher.py (rule #1) ---
    # Honest User-Agent so site owners can identify and block us if they wish.
    crawler_user_agent: str = "FirstImpressionBot/0.1 (learning project; respects robots.txt)"
    request_delay_seconds: float = 1.0  # pause between requests — never hammer a site
    max_pages_hard_limit: int = 50


# One shared instance, imported by the rest of the app.
settings = Settings()
