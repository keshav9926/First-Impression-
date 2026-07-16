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
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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
    gemini_model: str = "gemini-2.5-flash"  # single-call workloads (/ask)
    # The Phase 3 agent makes MANY calls per report. On THIS project's Gemini
    # free tier that's a problem (~20 requests/DAY on 2.5-flash; the flash-lite
    # models are unusable here), so the agent can run on Groq instead — see
    # agent_provider below. This is the model used when agent_provider="gemini".
    gemini_agent_model: str = "gemini-2.5-flash"

    # --- LLM (Groq) — used by the report agent when agent_provider="groq" ---
    # Free-tier key from https://console.groq.com (no card). Groq's free limits
    # are generous (tens of requests/minute), which suits the agent's burst of
    # calls far better than Gemini's tiny daily free quota.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"  # tool-calling capable

    # --- LLM (Cerebras) — second free provider in agent/llm_pool.py ---
    # Free key from https://cloud.cerebras.ai. Groq's 100K tokens/DAY can't fit
    # 20 reports; the pool prefers Groq for explore, Cerebras for persona/judge
    # JSON verdicts, and fails over across the two on daily-quota 429s.
    cerebras_api_key: str = ""
    cerebras_model: str = "zai-glm-4.7"  # tool-calling + JSON mode verified

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
