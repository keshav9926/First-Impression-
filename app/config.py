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

    # --- LLM provider switch — used by qa.py ---
    # "gemini" (free tier) or "anthropic" (paid). The rest of the pipeline
    # doesn't know or care which LLM answers — only qa.py reads this.
    llm_provider: str = "gemini"

    # --- LLM (Anthropic) — used by qa.py when llm_provider="anthropic" ---
    # Default "" instead of required: lets /health and tests run without keys.
    # Endpoints that need a key check for it and return a clear 503 if absent
    # (see main.py _require_keys).
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-8"

    # --- LLM (Google Gemini) — used by qa.py when llm_provider="gemini" ---
    # Free-tier key from https://aistudio.google.com (no card required).
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # --- Embeddings (Voyage AI) — used by embeddings.py ---
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3.5"

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
