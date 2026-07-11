# app/config.py — typed application configuration.
# Loads settings from environment variables (and a local .env file in dev).
# Every later phase adds its settings here (API keys, model names, etc.),
# so configuration lives in exactly one place and fails loudly at startup
# if something required is missing — instead of a None value surfacing
# mid-request.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Reads variables from a `.env` file if present; real environment
    # variables always take precedence over the file.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "First Impression"
    environment: str = "development"  # "development" | "production"

    # --- LLM (Anthropic) ---
    # Default "" instead of required: lets /health and tests run without keys.
    # Endpoints that need a key check for it and return a clear 503 if absent.
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-8"

    # --- Embeddings (Voyage AI) ---
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3.5"

    # --- Vector store (Chroma) ---
    chroma_dir: str = "./chroma_data"  # where Chroma persists vectors on disk
    collection_name: str = "company_docs"

    # --- Crawler politeness (enforces the "public data only" rule) ---
    # Honest User-Agent so site owners can identify and block us if they wish.
    crawler_user_agent: str = "FirstImpressionBot/0.1 (learning project; respects robots.txt)"
    request_delay_seconds: float = 1.0  # pause between requests — never hammer a site
    max_pages_hard_limit: int = 50


# One shared instance, imported by the rest of the app.
settings = Settings()
