# First Impression

AI system that analyzes a startup's **public** product experience and produces a
grounded, citation-backed report on the new-user journey. Full spec: [PROJECT.md](PROJECT.md).

## Run locally

```sh
# 1. One-time setup: create venv + install dependencies
uv sync

# 2. Configure
copy .env.example .env

# 3. Start the dev server (auto-reloads on code changes)
uv run uvicorn app.main:app --reload
```

Then open:
- http://127.0.0.1:8000/health — liveness check
- http://127.0.0.1:8000/docs — interactive API docs

## Run tests

```sh
uv run pytest
```

## Run in Docker

```sh
docker compose up --build
```
