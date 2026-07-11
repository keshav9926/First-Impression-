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

## Design decisions (build vs buy)

- **Chunking is hand-written** ([app/ingestion/chunker.py](app/ingestion/chunker.py)) as a
  deliberate learning artifact — ~60 testable lines whose strategy we fully own.
  LangChain's splitters (`RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`)
  are a drop-in alternative and the reasonable "buy" choice in a production rush;
  speed is identical (chunking is a negligible share of ingest time) and quality is
  comparable on clean docs text.
- **HTML extraction is bought** (`trafilatura`) — the inverse call: stripping
  boilerplate from arbitrary HTML is genuinely hard, so a battle-tested library wins.
- Same principle later: LangGraph is used for multi-agent orchestration (hard
  machinery), while prompts, schemas, and guardrail logic stay hand-written.
