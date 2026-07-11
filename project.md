# PROJECT.md — "First Impression"

## What this is
An AI system that analyzes a startup's **public-facing** product experience and produces an accurate, respectful report on the new-user journey. Given a company's public site + docs URL, it:
1. Ingests public docs/marketing/pricing (RAG).
2. Runs a multi-agent analysis: a researcher, a "new user" simulator, a feature evaluator, and a skeptic that fact-checks the others.
3. Produces a structured report: what the product does, the first-run experience, friction points, and standout features — all cited to the source page.
4. Runs eval + guardrail checks so it never states anything not grounded in the retrieved source.

## Hard rules (non-negotiable, enforced in code)
- **Public data only.** No authenticated areas, no scraping behind logins, no probing/scanning of APIs or infrastructure. Respect robots.txt and rate limits.
- **Grounded output only.** Every claim in the report must cite a retrieved source chunk. If it can't be cited, it doesn't ship.
- **Observational tone, not judgmental.** Report describes the user experience ("a new user might hesitate here"); it does not grade or attack.

## Tech stack (the study topics, mapped to components)
- **LLM fundamentals / prompt engineering** → agent prompt design
- **RAG** (chunking, embeddings, vector DB, hybrid search, re-ranking) → doc ingestion + retrieval
- **Agentic AI** (ReAct, planning, tool use, multi-agent orchestration via LangGraph) → the analysis crew
- **MCP** → a custom MCP server exposing the analysis as a tool
- **Structured outputs** → report schema (Pydantic)
- **Evals** (LLM-as-judge, RAGAS) → groundedness / hallucination checks
- **Observability** (Langfuse) → trace every agent step
- **Guardrails / prompt-injection defense** → public-data enforcement + injection filtering on scraped content
- **FastAPI + streaming** → backend + live analysis dashboard
- **Docker** → deployment

## Build phases (build + review ONE phase at a time)
- **Phase 0** — Repo skeleton, FastAPI hello-world, Docker, env config, dependency setup.
- **Phase 1** — Public-content ingestion (fetch site/docs, respect robots.txt), chunking, embeddings, vector store. Plain RAG Q&A over one company's docs.
- **Phase 2** — Retrieval quality: hybrid search + re-ranking. First eval: retrieval precision.
- **Phase 3** — Single analysis agent (ReAct, tool use) producing a structured report with citations.
- **Phase 4** — Multi-agent crew (researcher / user-sim / evaluator / skeptic) via LangGraph.
- **Phase 5** — Evals + guardrails: groundedness check (LLM-as-judge + RAGAS), public-data guardrail, prompt-injection filter on scraped text.
- **Phase 6** — FastAPI streaming endpoint + live dashboard showing each agent's step.
- **Phase 7** — Custom MCP server exposing the analyzer as a tool.
- **Phase 8** — Observability (Langfuse traces), Dockerize, deploy, write the evals dashboard page.

## Working agreement with the coding agent
- Do ONE phase at a time. Stop after each phase for review.
- Explain each file's purpose in a comment header.
- No new dependency without noting why in the response.
- After each phase, list what I should study to understand what was built.