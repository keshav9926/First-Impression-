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
    nvidia_dsflash_model: str = "deepseek-ai/deepseek-v4-flash"
    nvidia_nemo_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_mistral_model: str = "mistralai/mistral-medium-3.5-128b"

    # --- Vision: read product screenshots the text extractor is blind to ---
    # 2026-07-19 vision bake-off on real product dashboards (7 VLMs): omni-30b
    # won on BOTH speed (2.3s/img) and accuracy (read exact shipment IDs,
    # statuses, chart contents) — beating llama-3.2-90b-vision (NVIDIA 504s),
    # gemma-3n, and the nemotron VL 8b/12b. Captions ride into chunk metadata so
    # the agent/personas/judge SEE what a screenshot shows, killing false
    # "no product screenshots" / "can't tell what the UI does" findings.
    vision_enabled: bool = True
    vision_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    # Quality-first (2026-07-20): the pipeline runs on-demand per company, so
    # thoroughness beats cost. Caps raised so EVERY product screenshot on a page
    # gets read, not a sample. Lower these again if a batch/free-tier run needs
    # a budget.
    vision_max_images_per_page: int = 40
    vision_max_images_total: int = 200  # ~2s each; effectively "read them all"
    # Vision FAILOVER chain (image→text), mirroring the LLM pool. The 503
    # "Worker local total request limit reached (16/16)" is a PER-WORKER
    # concurrency cap — each VLM is a separate NVIDIA deployment, so when omni
    # is saturated the next model's worker usually isn't. Tried in order per
    # image (with short backoff-retries first); first successful caption wins.
    # omni-30b leads (bake-off winner); the two llama VLMs are the safety net.
    vision_models: list[str] = [
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "meta/llama-3.2-90b-vision-instruct",
        "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    ]
    vision_retries_per_model: int = 3  # backoff-retries on transient 503/504/429

    # Two report pipelines (2026-07-19 bake-off). "→" = failover: try the next
    # model if one errors OR produces no valid report.
    #   deep   (no time budget): glm-5.2 → v4-pro → v4-flash → nemotron
    #   normal (fast, default):           v4-pro → v4-flash → nemotron
    # GLM leads "deep" as the accuracy anchor; nemotron tails both as the fast,
    # rock-solid 3/3 safety net. (mistral dropped — too slow; qwen/minimax/kimi
    # unavailable or NVIDIA-degraded.)
    pipeline_mode: str = "normal"  # "normal" | "deep"

    # Which provider the agent workloads (explore, personas, judge, synthesis)
    # try FIRST, then the rest of the chain as failover. "dspro" = DeepSeek-V4-Pro
    # (2026-07-18 bake-off: cleanest full-pipeline run — explored fine with tools,
    # rich synthesis, ~8 min vs GLM's ~30). glm/nemo/mistral remain as failover.
    pool_prefer: str = "dspro"

    # --- LEGACY: which LLM runs the /report agent ---
    # No longer routes anything (2026-07-18): the report pipeline is NVIDIA-only
    # and always runs on the pool (agent/groq_driver.py over settings.pool_prefer).
    # Kept only so old .env files with AGENT_PROVIDER set still boot. Remove once
    # a non-NVIDIA report path is reintroduced.
    agent_provider: str = "nvidia"

    # Hard cap on the agent's explore loop. Raised 5 → 7 (2026-07-18): the old
    # cap of 5 existed only to keep the resent-history token burst inside Groq's
    # ~12K tokens/minute free tier — and Groq is no longer in the pool (NVIDIA-
    # only). With that constraint gone, the only cost of more steps is GLM
    # latency. 7 covers list_pages + read 3-5 key pages + 2-3 targeted searches,
    # so multi-page sites are actually explored rather than the agent running out
    # of steps and reporting "not found" for pages it never reached. RE-BENCHMARK
    # latency if you push this higher; revert to 5 if GLM per-step time hurts.
    # Quality-first (2026-07-20): raised 7 → 40 so the agent can actually READ
    # every page of a larger site and run many targeted searches/questions,
    # rather than running out of steps and reporting "not found" for pages it
    # never reached. The only cost is latency (on-demand runs, time is fine).
    agent_max_steps: int = 40

    # Phase 5: one extra Gemini call per report that fact-checks every claim
    # against its cited page's stored text; unsupported claims are dropped.
    # Fail-open on judge errors. Disable to save the call (or when Gemini
    # daily quota is exhausted).
    groundedness_judge: bool = True

    # --- Observability (Phase 8, Langfuse) — used by app/observability.py ---
    # When BOTH keys are set, each report run is traced to Langfuse: one span
    # tree per report with every LLM call as a nested generation (model, tokens,
    # latency). Absent keys → tracing is a hard no-op (like events.py), so
    # /health, tests, and CLI runs pay nothing and need no Langfuse account.
    # Free cloud project at https://cloud.langfuse.com (EU host by default;
    # set LANGFUSE_HOST=https://us.cloud.langfuse.com for the US region).
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # The Langfuse SDK/CLI convention is LANGFUSE_HOST, but the official skill's
    # env template uses LANGFUSE_BASE_URL — accept BOTH so a .env written either
    # way just works (base_url wins when set). EU cloud by default; US region is
    # https://us.cloud.langfuse.com.
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_base_url: str = ""

    # --- Retrieval backends (embeddings + rerank) ---
    # Switched to NVIDIA (2026-07-19) to kill the wall-clock bottleneck: Voyage's
    # free tier caps at 3 requests/MINUTE, so every search_content during a report
    # ate ~20s waits. NVIDIA's embed/rerank endpoints have no such throttle (and
    # the bake-off showed nemotron-3-embed matched/edged Voyage on MRR). Trade-off:
    # everything now shares ONE NVIDIA account quota. Set either back to "voyage"
    # to revert (keys retained below).
    # embeddings -> NVIDIA: removes the biggest throttle (all of ingestion + the
    # query-embed on every search_content) at no accuracy cost (bake-off MRR tie).
    embed_provider: str = "nvidia"   # "nvidia" | "voyage"
    # rerank stays VOYAGE for now: NVIDIA rerank returns strongly-negative logits
    # (relevant ~-7, junk ~-14) that sigmoid bunches near 0, so the calibrated
    # min_relevance gate would reject real content. Voyage's 0..1 keeps the gate
    # valid. Flip to "nvidia" only after picking a logit threshold from real
    # multi-company data. (Reranker reads TEXT, so NVIDIA-embed + Voyage-rerank
    # mix cleanly.)
    rerank_provider: str = "voyage"  # "nvidia" | "voyage"

    # NVIDIA retrieval models (embed via integrate.api /v1/embeddings, rerank via
    # ai.api /v1/retrieval/nvidia/reranking). One nvapi- key serves both.
    nvidia_embed_model: str = "nvidia/nemotron-3-embed-1b"     # 2048-dim
    nvidia_rerank_model: str = "nvidia/rerank-qa-mistral-4b"   # returns logits
    # NVIDIA rerank logits are strongly negative even for good matches (relevant
    # ~-7, junk ~-14). A plain sigmoid bunches them near 0, breaking the 0..1
    # gate. This SHIFTED sigmoid — 1/(1+exp(-(logit-center)/scale)) — recenters
    # so relevant lands high and junk low. center/scale are provisional (from a
    # handful of observations); re-fit from logged logits once we have more data.
    nvidia_rerank_center: float = -10.0
    nvidia_rerank_scale: float = 3.0

    # --- Embeddings (Voyage AI) — used by embeddings.py when embed_provider="voyage" ---
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3.5"

    # --- Retrieval quality (Phase 2) — used by rerank.py when rerank_provider="voyage" ---
    rerank_model: str = "rerank-2.5-lite"
    # Candidates scoring below this after re-ranking are treated as irrelevant;
    # if NONE clear the bar, /ask answers "no relevant content" without calling
    # the LLM. 0.30 = a conservative SAFETY FLOOR (2026-07-19), not a tuned
    # threshold: both eval runs put obvious junk well below 0.30 and real
    # answers >= 0.48, so the floor only blocks clear garbage and can't eat
    # real content. The ambiguous 0.40-0.50 band is handled by layer 2 — the
    # qa.py prompt instructs the LLM to say plainly when excerpts don't contain
    # the answer. Re-calibrate to a sharper value from logged top-scores after
    # running many companies (evals/run_retrieval_eval.py).
    min_relevance: float = 0.30

    # --- Vector store (Chroma) — used by store.py ---
    chroma_dir: str = "./chroma_data"  # where Chroma persists vectors on disk
    collection_name: str = "company_docs"

    # --- Crawler politeness — used by robots.py + fetcher.py (rule #1) ---
    # Honest User-Agent so site owners can identify and block us if they wish.
    crawler_user_agent: str = "FirstImpressionBot/0.1 (learning project; respects robots.txt)"
    request_delay_seconds: float = 1.0  # pause between requests — never hammer a site
    max_pages_hard_limit: int = 300

    # Quality-first (2026-07-20): always do the headless-render pass, don't wait
    # for the static crawl to look "thin". JS-heavy sites (Vortexify, Framer,
    # SPAs) only expose their real nav AND their product screenshots after
    # render — forcing it means we crawl every page and the vision captioner
    # actually sees the dashboards. Set False to revert to cheap static-first.
    force_render: bool = True


# One shared instance, imported by the rest of the app.
settings = Settings()
