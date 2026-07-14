# app/rag/qa.py — the "G" in RAG: generate an answer grounded in retrieved chunks.
#
# CALL FLOW:
#   main.py: ask() → answer(question, hits)
#     where `hits` came from store.search() — the top-k relevant chunks.
#   answer() numbers the chunks, builds the prompt, then dispatches to ONE
#   provider based on settings.llm_provider:
#       "groq"      → _ask_groq()      (free tier, high rate-limits — default for /ask)
#       "gemini"    → _ask_gemini()    (free tier — daily quota burns fast)
#       "anthropic" → _ask_claude()    (paid — switch via .env when funded)
#
# This file is the ONLY place that talks to an LLM for Q&A. That isolation is
# why swapping providers was a small, local change — the crawler, chunker,
# embeddings, and store never knew it happened.
#
# The system prompt applies hard rule #2 (grounded output only) at the prompt
# level: answer ONLY from the excerpts, cite every claim, admit when the
# answer isn't there. Phase 5 adds automated checks that verify compliance.

import anthropic
import groq as groq_sdk
from google import genai
from google.genai import types as genai_types

from app.config import settings

SYSTEM_PROMPT = """\
You answer questions about a company's product using ONLY the numbered source \
excerpts provided in the user message.

Rules:
- Base every statement on the excerpts. Cite the excerpt number for each claim, e.g. [1] or [2][3].
- If the excerpts do not contain the answer, say so plainly. Never use outside \
knowledge and never guess.
- Describe, don't judge: observational and neutral in tone."""


def _build_user_message(question: str, hits: list[dict]) -> str:
    """Format the retrieved chunks as a numbered source list + the question.

    Called by: answer().
    The [1], [2] numbers here are what the LLM cites — and they match the
    `sources` array main.ask() returns, so every claim is checkable.
    """
    sources_block = "\n\n".join(
        f"[{i + 1}] (from {hit['url']})\n{hit['text']}" for i, hit in enumerate(hits)
    )
    return f"Source excerpts:\n\n{sources_block}\n\nQuestion: {question}"


def _ask_gemini(user_message: str) -> str:
    """Send the prompt to Google Gemini and return its text answer.

    Called by: answer() when settings.llm_provider == "gemini".
    Calls: the Gemini API (network call; free-tier quota).
    """
    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_message,
        config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text or ""


def _ask_claude(user_message: str) -> str:
    """Send the prompt to Anthropic's Claude and return its text answer.

    Called by: answer() when settings.llm_provider == "anthropic".
    Calls: the Anthropic API (network call; costs credits).
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    # The response is a list of content blocks; collect the text ones.
    return "".join(block.text for block in response.content if block.type == "text")


def _ask_groq(user_message: str) -> str:
    """Send the prompt to Groq (Llama) and return its text answer.

    Called by: answer() when settings.llm_provider == "groq".
    Uses the same groq_model configured for the report agent; generous
    free-tier rate limits (tens of RPM vs Gemini's ~20 req/day) make
    this the better choice for interactive /ask calls.
    """
    client = groq_sdk.Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""


def answer(question: str, hits: list[dict]) -> str:
    """Ask the configured LLM the question, constrained to the retrieved chunks.

    Called by: main.py ask(), after retrieval.
    Calls: _build_user_message(), then whichever of _ask_groq / _ask_gemini /
    _ask_claude is selected by settings.llm_provider.
    Default (llm_provider="groq"): Groq API — high rate-limits, great for
    interactive calls. Switch to "gemini" or "anthropic" via .env.
    """
    user_message = _build_user_message(question, hits)
    if settings.llm_provider == "anthropic":
        return _ask_claude(user_message)
    if settings.llm_provider == "gemini":
        return _ask_gemini(user_message)
    # Default: groq
    return _ask_groq(user_message)
