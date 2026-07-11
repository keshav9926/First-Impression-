# app/rag/qa.py — the "G" in RAG: generate an answer grounded in retrieved chunks.
#
# CALL FLOW:
#   main.py: ask() → answer(question, hits)
#     where `hits` came from store.search() — the top-k relevant chunks.
#   answer() numbers the chunks, builds the prompt, then dispatches to ONE
#   provider based on settings.llm_provider:
#       "gemini"    → _ask_gemini()    (free tier — current default)
#       "anthropic" → _ask_claude()    (paid — switch back via .env when funded)
#
# This file is the ONLY place that talks to an LLM. That isolation is why
# swapping providers was a small, local change — the crawler, chunker,
# embeddings, and store never knew it happened.
#
# The system prompt applies hard rule #2 (grounded output only) at the prompt
# level: answer ONLY from the excerpts, cite every claim, admit when the
# answer isn't there. Phase 5 adds automated checks that verify compliance.

import anthropic
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


def answer(question: str, hits: list[dict]) -> str:
    """Ask the configured LLM the question, constrained to the retrieved chunks.

    Called by: main.py ask(), after retrieval.
    Calls: _build_user_message(), then _ask_gemini() or _ask_claude()
    depending on settings.llm_provider.
    """
    user_message = _build_user_message(question, hits)
    if settings.llm_provider == "anthropic":
        return _ask_claude(user_message)
    return _ask_gemini(user_message)
