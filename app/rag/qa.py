# app/rag/qa.py — the "G" in RAG: generate an answer grounded in retrieved chunks.
# The retrieved chunks are numbered and given to Claude with strict rules:
# answer ONLY from them, cite every claim, admit when the answer isn't there.
# This is hard rule #2 (grounded output only) applied at the prompt level;
# Phase 5 adds automated checks that verify it.

import anthropic

from app.config import settings

SYSTEM_PROMPT = """\
You answer questions about a company's product using ONLY the numbered source \
excerpts provided in the user message.

Rules:
- Base every statement on the excerpts. Cite the excerpt number for each claim, e.g. [1] or [2][3].
- If the excerpts do not contain the answer, say so plainly. Never use outside \
knowledge and never guess.
- Describe, don't judge: observational and neutral in tone."""


def answer(question: str, hits: list[dict]) -> str:
    """Ask Claude the question, constrained to the retrieved chunks."""
    sources_block = "\n\n".join(
        f"[{i + 1}] (from {hit['url']})\n{hit['text']}" for i, hit in enumerate(hits)
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Source excerpts:\n\n{sources_block}\n\nQuestion: {question}",
            }
        ],
    )
    # The response is a list of content blocks; collect the text ones.
    return "".join(block.text for block in response.content if block.type == "text")
