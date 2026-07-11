# app/rag/qa.py — the "G" in RAG: generate an answer grounded in retrieved chunks.
#
# CALL FLOW:
#   main.py: ask() → answer(question, hits)
#     where `hits` came from store.search() — the top-k relevant chunks.
#   answer() numbers the chunks, sends them + the question to Claude,
#   and returns Claude's text back to the endpoint.
#
# The system prompt applies hard rule #2 (grounded output only) at the prompt
# level: answer ONLY from the excerpts, cite every claim, admit when the
# answer isn't there. Phase 5 adds automated checks that verify compliance.

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
    """Ask Claude the question, constrained to the retrieved chunks.

    Called by: main.py ask(), after retrieval.
    Calls: the Anthropic API (network call, costs tokens).

    Steps:
      1. Format the hits as a numbered source list — the numbers are what
         Claude cites as [1], [2], and they match the `sources` array the
         endpoint returns, so a reader can check every claim.
      2. One messages.create() call: system prompt = the rules,
         user message = sources + question.
      3. The response arrives as a list of content blocks; join the text ones.
    """
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
