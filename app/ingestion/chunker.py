# app/ingestion/chunker.py — splits page text into chunks for embedding.
# Written by hand (no LangChain) because chunking is a core RAG concept:
#
# WHY chunk at all? Two reasons:
#   1. Retrieval precision — an embedding of a whole page averages many topics
#      into one vector; a question matches a focused paragraph far better.
#   2. Context budget — we send only the top-k relevant chunks to the LLM,
#      not entire pages.
#
# Strategy: greedy paragraph packing. Keep whole paragraphs together (they are
# natural meaning boundaries), pack them into chunks up to ~max_chars, and
# carry the last paragraph of each chunk into the next one (overlap) so a
# sentence's context isn't lost when a topic straddles a boundary.

DEFAULT_MAX_CHARS = 1600  # ~400 tokens: big enough for context, small enough to stay focused
DEFAULT_OVERLAP = 1  # paragraphs carried over between consecutive chunks


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks of at most ~max_chars characters."""
    # trafilatura separates blocks with newlines; each block ≈ one paragraph.
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # A single paragraph longer than max_chars can't be packed — hard-split it.
    units: list[str] = []
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            units.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        if paragraph:
            units.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        if current and current_len + len(unit) > max_chars:
            # Current chunk is full — emit it.
            chunks.append("\n".join(current))
            # Overlap: start the next chunk with the tail of this one,
            # but only if that still leaves room for the new paragraph.
            tail = current[-overlap:] if overlap > 0 else []
            tail_len = sum(len(t) for t in tail)
            if tail_len + len(unit) <= max_chars:
                current, current_len = list(tail), tail_len
            else:
                current, current_len = [], 0
        current.append(unit)
        current_len += len(unit)

    if current:
        chunks.append("\n".join(current))

    return chunks
