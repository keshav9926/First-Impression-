# tests/test_chunker.py — unit tests for the hand-written chunker.
#
# TARGET: app/ingestion/chunker.py chunk_text() — a pure function
# (text in, list out — no network, no state), which is exactly what makes
# it easy to test thoroughly. Each test pins down one guarantee the rest
# of the pipeline relies on.

from app.ingestion.chunker import chunk_text


def test_empty_text_gives_no_chunks():
    """No content in → no chunks out (and no crash on whitespace-only text)."""
    assert chunk_text("") == []
    assert chunk_text("   \n  \n ") == []


def test_short_text_is_one_chunk():
    """Text that fits the budget stays together as a single chunk."""
    text = "First paragraph.\nSecond paragraph."
    chunks = chunk_text(text, max_chars=1600)
    assert len(chunks) == 1
    assert "First paragraph." in chunks[0]
    assert "Second paragraph." in chunks[0]


def test_long_text_is_split_into_multiple_chunks():
    """Text beyond the budget must be split (10 × ~80 chars vs a 200-char budget)."""
    text = "\n".join(f"Paragraph number {i} " + "x" * 60 for i in range(10))
    chunks = chunk_text(text, max_chars=200, overlap=0)
    assert len(chunks) > 1


def test_overlap_carries_last_paragraph_forward():
    """With overlap=1, each chunk's last paragraph reappears at the start of
    the next chunk — the continuity guarantee for topics that straddle a
    chunk boundary."""
    paragraphs = [f"para-{i} " + "y" * 50 for i in range(6)]
    chunks = chunk_text("\n".join(paragraphs), max_chars=150, overlap=1)
    for previous, current in zip(chunks, chunks[1:]):
        last_para_of_previous = previous.split("\n")[-1]
        assert current.startswith(last_para_of_previous)


def test_giant_single_paragraph_is_hard_split():
    """A single paragraph bigger than the whole budget gets hard-split into
    budget-sized pieces, and no characters are lost in the process."""
    text = "z" * 5000  # one paragraph, no newlines
    chunks = chunk_text(text, max_chars=1000, overlap=0)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks) == text  # nothing lost
