# tests/test_chunker.py — unit tests for the hand-written chunker.
# The chunker is a pure function (text in, list out — no network, no state),
# which is exactly what makes it easy to test thoroughly.

from app.ingestion.chunker import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  \n ") == []


def test_short_text_is_one_chunk():
    text = "First paragraph.\nSecond paragraph."
    chunks = chunk_text(text, max_chars=1600)
    assert len(chunks) == 1
    assert "First paragraph." in chunks[0]
    assert "Second paragraph." in chunks[0]


def test_long_text_is_split_into_multiple_chunks():
    # 10 paragraphs of ~80 chars with a 200-char budget must split.
    text = "\n".join(f"Paragraph number {i} " + "x" * 60 for i in range(10))
    chunks = chunk_text(text, max_chars=200, overlap=0)
    assert len(chunks) > 1


def test_overlap_carries_last_paragraph_forward():
    paragraphs = [f"para-{i} " + "y" * 50 for i in range(6)]
    chunks = chunk_text("\n".join(paragraphs), max_chars=150, overlap=1)
    # The last paragraph of each chunk should reappear at the start of the next.
    for previous, current in zip(chunks, chunks[1:]):
        last_para_of_previous = previous.split("\n")[-1]
        assert current.startswith(last_para_of_previous)


def test_giant_single_paragraph_is_hard_split():
    text = "z" * 5000  # one paragraph, no newlines
    chunks = chunk_text(text, max_chars=1000, overlap=0)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks) == text  # nothing lost
