
import pytest

from src.chunker import chunk_pages


def test_short_page_creates_one_chunk():
    pages = [{"page_number": 1, "text": "Employee leave policy."}]

    chunks = chunk_pages(pages, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["text"] == "Employee leave policy."


def test_long_text_is_split_with_overlap():
    pages = [{"page_number": 2, "text": "ABCDEFGHIJ"}]

    chunks = chunk_pages(pages, chunk_size=6, overlap=2)

    assert [chunk["text"] for chunk in chunks] == [
        "ABCDEF",
        "EFGHIJ",
    ]
    assert all(chunk["page_number"] == 2 for chunk in chunks)


def test_empty_pages_are_skipped():
    pages = [{"page_number": 1, "text": "   "}]

    assert chunk_pages(pages) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_chunk_settings_raise_error(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(
            [{"page_number": 1, "text": "Example"}],
            chunk_size=chunk_size,
            overlap=overlap,
        )
