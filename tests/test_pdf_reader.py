
import pytest
import pymupdf

from src.pdf_reader import extract_text_from_pdf


def create_test_pdf(path):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Test policy: 10 days leave.")
        document.save(path)


def test_extracts_text_and_page_numbers(tmp_path):
    pdf_path = tmp_path / "policy.pdf"
    create_test_pdf(pdf_path)

    pages = extract_text_from_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "10 days leave" in pages[0]["text"]


def test_missing_pdf_raises_error(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(missing_path)


def test_non_pdf_extension_is_rejected(tmp_path):
    text_path = tmp_path / "policy.txt"
    text_path.write_text("This is not a PDF.")

    with pytest.raises(ValueError, match="Expected a PDF"):
        extract_text_from_pdf(text_path)
