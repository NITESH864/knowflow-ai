
from pathlib import Path

from src.pdf_reader import extract_text_from_pdf
from src.chunker import chunk_pages


def ingest_pdf(
    pdf_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
) -> dict:
    """
    Extract text from a PDF and split it into chunks.

    Returns the source path, page count, and extracted chunks.
    """
    pages = extract_text_from_pdf(pdf_path)

    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return {
        "source": Path(pdf_path).name,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
