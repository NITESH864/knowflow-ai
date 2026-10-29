
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Split page text into overlapping character-based chunks.

    Each chunk retains its original page number.
    """

    # Validate chunk settings
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        page_number = page["page_number"]
        text = page["text"].strip()

        # Skip empty pages
        if not text:
            continue

        start = 0

        while start < len(text):
            chunk_text = text[start:start + chunk_size].strip()

            if chunk_text:
                chunks.append({
                    "page_number": page_number,
                    "text": chunk_text,
                })

            # Stop when the current chunk reaches the end.
            # This prevents a redundant trailing chunk.
            if start + chunk_size >= len(text):
                break

            start += step

    return chunks
