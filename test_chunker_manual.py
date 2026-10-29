
from src.pdf_reader import extract_text_from_pdf
from src.chunker import chunk_pages


pages = extract_text_from_pdf("data/sample_policy.pdf")
chunks = chunk_pages(pages, chunk_size=60, overlap=10)

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index} | Page {chunk['page_number']}")
    print(chunk["text"])
