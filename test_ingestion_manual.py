
from src.ingestion import ingest_pdf


result = ingest_pdf(
    "data/sample_policy.pdf",
    chunk_size=60,
    overlap=10,
)

print("Source:", result["source"])
print("Total pages:", result["total_pages"])
print("Total chunks:", result["total_chunks"])

for index, chunk in enumerate(result["chunks"], start=1):
    print(f"\nChunk {index} | Page {chunk['page_number']}")
    print(chunk["text"])
