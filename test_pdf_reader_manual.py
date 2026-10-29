
from src.pdf_reader import extract_text_from_pdf


pages = extract_text_from_pdf("data/sample_policy.pdf")

print(f"Total pages: {len(pages)}")

for page in pages:
    print(f"\n--- Page {page['page_number']} ---")
    print(page["text"])
