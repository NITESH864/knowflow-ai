
from src.retriever import DocumentRetriever

PDF_PATH = "data/sample_policy.pdf"

retriever = DocumentRetriever()

summary = retriever.index_pdf(PDF_PATH)

print("\n--- INDEX SUMMARY ---")
print(summary)

questions = [
    "How many annual leave days do employees receive?",
    "When can employees work remotely?",
]

for question in questions:
    print(f"\n--- Question: {question} ---")

    results = retriever.search(question, top_k=2)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page_number']}")
        print(f"Similarity: {result['score']:.3f}")
        print(f"Text: {result['text']}")
