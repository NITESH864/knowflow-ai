
from src.retriever import DocumentRetriever

PDF_PATH = r"C:\Users\Nitesh Gupta\Downloads\Tesla_Info.pdf"

retriever = DocumentRetriever()
info = retriever.index_pdf(PDF_PATH)

print("Indexed:", info)

questions = [
    "What was Tesla originally called?",
    "What are Tesla's main business areas?",
    "What are Tesla's main business areas and headquarters?"
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    results = retriever.search(question, top_k=5)

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Score:", round(result["score"], 4))
        print("Page:", result.get("page_number"))
        print("Source:", result.get("source"))
        print("Text:")
        print(result["text"][:1500])
        print("-" * 60)
