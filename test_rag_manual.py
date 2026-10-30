
from src.retriever import DocumentRetriever
from src.rag import answer_question

PDF_PATH = "data/sample_policy.pdf"

retriever = DocumentRetriever()

print("Indexing PDF...")
summary = retriever.index_pdf(PDF_PATH)
print("Indexed:", summary)

questions = [
    "How many annual leave days do employees receive?",
    "When can employees work remotely?",
    "What is the company's maternity leave policy?",
]

for question in questions:
    print("\n" + "=" * 60)
    print("QUESTION:", question)

    result = answer_question(question, retriever)

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED SOURCES:")
    for source in result["sources"]:
        print(
            f"- {source['source']}, "
            f"Page {source['page_number']}"
        )
