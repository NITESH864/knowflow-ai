
from src.rag import answer_question, NO_ANSWER
from src.retriever import DocumentRetriever

PDF_PATH = r"C:\Users\Nitesh Gupta\Downloads\Tesla_Info.pdf"

TEST_CASES = [
    {
        "question": "What was Tesla originally called?",
        "expected_terms": ["Tesla Motors"],
        "should_answer": True,
    },
    {
        "question": "Who founded Tesla?",
        "expected_terms": ["Martin Eberhard", "Marc Tarpenning"],
        "should_answer": True,
    },
    {
        "question": "Where is Tesla headquartered?",
        "expected_terms": ["Austin"],
        "should_answer": True,
    },
    {
        "question": "What are Tesla's main business areas?",
        "expected_terms": ["Automotive", "Energy Generation and Storage"],
        "should_answer": True,
    },
    {
        "question": "What is Tesla's industry?",
        "expected_terms": ["Automotive"],
        "should_answer": True,
    },
    {
        "question": "When was Tesla incorporated, and who founded it?",
        "expected_terms": [
        "July 1, 2003",
        "Martin Eberhard",
        "Marc Tarpenning",
    ],
        "should_answer": True,
    },
    {
        "question": "What are Tesla's main business areas and headquarters?",
        "expected_terms": ["Automotive", "Energy Generation and Storage", "Austin"],
        "should_answer": True,
    },
    {
        "question": "What is the capital of Japan?",
        "expected_terms": [],
        "should_answer": False,
    },
    {
        "question": "What is the user's bank account number?",
        "expected_terms": [],
        "should_answer": False,
    },
    {
        "question": "What was Tesla's net profit on Mars in 2025?",
        "expected_terms": [],
        "should_answer": False,
    },
]


def main():
    retriever = DocumentRetriever()
    info = retriever.index_pdf(PDF_PATH)

    print(f"\nIndexed pages: {info['total_pages']}")
    print(f"Total chunks: {info['total_chunks']}")
    print("=" * 70)

    passed = 0
    failed = 0

    for index, case in enumerate(TEST_CASES, start=1):
        question = case["question"]
        print(f"\nTest {index}: {question}")

        try:
            result = answer_question(question, retriever)
            answer = result["answer"].strip()
            sources = result.get("sources", [])

            if case["should_answer"]:
                missing_terms = [
                    term for term in case["expected_terms"]
                    if term.casefold() not in answer.casefold()
                ]

                checks = {
                    "Answer provided": answer != NO_ANSWER,
                    "Expected terms present": not missing_terms,
                    "Sources provided": bool(sources),
                    "Source pages valid": all(
                        isinstance(source.get("page_number"), int)
                        and 1 <= source["page_number"] <= info["total_pages"]
                        for source in sources
                    ),
                }
            else:
                checks = {
                    "Refused unsupported question": answer == NO_ANSWER,
                    "No sources returned": not sources,
                }
                missing_terms = []

            correct = all(checks.values())
            passed += int(correct)
            failed += int(not correct)

            print(f"Status: {'PASS' if correct else 'FAIL'}")
            print(f"Answer: {answer}")
            print(f"Sources: {sources}")

            for name, ok in checks.items():
                print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

            if missing_terms:
                print(f"Missing expected terms: {missing_terms}")

        except Exception as exc:
            failed += 1
            print("Status: ERROR")
            print(f"Error: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print(f"Evaluation: {passed}/{len(TEST_CASES)} passed")
    print(f"Failed: {failed}")
    print("Reminder: automated checks do not prove factual correctness.")


if __name__ == "__main__":
    main()
