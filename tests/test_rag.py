
import pytest

from src import rag


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.last_query = None

    def search(self, query, top_k=3):
        self.last_query = query
        return self.results


def test_answer_question_uses_relevant_context(monkeypatch):
    retriever = FakeRetriever([
        {
            "score": 0.80,
            "source": "sample_policy.pdf",
            "page_number": 1,
            "text": "Employees receive 20 days of annual leave.",
        }
    ])

    def fake_generate_text(prompt):
        assert "20 days of annual leave" in prompt
        return "Employees receive 20 days of annual leave."

    monkeypatch.setattr(rag, "generate_text", fake_generate_text)

    result = rag.answer_question(
        "How many annual leave days do employees receive?",
        retriever,
    )

    assert "20 days" in result["answer"]
    assert result["sources"] == [
        {"source": "sample_policy.pdf", "page_number": 1}
    ]


def test_low_relevance_does_not_call_llm(monkeypatch):
    retriever = FakeRetriever([
        {
            "score": 0.30,
            "source": "sample_policy.pdf",
            "page_number": 1,
            "text": "Employees receive 20 days of annual leave.",
        }
    ])

    def unexpected_llm_call(prompt):
        pytest.fail("LLM should not be called for low-relevance context")

    monkeypatch.setattr(rag, "generate_text", unexpected_llm_call)

    result = rag.answer_question(
        "What is the maternity leave policy?",
        retriever,
    )

    assert result["answer"] == rag.NO_ANSWER
    assert result["sources"] == []


def test_irrelevant_chunks_are_excluded(monkeypatch):
    retriever = FakeRetriever([
        {
            "score": 0.80,
            "source": "sample_policy.pdf",
            "page_number": 1,
            "text": "Employees receive 20 days of annual leave.",
        },
        {
            "score": 0.20,
            "source": "sample_policy.pdf",
            "page_number": 2,
            "text": "Employees may work remotely with manager approval.",
        },
    ])

    def fake_generate_text(prompt):
        assert "20 days of annual leave" in prompt
        assert "work remotely" not in prompt
        return "Employees receive 20 days of annual leave."

    monkeypatch.setattr(rag, "generate_text", fake_generate_text)

    result = rag.answer_question(
        "How many annual leave days are available?",
        retriever,
    )

    assert len(result["sources"]) == 1
    assert result["sources"][0]["page_number"] == 1


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(question):
    retriever = FakeRetriever([])

    with pytest.raises(ValueError, match="Question cannot be empty"):
        rag.answer_question(question, retriever)


def test_invalid_threshold_is_rejected():
    retriever = FakeRetriever([])

    with pytest.raises(ValueError, match="relevance_threshold"):
        rag.answer_question(
            "Annual leave policy",
            retriever,
            relevance_threshold=1.5,
        )
