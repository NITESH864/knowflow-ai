
import numpy as np
import pytest

from src import retriever as retriever_module
from src.retriever import DocumentRetriever


@pytest.fixture
def setup_retriever(monkeypatch):
    chunks = [
        {"page_number": 1, "text": "Annual leave policy"},
        {"page_number": 2, "text": "Remote work policy"},
    ]

    def fake_ingest_pdf(pdf_path, chunk_size=500, overlap=50):
        return {
            "source": "sample_policy.pdf",
            "total_pages": 2,
            "total_chunks": 2,
            "chunks": chunks,
        }

    def fake_embed_texts(texts):
        if texts == [chunk["text"] for chunk in chunks]:
            return np.array(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            )

        if "leave" in texts[0].lower():
            return np.array([[1.0, 0.0]], dtype=np.float32)

        return np.array([[0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(retriever_module, "ingest_pdf", fake_ingest_pdf)
    monkeypatch.setattr(retriever_module, "embed_texts", fake_embed_texts)

    instance = DocumentRetriever()
    summary = instance.index_pdf("sample_policy.pdf")

    return instance, summary


def test_index_pdf_returns_summary(setup_retriever):
    _, summary = setup_retriever

    assert summary["source"] == "sample_policy.pdf"
    assert summary["total_pages"] == 2
    assert summary["total_chunks"] == 2
    assert summary["embedding_dimension"] == 2


def test_search_returns_relevant_chunk(setup_retriever):
    instance, _ = setup_retriever

    results = instance.search("annual leave", top_k=1)

    assert len(results) == 1
    assert results[0]["page_number"] == 1
    assert results[0]["text"] == "Annual leave policy"
    assert results[0]["source"] == "sample_policy.pdf"


def test_search_returns_remote_work_chunk(setup_retriever):
    instance, _ = setup_retriever

    results = instance.search("remote work", top_k=1)

    assert results[0]["page_number"] == 2
    assert results[0]["text"] == "Remote work policy"


def test_blank_query_is_rejected():
    instance = DocumentRetriever()

    with pytest.raises(ValueError, match="non-empty string"):
        instance.search("   ")


def test_search_without_indexing_is_rejected():
    instance = DocumentRetriever()

    with pytest.raises(ValueError, match="No document indexed"):
        instance.search("annual leave")
