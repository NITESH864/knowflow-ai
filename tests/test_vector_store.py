
import numpy as np
import pytest

from src.vector_store import VectorStore


@pytest.fixture
def sample_store():
    chunks = [
        {"page_number": 1, "text": "Employees receive 20 days of annual leave."},
        {"page_number": 2, "text": "Employees may work remotely with manager approval."},
        {"page_number": 3, "text": "The office uses computer graphics software."},
    ]

    # Simple normalized vectors for testing; no embedding model needed.
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store = VectorStore()
    store.add_chunks(chunks, embeddings)
    return store


def test_search_returns_most_similar_chunk(sample_store):
    query = np.array([0.99, 0.01, 0.0], dtype=np.float32)

    results = sample_store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0]["page_number"] == 1
    assert "annual leave" in results[0]["text"]
    assert results[0]["score"] > results[1]["score"]


def test_top_k_is_limited_to_available_chunks(sample_store):
    results = sample_store.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=10,
    )

    assert len(results) == 3


def test_empty_chunks_are_rejected():
    store = VectorStore()

    with pytest.raises(ValueError, match="chunks cannot be empty"):
        store.add_chunks([], np.empty((0, 3), dtype=np.float32))


def test_mismatched_embedding_count_is_rejected():
    store = VectorStore()
    chunks = [{"page_number": 1, "text": "Annual leave"}]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="must match"):
        store.add_chunks(chunks, embeddings)


def test_search_before_adding_chunks_is_rejected():
    store = VectorStore()

    with pytest.raises(ValueError, match="Vector store is empty"):
        store.search(np.array([1.0, 0.0], dtype=np.float32))


def test_invalid_top_k_is_rejected(sample_store):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        sample_store.search(np.array([1.0, 0.0, 0.0]), top_k=0)
