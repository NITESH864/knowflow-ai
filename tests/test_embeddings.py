
import numpy as np
import pytest

from src import embeddings


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )


def test_embed_texts_returns_float32_vectors(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    vectors = embeddings.embed_texts(["Annual leave", "Remote work"])

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()


def test_empty_list_is_rejected():
    with pytest.raises(ValueError, match="texts cannot be empty"):
        embeddings.embed_texts([])


@pytest.mark.parametrize("texts", [[""], ["   "], ["Annual leave", " "]])
def test_blank_text_is_rejected(texts):
    with pytest.raises(ValueError, match="non-empty string"):
        embeddings.embed_texts(texts)
