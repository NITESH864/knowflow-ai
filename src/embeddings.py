
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into normalized embeddings.

    Returns a float32 NumPy array with shape:
    (number_of_texts, embedding_dimension)
    """
    if not texts:
        raise ValueError("texts cannot be empty")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text must be a non-empty string")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)
