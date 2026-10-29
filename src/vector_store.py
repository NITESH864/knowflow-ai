
import faiss
import numpy as np


class VectorStore:
    """Store document chunks and retrieve them using vector similarity."""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.embedding_dimension = None

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
    ) -> None:
        """Build a FAISS index from document chunk embeddings."""
        if not chunks:
            raise ValueError("chunks cannot be empty")

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        if vectors.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        if vectors.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero")

        if not np.isfinite(vectors).all():
            raise ValueError("embeddings must contain only finite values")

        # Our embeddings are normalized, so inner product gives cosine similarity.
        self.embedding_dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index.add(np.ascontiguousarray(vectors))

        # Keep each vector's metadata aligned with its FAISS position.
        self.chunks = list(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        """Return the most similar chunks for a query embedding."""
        if self.index is None:
            raise ValueError("Vector store is empty. Add chunks first.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("Query embedding must contain exactly one vector")

        if query.shape[1] != self.embedding_dimension:
            raise ValueError("Query embedding dimension does not match the index")

        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain only finite values")

        query = np.ascontiguousarray(query)
        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(query, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk = self.chunks[int(idx)]

            results.append({
                "score": float(score),
                "page_number": chunk["page_number"],
                "text": chunk["text"],
            })

        return results
