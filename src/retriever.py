
from pathlib import Path

from src.embeddings import embed_texts
from src.ingestion import ingest_pdf
from src.vector_store import VectorStore


class DocumentRetriever:
    """Ingest a PDF and retrieve relevant chunks using semantic search."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.source = None
        self.total_pages = 0
        self.total_chunks = 0

    def index_pdf(
        self,
        pdf_path: str | Path,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> dict:
        """Extract, chunk, embed, and index a PDF document."""
        document = ingest_pdf(
            pdf_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks = document["chunks"]

        if not chunks:
            raise ValueError("PDF contains no text to index")

        embeddings = embed_texts([chunk["text"] for chunk in chunks])

        self.vector_store.add_chunks(chunks, embeddings)

        self.source = document["source"]
        self.total_pages = document["total_pages"]
        self.total_chunks = document["total_chunks"]

        return {
            "source": self.source,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "embedding_dimension": int(embeddings.shape[1]),
        }

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Find the most relevant indexed chunks for a question."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if self.vector_store.index is None:
            raise ValueError("No document indexed. Call index_pdf first.")

        query_embedding = embed_texts([query])[0]

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        for result in results:
            result["source"] = self.source

        return results
