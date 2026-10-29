
import numpy as np

from src.embeddings import embed_texts


texts = [
    "Employees receive 20 days of annual leave.",
    "Staff members can take paid vacation days.",
    "The computer needs a graphics driver.",
]

vectors = embed_texts(texts)

print("Embedding shape:", vectors.shape)
print("Data type:", vectors.dtype)
print("All values finite:", np.isfinite(vectors).all())

similarities = vectors @ vectors.T

print("\nSimilarity matrix:")
print(np.round(similarities, 3))

print("\nPolicy similarity:", round(float(similarities[0, 1]), 3))
print("Unrelated similarity:", round(float(similarities[0, 2]), 3))
