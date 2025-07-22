# app/services/retriever.py

from typing import List, Tuple
import numpy as np
import faiss
from services.embedder import get_embeddings


def search_faiss_index(
    index: faiss.IndexFlatL2, query: str, chunks: List[str], top_k: int = 5
) -> List[Tuple[str, float]]:
    # Embed the query (Gemini returns batch even for single input)
    query_embedding = get_embeddings([query])[0]
    query_vector = np.array([query_embedding], dtype="float32")

    # Search the index
    distances, indices = index.search(query_vector, top_k)

    # Map results to text chunks
    results = []
    for i, score in zip(indices[0], distances[0]):
        results.append((chunks[i], float(score)))

    return results
