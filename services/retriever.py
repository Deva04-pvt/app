# app/services/retriever.py
import numpy as np
import faiss
from typing import List, Tuple

from services.embedder import get_embeddings
from services.document_store import DocumentStore


def search_index(
    index: faiss.Index,
    doc_store: DocumentStore,
    query: str,
    top_k: int = 5,
    nprobe: int = 10,  # Number of nearby clusters to search. Higher is more accurate but slower.
) -> List[Tuple[str, float]]:
    """
    Searches the index for a query and retrieves the relevant text chunks.
    """
    # Set how many clusters to search. This is the key to the speed/accuracy tradeoff.
    index.nprobe = nprobe

    # 1. Embed the query
    query_embedding = get_embeddings([query])[0]
    if query_embedding is None:
        print("Failed to generate query embedding.")
        return []

    query_vector = np.array([query_embedding], dtype="float32")

    # 2. Search the index
    # Returns L2 distances and the original IDs we stored.
    distances, ids = index.search(query_vector, top_k)

    # 3. Retrieve chunks from the document store using the IDs
    retrieved_ids = [int(i) for i in ids[0] if i != -1]
    if not retrieved_ids:
        return []

    retrieved_chunks = doc_store.get_by_ids(retrieved_ids)

    # 4. Map results to text chunks and scores
    results = []
    for i, chunk in enumerate(retrieved_chunks):
        score = float(distances[0][i])
        results.append((chunk, score))

    return results
