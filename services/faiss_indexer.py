# app/services/faiss_indexer.py
import faiss
import numpy as np
import os
from typing import List

# --- Configuration ---
# The dimension of your embedding model (e.g., nomic-embed-text is 768)
DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))


# This function is now corrected to accept the path dynamically.
def build_and_save_index(embeddings: List[List[float]], index_path: str):
    """
    Builds a scalable FAISS index (IndexIVFFlat) and saves it to a specified path.

    Args:
        embeddings: The list of vector embeddings.
        index_path: The file path where the index should be saved.
    """
    if not embeddings:
        print("No embeddings provided to build index.")
        return

    d = len(embeddings[0])
    if d != DIMENSIONS:
        raise ValueError(
            f"Embedding dimension mismatch. Expected {DIMENSIONS}, got {d}"
        )

    embeddings_np = np.array(embeddings).astype("float32")

    # Define the Index
    nlist = min(100, int(4 * np.sqrt(len(embeddings))))
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # Train the Index
    print(f"Training index with {len(embeddings_np)} vectors...")
    index.train(embeddings_np)
    print("Training complete.")

    # Add vectors with their IDs
    ids = np.arange(len(embeddings_np))
    index.add_with_ids(embeddings_np, ids)
    print(f"Index built successfully. Total vectors: {index.ntotal}")

    # Save the Index to the provided path
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)  # Use the 'index_path' argument here
    print(f"Index saved to {index_path}")


def load_index(index_path: str) -> faiss.Index:
    """Loads the FAISS index from the specified path."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    print(f"Loading index from {index_path}")
    return faiss.read_index(index_path)
