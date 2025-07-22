# app/services/faiss_indexer.py

import faiss
import numpy as np
from typing import List

def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_index(index: faiss.IndexFlatL2, query_vector: List[float], top_k: int = 5):
    D, I = index.search(np.array([query_vector]).astype('float32'), top_k)
    return I[0], D[0]
