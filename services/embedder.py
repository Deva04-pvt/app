import requests
from typing import List

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api/embeddings"


def get_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks using a local Ollama model.

    Args:
        text_chunks: A list of strings to be embedded.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """
    embeddings = []

    for chunk in text_chunks:
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": chunk},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "embedding" in data:
                embeddings.append(data["embedding"])
            else:
                raise ValueError("Embedding not returned in response")

        except requests.RequestException as e:
            raise RuntimeError(f"Embedding request failed: {e}")

    return embeddings
