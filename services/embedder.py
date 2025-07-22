# app/services/embedder.py
#
# Required libraries:
# pip install ollama tqdm

import os
import ollama
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---
# Use environment variables for flexible configuration.
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
# The ollama client library will automatically use OLLAMA_HOST or the default localhost.
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))


class EmbeddingError(Exception):
    """Custom exception for embedding failures."""

    pass


def get_embedding_for_chunk(chunk: str) -> List[float]:
    """
    Generates an embedding for a single text chunk.
    This function will be run in a separate thread for each chunk.
    """
    try:
        # The ollama client handles the HTTP request and response parsing.
        response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=chunk)
        return response["embedding"]
    except Exception as e:
        # Catch potential errors from the ollama client or network issues.
        raise EmbeddingError(f"Failed to get embedding for chunk. Error: {e}")


def get_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks in parallel.

    Args:
        text_chunks: A list of strings to be embedded.

    Returns:
        A list of embeddings in the same order as the input chunks.
    """
    # Create a dictionary to map future objects to their original index.
    # This ensures we can return the embeddings in the correct order.
    future_to_index = {}
    results = [None] * len(text_chunks)

    # Use ThreadPoolExecutor to make requests concurrently.
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        # Submit all tasks to the executor.
        for i, chunk in enumerate(text_chunks):
            future = executor.submit(get_embedding_for_chunk, chunk)
            future_to_index[future] = i

        # Use tqdm to create a progress bar as tasks complete.
        for future in tqdm(
            as_completed(future_to_index),
            total=len(text_chunks),
            desc="Generating Embeddings",
        ):
            index = future_to_index[future]
            try:
                # Get the result from the completed future.
                results[index] = future.result()
            except EmbeddingError as e:
                # Log the error and continue, or handle as needed.
                # Here we place a None, which can be filtered out later.
                print(f"Warning: Chunk {index} failed to embed. {e}")
                results[index] = None

    # Filter out any chunks that failed to embed.
    return [res for res in results if res is not None]
