# app/services/embedder.py

import os
from dotenv import load_dotenv
from typing import List
from google import generativeai as genai

load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embeddings_gemini(text_chunks: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks using the Gemini API.

    Args:
        text_chunks: A list of strings to be embedded.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """
    # Use the correct model for text embeddings
    model = "models/embedding-001"
    
    # Call the embedding model once with all the chunks
    # The API handles batching internally.
    result = genai.embed_content(
        model=model,
        content=text_chunks,  # Pass the entire list of chunks
        task_type="retrieval_document",
        title="Embedding of documents" # Optional but recommended
    )

    return result["embedding"] if isinstance(result["embedding"][0], list) else [result["embedding"]]