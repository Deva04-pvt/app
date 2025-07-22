# app/services/openai_embedder.py

import os
from typing import List
from services.openai_client import get_openai_client

def get_embeddings_openai(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """
    Generate embeddings using OpenAI's embedding models.
    
    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model to use
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    try:
        client = get_openai_client()
        embeddings = client.get_embeddings(texts, model=model)
        return embeddings
    except Exception as e:
        print(f"Error generating OpenAI embeddings: {e}")
        raise

def get_embedding_openai(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Generate embedding for a single text using OpenAI.
    
    Args:
        text: Text string to embed
        model: OpenAI embedding model to use
        
    Returns:
        Embedding vector
    """
    embeddings = get_embeddings_openai([text], model=model)
    return embeddings[0] if embeddings else []
