# app/services/unified_embedder.py

from typing import List
from config import get_model_config

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using the configured provider (Gemini or OpenAI).
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    model_config = get_model_config()
    provider = model_config["embedding_provider"]
    
    if provider == "openai":
        from services.openai_embedder import get_embeddings_openai
        model = model_config["openai_embedding_model"]
        return get_embeddings_openai(texts, model=model)
    else:  # Default to Gemini
        from services.embedder import get_embeddings_gemini
        return get_embeddings_gemini(texts)

def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text using the configured provider.
    
    Args:
        text: Text string to embed
        
    Returns:
        Embedding vector
    """
    embeddings = get_embeddings([text])
    return embeddings[0] if embeddings else []
