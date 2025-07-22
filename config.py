# app/config.py

"""
Configuration settings for the document processing pipeline.
"""

# Chunking Configuration
CHUNKING_CONFIG = {
    "max_tokens": 400,           # Maximum tokens per chunk
    "overlap_tokens": 50,        # Overlap between chunks
    "min_chunk_length": 30,      # Minimum characters for valid chunk
    "use_smart_chunking": True,  # Use token-aware smart chunking
}

# Text Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "min_semantic_density": 0.2,     # Minimum information density
    "max_noise_ratio": 0.7,          # Maximum noise-to-content ratio
    "enable_quality_filtering": True,  # Enable chunk quality filtering
    "remove_headers_footers": True,   # Remove page headers/footers
    "normalize_unicode": True,        # Normalize unicode characters
}

# Answer Generation Configuration
ANSWER_CONFIG = {
    "max_context_chunks": 3,         # Maximum chunks to use for answer generation
    "min_relevance_threshold": 0.1,  # Minimum relevance score for chunk inclusion
    "enable_advanced_filtering": True, # Use advanced context filtering
    "fallback_to_basic": True,       # Fallback to basic filtering if advanced fails
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "default_top_k": 5,              # Default number of chunks to retrieve
    "max_top_k": 10,                 # Maximum allowed top_k value
    "distance_threshold": None,      # Optional distance threshold for filtering
}

# Model Configuration
MODEL_CONFIG = {
    "embedding_provider": "gemini",      # "gemini" or "openai"
    "generation_provider": "gemini",     # "gemini" or "openai"
    "embedding_model": "gemini",         # For Gemini embeddings
    "openai_embedding_model": "text-embedding-ada-002",  # For OpenAI embeddings
    "generation_model": "gemini-1.5-flash",             # For Gemini generation
    "openai_generation_model": "gpt-3.5-turbo",         # For OpenAI generation
    "tokenizer_model": "cl100k_base",                    # Tokenizer model for chunking
}

def get_chunking_config():
    """Get chunking configuration."""
    return CHUNKING_CONFIG.copy()

def get_preprocessing_config():
    """Get preprocessing configuration."""
    return PREPROCESSING_CONFIG.copy()

def get_answer_config():
    """Get answer generation configuration."""
    return ANSWER_CONFIG.copy()

def get_retrieval_config():
    """Get retrieval configuration."""
    return RETRIEVAL_CONFIG.copy()

def get_model_config():
    """Get model configuration."""
    return MODEL_CONFIG.copy()
