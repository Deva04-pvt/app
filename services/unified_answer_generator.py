# app/services/unified_answer_generator.py

from typing import List
from config import get_model_config

def generate_answer_unified(context_chunks: List[str], question: str) -> str:
    """
    Generate an answer using the configured provider (Gemini or OpenAI).
    
    Args:
        context_chunks: List of text strings providing context
        question: The question to be answered
        
    Returns:
        Generated answer
    """
    model_config = get_model_config()
    provider = model_config["generation_provider"]
    
    if provider == "openai":
        from services.openai_answer_generator import generate_answer_openai
        model = model_config["openai_generation_model"]
        return generate_answer_openai(context_chunks, question, model=model)
    else:  # Default to Gemini
        from services.answer_generator import generate_answer
        return generate_answer(context_chunks, question)
