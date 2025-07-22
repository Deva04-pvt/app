# app/services/answer_generator.py

import os
import re
import sys
from typing import List, Tuple

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from google import generativeai as genai
from services.text_preprocessor import TextPreprocessor
from config import get_answer_config, get_model_config

load_dotenv()
# It's better to get the API key from environment variables
# rather than hardcoding it.
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the text preprocessor and get config
preprocessor = TextPreprocessor()
answer_config = get_answer_config()
model_config = get_model_config()

def clean_text(text: str) -> str:
    """
    Clean and normalize text chunk.
    
    Args:
        text: Raw text chunk to clean
        
    Returns:
        Cleaned text
    """
    if not text or not text.strip():
        return ""
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common OCR artifacts and noise
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>\~\`]', ' ', text)
    
    # Remove very short fragments that are likely noise
    if len(text) < 20:
        return ""
    
    # Remove chunks that are mostly numbers or special characters
    word_chars = re.findall(r'[a-zA-Z]', text)
    if len(word_chars) < len(text) * 0.3:  # Less than 30% actual letters
        return ""
    
    return text

def calculate_relevance_score(chunk: str, question: str) -> float:
    """
    Calculate relevance score between chunk and question using keyword overlap.
    
    Args:
        chunk: Text chunk to score
        question: User question
        
    Returns:
        Relevance score (0-1)
    """
    # Simple keyword-based relevance scoring
    question_words = set(re.findall(r'\b\w+\b', question.lower()))
    chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
    
    # Remove common stop words for better relevance
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    question_words = question_words - stop_words
    chunk_words = chunk_words - stop_words
    
    if not question_words:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(question_words & chunk_words)
    union = len(question_words | chunk_words)
    
    return intersection / union if union > 0 else 0.0

def filter_and_rank_chunks(context_chunks: List[str], question: str, max_chunks: int = None) -> List[str]:
    """
    Clean, filter, and rank context chunks by relevance.
    
    Args:
        context_chunks: List of raw context chunks
        question: User question
        max_chunks: Maximum number of chunks to keep (uses config if None)
        
    Returns:
        List of cleaned and filtered chunks, ranked by relevance
    """
    if max_chunks is None:
        max_chunks = answer_config["max_context_chunks"]
    
    min_relevance = answer_config["min_relevance_threshold"]
    # Clean all chunks
    cleaned_chunks = []
    for chunk in context_chunks:
        cleaned = clean_text(chunk)
        if cleaned:  # Only keep non-empty cleaned chunks
            cleaned_chunks.append(cleaned)
    
    if not cleaned_chunks:
        return []
    
    # Calculate relevance scores
    chunk_scores = []
    for chunk in cleaned_chunks:
        score = calculate_relevance_score(chunk, question)
        chunk_scores.append((chunk, score))
    
    # Sort by relevance score (highest first)
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out chunks with very low relevance
    relevant_chunks = [chunk for chunk, score in chunk_scores if score >= min_relevance]
    
    # Return top max_chunks
    return relevant_chunks[:max_chunks]

def generate_answer(context_chunks: list[str], question: str) -> str:
    """
    Generates an answer to a question based on provided context chunks.
    Now includes advanced context cleanup and filtering for improved accuracy.

    Args:
        context_chunks: A list of text strings providing context.
        question: The question to be answered.

    Returns:
        The generated answer as a string.
    """
    # Use advanced preprocessing to clean and filter context chunks
    try:
        if answer_config["enable_advanced_filtering"]:
            # First, use the advanced preprocessor to clean and rank chunks
            cleaned_chunks = preprocessor.preprocess_document_chunks(
                context_chunks, 
                max_chunks=answer_config["max_context_chunks"]
            )
            
            if not cleaned_chunks:
                return "The answer is not available in the provided context."
            
            # Then apply question-specific filtering
            filtered_chunks = filter_and_rank_chunks(cleaned_chunks, question)
            
            if not filtered_chunks:
                return "The answer is not available in the provided context."
        else:
            # Use basic filtering only
            filtered_chunks = filter_and_rank_chunks(context_chunks, question)
            
    except Exception as e:
        # Fallback to basic filtering if advanced preprocessing fails
        if answer_config["fallback_to_basic"]:
            print(f"Advanced preprocessing failed: {e}. Using basic filtering.")
            filtered_chunks = filter_and_rank_chunks(context_chunks, question)
            
            if not filtered_chunks:
                return "The answer is not available in the provided context."
        else:
            raise e
    
    # Join the filtered context chunks into a single block of text
    context = "\n\n".join(filtered_chunks)

    prompt = f"""
You are a helpful assistant who answers questions based ONLY on the provided context.

Context:
---
{context}
---

Question: {question}

Instructions:
1. Read the context carefully and identify the most relevant information.
2. Formulate a clear, concise, and accurate answer based strictly on the information given in the context.
3. If the information needed to answer the question is not in the context, you must respond with exactly this phrase: "The answer is not available in the provided context."
4. Focus on providing factual information and avoid making assumptions or inferences beyond what is explicitly stated.
5. If multiple pieces of information are relevant, synthesize them coherently.
"""

    # Initialize the generative model
    model = genai.GenerativeModel(model_config["generation_model"])

    # Generate the content
    response = model.generate_content(prompt)

    return response.text.strip()