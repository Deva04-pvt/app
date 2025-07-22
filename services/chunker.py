# app/services/chunker.py

import re
from typing import List

def clean_extracted_text(text: str) -> str:
    """
    Clean raw extracted text before chunking.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page headers/footers patterns
    text = re.sub(r'Page\s+\d+.*?\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Fix common OCR issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space before capitals
    text = re.sub(r'\.([A-Z])', r'. \1', text)  # Add space after periods
    
    # Remove excessive punctuation
    text = re.sub(r'[\.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', '---', text)
    
    return text.strip()

def split_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """
    Improved chunking with better sentence boundary detection and content preservation.
    """
    # Clean the text first
    text = clean_extracted_text(text)
    
    # Better sentence splitting that handles common abbreviations
    # Split on sentence endings but preserve abbreviations like Dr., Mr., etc.
    sentence_pattern = r'(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Co|vs|etc|i\.e|e\.g))\s*[.!?]+\s+'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if adding this sentence would exceed max_length
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Save current chunk if it has meaningful content
            if current_chunk and len(current_chunk.strip()) > 20:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    # Don't forget the last chunk
    if current_chunk and len(current_chunk.strip()) > 20:
        chunks.append(current_chunk.strip())

    # Add overlap between chunks for better context preservation
    final_chunks = []
    for i in range(len(chunks)):
        chunk = chunks[i]
        
        # Add overlap from previous chunk
        if i > 0 and overlap > 0:
            prev_chunk = chunks[i - 1]
            # Take last 'overlap' characters from previous chunk
            overlap_text = prev_chunk[-overlap:].strip()
            # Find a good breaking point (end of word)
            if overlap_text:
                word_boundary = overlap_text.rfind(' ')
                if word_boundary > overlap // 2:  # Only if we can get a reasonable portion
                    overlap_text = overlap_text[word_boundary:].strip()
                if overlap_text:
                    chunk = overlap_text + " " + chunk
        
        final_chunks.append(chunk)

    # Filter out very short or empty chunks
    final_chunks = [chunk for chunk in final_chunks if len(chunk.strip()) > 30]
    
    return final_chunks
