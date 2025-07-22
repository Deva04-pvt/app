# app/services/smart_chunker.py

import re
import os
import sys
from typing import List, Dict, Tuple

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from services.text_preprocessor import TextPreprocessor

class SmartChunker:
    """
    Token-aware chunker that creates higher quality chunks for better embeddings.
    """
    
    def __init__(self, model_name: str = "cl100k_base", max_tokens: int = 400, overlap_tokens: int = 50):
        """
        Initialize the smart chunker.
        
        Args:
            model_name: The tokenizer model to use
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
        """
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.preprocessor = TextPreprocessor()
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        return self.tokenizer.encode(text)
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(tokens)
    
    def split_by_semantic_boundaries(self, text: str) -> List[str]:
        """
        Split text at semantic boundaries (paragraphs, sentences, etc.)
        """
        # First split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        segments = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph is short enough, keep as is
            if self.count_tokens(paragraph) <= self.max_tokens:
                segments.append(paragraph)
            else:
                # Split by sentences within the paragraph
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_segment = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    test_segment = current_segment + " " + sentence if current_segment else sentence
                    
                    if self.count_tokens(test_segment) <= self.max_tokens:
                        current_segment = test_segment
                    else:
                        if current_segment:
                            segments.append(current_segment)
                        current_segment = sentence
                
                if current_segment:
                    segments.append(current_segment)
        
        return segments
    
    def create_overlapping_chunks(self, segments: List[str]) -> List[str]:
        """
        Create chunks with token-aware overlapping.
        """
        if not segments:
            return []
            
        chunks = []
        current_chunk_tokens = []
        
        for segment in segments:
            segment_tokens = self.tokenize_text(segment)
            
            # If current chunk + new segment exceeds max tokens
            if len(current_chunk_tokens) + len(segment_tokens) > self.max_tokens:
                if current_chunk_tokens:
                    # Save current chunk
                    chunk_text = self.detokenize(current_chunk_tokens)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap from previous
                    if len(current_chunk_tokens) > self.overlap_tokens:
                        overlap_start = len(current_chunk_tokens) - self.overlap_tokens
                        current_chunk_tokens = current_chunk_tokens[overlap_start:]
                    else:
                        current_chunk_tokens = []
                
                # Add new segment
                current_chunk_tokens.extend(segment_tokens)
            else:
                # Add segment to current chunk
                if current_chunk_tokens:
                    current_chunk_tokens.append(self.tokenize_text(" ")[0])  # Add space
                current_chunk_tokens.extend(segment_tokens)
        
        # Don't forget the last chunk
        if current_chunk_tokens:
            chunk_text = self.detokenize(current_chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def filter_low_quality_chunks(self, chunks: List[str]) -> List[str]:
        """
        Filter out low-quality chunks using the text preprocessor.
        """
        quality_chunks = []
        
        for chunk in chunks:
            # Clean the chunk
            cleaned = self.preprocessor.clean_chunk(chunk)
            if not cleaned:
                continue
                
            # Check semantic density
            density = self.preprocessor.calculate_semantic_density(cleaned)
            if density < 0.2:  # Very low information content
                continue
                
            # Check minimum token count
            if self.count_tokens(cleaned) < 20:  # Too short to be meaningful
                continue
                
            quality_chunks.append(cleaned)
        
        return quality_chunks
    
    def smart_chunk_text(self, text: str) -> List[str]:
        """
        Main method to create high-quality, token-aware chunks.
        
        Args:
            text: Raw text to chunk
            
        Returns:
            List of high-quality text chunks
        """
        # Preprocess the text
        cleaned_text = self.preprocessor.normalize_unicode(text)
        cleaned_text = self.preprocessor.remove_excessive_whitespace(cleaned_text)
        
        # Split into semantic segments
        segments = self.split_by_semantic_boundaries(cleaned_text)
        
        # Create overlapping chunks
        chunks = self.create_overlapping_chunks(segments)
        
        # Filter low quality chunks
        quality_chunks = self.filter_low_quality_chunks(chunks)
        
        return quality_chunks

# Convenience function for backward compatibility
def smart_split_text(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> List[str]:
    """
    Smart text splitting with token awareness and quality filtering.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
        
    Returns:
        List of high-quality chunks
    """
    chunker = SmartChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    return chunker.smart_chunk_text(text)
