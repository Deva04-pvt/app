# app/services/text_preprocessor.py

import re
from typing import List, Dict, Set
import unicodedata

class TextPreprocessor:
    """
    Advanced text preprocessing for improving document quality before embedding.
    """
    
    def __init__(self):
        # Common patterns that indicate low-quality text
        self.noise_patterns = [
            r'^[^a-zA-Z0-9]*$',  # Only special characters
            r'^[\d\s\-\.\,]{10,}$',  # Only numbers and basic punctuation
            r'^[A-Z\s]{5,}$',  # Only uppercase letters (likely headers/noise)
            r'^\s*Page\s+\d+\s*$',  # Page numbers
            r'^\s*\d+\s*$',  # Just numbers
            r'^[^\w\s]{3,}$',  # Multiple consecutive special chars
        ]
        
        # Patterns for common document artifacts
        self.artifact_patterns = [
            r'\b[A-Z]{2,}\b',  # Excessive acronyms (keep but note)
            r'www\.[^\s]+',  # URLs
            r'http[s]?://[^\s]+',  # URLs
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
            r'\b\d{4}-\d{2}-\d{2}\b',  # Dates
            r'\b\d{1,2}:\d{2}(?::\d{2})?\b',  # Times
        ]
        
        # Common stop words for relevance calculation
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 
            'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }

    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to avoid encoding issues."""
        return unicodedata.normalize('NFKD', text)

    def remove_excessive_whitespace(self, text: str) -> str:
        """Remove excessive whitespace while preserving structure."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace multiple newlines with at most two
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def is_noise_chunk(self, text: str) -> bool:
        """Determine if a text chunk is likely noise."""
        if len(text.strip()) < 15:  # Too short
            return True
            
        for pattern in self.noise_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
                
        # Check character composition
        word_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars > 0 and word_chars / total_chars < 0.3:
            return True
            
        return False

    def extract_key_phrases(self, text: str) -> Set[str]:
        """Extract potential key phrases from text."""
        # Simple approach: find sequences of 1-3 capitalized words
        phrases = set()
        
        # Multi-word phrases
        multi_word = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', text)
        phrases.update(multi_word)
        
        # Important single words (capitalized, longer than 4 chars)
        single_words = re.findall(r'\b[A-Z][a-z]{4,}\b', text)
        phrases.update(single_words)
        
        return phrases

    def clean_chunk(self, text: str) -> str:
        """Comprehensive cleaning of a text chunk."""
        if not text or not text.strip():
            return ""
            
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove excessive whitespace
        text = self.remove_excessive_whitespace(text)
        
        # Check if it's noise
        if self.is_noise_chunk(text):
            return ""
            
        # Fix common OCR/extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space before capitals
        text = re.sub(r'\.([A-Z])', r'. \1', text)  # Add space after periods
        text = re.sub(r'([a-z])(\d)', r'\1 \2', text)  # Add space before numbers
        text = re.sub(r'(\d)([a-z])', r'\1 \2', text)  # Add space after numbers
        
        # Remove excessive punctuation
        text = re.sub(r'[\.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text.strip()

    def calculate_semantic_density(self, text: str) -> float:
        """Calculate how information-dense a text chunk is."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
            
        # Remove stop words
        content_words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Calculate unique word ratio
        unique_words = len(set(content_words))
        total_content_words = len(content_words)
        
        if total_content_words == 0:
            return 0.0
            
        # Density score based on unique content words
        density = unique_words / max(total_content_words, 1)
        
        # Bonus for longer content words (more specific terms)
        avg_word_length = sum(len(w) for w in content_words) / max(len(content_words), 1)
        length_bonus = min(avg_word_length / 8.0, 1.0)  # Cap at 1.0
        
        return min(density + length_bonus * 0.2, 1.0)

    def rank_chunks_by_quality(self, chunks: List[str]) -> List[tuple]:
        """Rank chunks by their quality and information density."""
        ranked_chunks = []
        
        for chunk in chunks:
            cleaned = self.clean_chunk(chunk)
            if not cleaned:
                continue
                
            density = self.calculate_semantic_density(cleaned)
            key_phrases = self.extract_key_phrases(cleaned)
            
            # Quality score combines density and key phrase count
            quality_score = density + (len(key_phrases) * 0.1)
            
            ranked_chunks.append((cleaned, quality_score, key_phrases))
        
        # Sort by quality score (highest first)
        return sorted(ranked_chunks, key=lambda x: x[1], reverse=True)

    def preprocess_document_chunks(self, chunks: List[str], max_chunks: int = 5) -> List[str]:
        """
        Main preprocessing pipeline for document chunks.
        
        Args:
            chunks: Raw text chunks from document
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of cleaned and ranked chunks
        """
        # Rank chunks by quality
        ranked_chunks = self.rank_chunks_by_quality(chunks)
        
        # Filter out very low quality chunks (score < 0.3)
        quality_chunks = [chunk for chunk, score, _ in ranked_chunks if score >= 0.3]
        
        # Return top chunks
        return quality_chunks[:max_chunks]
