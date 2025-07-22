# app/services/chunker.py

import re
from typing import List
def split_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """
    Naive chunking by sentences. Real production setup should use token-aware logic (tiktoken/Tokenizer).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_length:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence

    if chunk:
        chunks.append(chunk.strip())

    # Add overlap between chunks
    final_chunks = []
    for i in range(0, len(chunks)):
        chunk = chunks[i]
        prev = chunks[i - 1][-overlap:] if i > 0 else ""
        final_chunks.append(prev + " " + chunk)

    return final_chunks
