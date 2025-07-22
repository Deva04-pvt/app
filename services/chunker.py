# app/services/chunker.py
#
# Required libraries:
# pip install langchain tiktoken

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# --- Tokenizer Setup ---
# Use the tokenizer that corresponds to modern OpenAI models (e.g., gpt-4, text-embedding-ada-002)
# This ensures our chunk size is measured exactly as the model sees it.
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    """A helper function to calculate the number of tokens in a string."""
    return len(TOKENIZER.encode(text))


def split_text(
    text: str, max_tokens: int = 512, overlap_tokens: int = 100
) -> List[str]:
    """
    Splits text into chunks using a recursive, token-aware strategy.

    Args:
        text: The input text to be split.
        max_tokens: The maximum number of tokens allowed in a chunk.
        overlap_tokens: The number of tokens to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    # --- Text Splitter Initialization ---
    # This splitter respects semantic boundaries and uses token count for length.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=token_length,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],  # Prioritized separators
    )

    return text_splitter.split_text(text)
