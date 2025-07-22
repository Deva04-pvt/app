# app/services/answer_generator.py

import os
from typing import List
from dotenv import load_dotenv
import openai

load_dotenv()

# Configure the API key and optionally a custom base URL (for proxy or Azure OpenAI)
openai.api_key = os.getenv("OPENAI_API_KEY")
if os.getenv("OPENAI_API_BASE"):
    openai.api_base = os.getenv("OPENAI_API_BASE")

def generate_answer(context_chunks: List[str], question: str) -> str:
    """
    Generates an answer to a question based on provided context chunks using OpenAI's GPT model.

    Args:
        context_chunks: A list of text strings providing context.
        question: The question to be answered.

    Returns:
        The generated answer as a string.
    """
    context = "\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who answers questions based ONLY on the provided context. "
                "If the answer is not found in the context, respond with exactly this phrase: "
                "\"The answer is not available in the provided context.\""
            )
        },
        {
            "role": "user",
            "content": f"""
Context:
---
{context}
---

Question: {question}
"""
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" if cost/speed is a concern
            messages=messages,
            temperature=0  # deterministic, factual output
        )
        return response["choices"][0]["message"]["content"].strip()

    except openai.error.OpenAIError as e:
        raise RuntimeError(f"Failed to generate answer: {str(e)}")
