# app/services/answer_generator.py

import os
from google import generativeai as genai

# It's better to get the API key from environment variables
# rather than hardcoding it.
genai.configure(api_key="AIzaSyBNeeAtc91tx67QxTvmYlCadC--4ZE8i4s")

def generate_answer(context_chunks: list[str], question: str) -> str:
    """
    Generates an answer to a question based on provided context chunks.

    Args:
        context_chunks: A list of text strings providing context.
        question: The question to be answered.

    Returns:
        The generated answer as a string.
    """
    # Join the context chunks into a single block of text
    context = "\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant who answers questions based ONLY on the provided context.

Context:
---
{context}
---

Question: {question}

Instructions:
1.  Read the context carefully.
2.  Formulate a clear and concise answer based strictly on the information given in the context.
3.  If the information needed to answer the question is not in the context, you must respond with exactly this phrase: "The answer is not available in the provided context."
"""

    # Initialize the generative model
    model = genai.GenerativeModel('gemini-1.5-flash') # Using a fast and capable model

    # Generate the content
    response = model.generate_content(prompt)

    return response.text.strip()