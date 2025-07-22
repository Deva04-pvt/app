# app/services/generator.py

import os
import json
import requests
from typing import List, Dict, Any

# --- Configuration ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")  # Use a powerful text model


def generate_structured_answer(
    context_chunks: List[str], question: str
) -> Dict[str, Any]:
    """
    Generates a structured answer from an LLM, ensuring it is grounded in the provided context and includes citations.
    """

    # --- 1. Structure the Context for the LLM ---
    # Enumerate chunks to make them distinct and referenceable.
    formatted_context = "\n\n".join(
        f"[[Citation: {i+1}]]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    # --- 2. Create the Advanced Prompt Template ---
    # This prompt forces the model to follow a strict reasoning and output process.
    prompt = f"""
You are a meticulous and impartial AI assistant for an insurance company. Your task is to answer questions based *exclusively* on the provided document excerpts.

**INSTRUCTIONS:**
1.  Carefully read the question and the following document excerpts (context).
2.  Locate the exact information in the context that answers the question.
3.  Do not use any external knowledge. Do not make assumptions or infer information not explicitly stated.
4.  You MUST cite the source of your answer by referencing the citation number (e.g., [[Citation: 1]]).
5.  Your final output must be a JSON object with the following schema:
    {{
        "answer": "Your detailed answer, synthesized from the context.",
        "is_answer_in_context": boolean, // true if you found the answer, false otherwise.
        "citations": [1, 2] // A list of integers referencing the citations you used. Empty if none.
    }}
6.  If the answer is not found in the context, the 'answer' field should state this, 'is_answer_in_context' must be false, and 'citations' must be empty.

**CONTEXT:**
{formatted_context}

**QUESTION:**
{question}

**FINAL JSON OUTPUT:**
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",  # Use Ollama's built-in JSON mode for reliability
        "stream": False,
    }

    try:
        response = requests.post(
            OLLAMA_URL, json=payload, timeout=120
        )  # Increased timeout
        response.raise_for_status()

        # The 'format=json' mode ensures the response 'content' is a valid JSON string.
        response_data = response.json()
        json_content_str = response_data.get("response", "{}")

        return json.loads(json_content_str)

    except requests.RequestException as e:
        raise RuntimeError(f"Ollama LLM generation failed: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to decode JSON response from Ollama: {e}\nResponse text: {response.text}"
        )
