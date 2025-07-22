import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision"


def generate_answer(context_chunks: list[str], question: str) -> str:
    prompt = f"""
You are a domain expert tasked with answering questions using the provided document context only.

Context:
\"\"\"
{chr(10).join(context_chunks)}
\"\"\"

Question:
{question}

Instructions:
- Provide a clear, concise answer based strictly on the context.
- If the answer is not present in the context, respond with: "Not mentioned in the document."
"""

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.RequestException as e:
        raise RuntimeError(f"Ollama LLM generation failed: {e}")
