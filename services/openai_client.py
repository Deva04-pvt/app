# app/services/openai_client.py

import os
import openai
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    """
    OpenAI client configured with custom base URL and API key.
    """
    
    def __init__(self):
        # Configure OpenAI with custom base URL
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_base = os.getenv('OPENAI_BASE_URL')
        
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if not openai.api_base:
            raise ValueError("OPENAI_BASE_URL environment variable not set")
    
    def get_embeddings(self, texts, model="text-embedding-ada-002"):
        """
        Get embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of strings to embed
            model: Embedding model to use
            
        Returns:
            List of embedding vectors
        """
        try:
            response = openai.Embedding.create(
                input=texts,
                model=model
            )
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            print(f"Error getting OpenAI embeddings: {e}")
            raise
    
    def generate_completion(self, prompt, model="gpt-3.5-turbo", max_tokens=500, temperature=0.7):
        """
        Generate text completion using OpenAI.
        
        Args:
            prompt: Input prompt
            model: Model to use for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating OpenAI completion: {e}")
            raise
    
    def generate_answer_with_context(self, context_chunks, question, model="gpt-3.5-turbo"):
        """
        Generate an answer using OpenAI with the same logic as the Gemini version.
        
        Args:
            context_chunks: List of context text chunks
            question: User question
            model: OpenAI model to use
            
        Returns:
            Generated answer
        """
        # Join the context chunks
        context = "\n\n".join(context_chunks)
        
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
        
        return self.generate_completion(prompt, model=model, temperature=0.3)

# Convenience function to get a configured client
def get_openai_client():
    """Get a configured OpenAI client."""
    return OpenAIClient()
