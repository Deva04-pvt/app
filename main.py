# main.py

import os
import uuid
import shutil
import tempfile
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool

# Import our refined services
from services import downloader, extractor, chunker, embedder
from services import faiss_indexer, retriever, answer_generator
from services.document_store import DocumentStore
from schemas import RunRequest, RunResponse

# --- Application Setup ---
app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="An advanced RAG pipeline for processing documents and answering questions.",
    version="1.0.0",
)

# --- Security ---
# Placeholder for bearer token validation
auth_scheme = HTTPBearer()
BEARER_TOKEN = os.getenv(
    "API_BEARER_TOKEN", "your_secret_token_here"
)  # Use an environment variable for the token


def validate_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    """Validates the bearer token provided in the request header."""
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing bearer token")


# --- Main Endpoint ---
@app.post("/hackrx/run", response_model=RunResponse)
async def run_full_pipeline(request: RunRequest):
    """
    Orchestrates the entire RAG pipeline from document download to answer generation.
    """
    # Create a unique temporary directory for this specific run
    run_temp_dir = tempfile.mkdtemp(prefix="rag_run_")

    try:
        # --- PHASE 1: INGESTION AND CHUNKING ---
        print("Starting Phase 1: Ingestion and Chunking")
        downloaded_path = await run_in_threadpool(
            downloader.download_document, str(request.documents)
        )

        # Move downloaded file to our temporary directory
        temp_doc_path = os.path.join(run_temp_dir, os.path.basename(downloaded_path))
        shutil.move(downloaded_path, temp_doc_path)

        full_text = await run_in_threadpool(extractor.extract_text, temp_doc_path)
        if not full_text.strip():
            raise HTTPException(
                status_code=400, detail="Failed to extract text from the document."
            )

        text_chunks = await run_in_threadpool(chunker.split_text, full_text)

        # --- PHASE 2: INDEXING ---
        print("Starting Phase 2: Indexing")
        doc_store = DocumentStore(chunks=text_chunks)

        # The embedder is already parallelized internally
        embeddings = await run_in_threadpool(embedder.get_embeddings, text_chunks)
        if not embeddings:
            raise HTTPException(
                status_code=500, detail="Failed to generate embeddings."
            )

        # Save the index inside the temporary directory
        temp_index_path = os.path.join(run_temp_dir, "document.index")
        await run_in_threadpool(
            faiss_indexer.build_and_save_index, embeddings, temp_index_path
        )

        # --- PHASE 3: RETRIEVAL AND GENERATION ---
        print("Starting Phase 3: Retrieval and Generation")
        final_answers = []
        index = faiss_indexer.load_index(temp_index_path)

        for question in request.questions:
            print(f"Processing question: '{question[:50]}...'")
            # 1. Retrieve relevant context
            retrieved_chunks_with_scores = await run_in_threadpool(
                retriever.search_index, index, doc_store, question
            )
            context_chunks = [chunk for chunk, score in retrieved_chunks_with_scores]

            # 2. Generate the answer from the context
            if not context_chunks:
                # If no context is found, use a default non-answer
                structured_response = {"answer": "Not mentioned in the document."}
            else:
                structured_response = await run_in_threadpool(
                    answer_generator.generate_structured_answer, context_chunks, question
                )

            final_answers.append(
                structured_response.get("answer", "Failed to generate a valid answer.")
            )

        return RunResponse(answers=final_answers)

    except Exception as e:
        # Catch any exception from the pipeline and return a server error
        print(f"An error occurred during the pipeline execution: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        # --- CLEANUP ---
        # Ensure the temporary directory is always removed
        print(f"Cleaning up temporary directory: {run_temp_dir}")
        shutil.rmtree(run_temp_dir)


# --- To run the application ---
# 1. Save this file as `main.py` in your project's root.
# 2. Make sure all service files are in the `app/services/` directory.
# 3. Run from your terminal: uvicorn main:app --reload
