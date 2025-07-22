# app/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from services.downloader import download_document
from services.extractor import extract_text
from services.chunker import split_text
from services.smart_chunker import smart_split_text
from services.embedder import get_embeddings_gemini
from services.faiss_indexer import build_faiss_index
from services.retriever import search_faiss_index
from services.answer_generator import generate_answer
from config import get_chunking_config, get_retrieval_config
from typing import List
import os
router = APIRouter()

# Get configuration
chunking_config = get_chunking_config()
retrieval_config = get_retrieval_config()

class DocumentRequest(BaseModel):
    document_url: HttpUrl

class ExtractRequest(BaseModel):
    filepath: str

class QueryRequest(BaseModel):
    filepath: str
    question: str
    top_k: int = None  # Will use config default if None

class AnswerRequest(BaseModel):
    filepath: str
    question: str
    top_k: int = None  # Will use config default if None

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    question: str
    answer: str

@router.post("/download")
async def download_endpoint(payload: DocumentRequest):
    try:
        path = download_document(payload.document_url)
        return {"status": "success", "filepath": path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/extract")
async def extract_endpoint(payload: ExtractRequest):
    try:
        raw_text = extract_text(payload.filepath)
        return {
            "status": "success",
            "length": len(raw_text),
            "text": raw_text
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/vectorize")
async def vectorize_document(payload: ExtractRequest):
    try:
        text = extract_text(payload.filepath)
        # Use smart chunking for better quality chunks
        if chunking_config["use_smart_chunking"]:
            chunks = smart_split_text(
                text, 
                max_tokens=chunking_config["max_tokens"], 
                overlap_tokens=chunking_config["overlap_tokens"]
            )
        else:
            chunks = split_text(text, max_length=chunking_config["max_tokens"])
        
        embeddings = get_embeddings_gemini(chunks)
        index = build_faiss_index(embeddings)
        return {
            "status": "success",
            "chunks": len(chunks),
            "dimension": len(embeddings[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/query")
async def query_endpoint(payload: QueryRequest):
    try:
        # Get top_k from payload or use config default
        top_k = payload.top_k or retrieval_config["default_top_k"]
        top_k = min(top_k, retrieval_config["max_top_k"])  # Cap at max allowed
        
        # Step 1: Extract and chunk document with smart chunking
        text = extract_text(payload.filepath)
        if chunking_config["use_smart_chunking"]:
            chunks = smart_split_text(
                text, 
                max_tokens=chunking_config["max_tokens"], 
                overlap_tokens=chunking_config["overlap_tokens"]
            )
        else:
            chunks = split_text(text, max_length=chunking_config["max_tokens"])

        # Step 2: Embed document
        embeddings = get_embeddings_gemini(chunks)
        index = build_faiss_index(embeddings)

        # Step 3: Search for top-k chunks
        results = search_faiss_index(index, payload.question, chunks, top_k)

        return {
            "status": "success",
            "matches": [
                {
                    "score": round(score, 4),
                    "text": chunk
                } for chunk, score in results
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer")
async def answer_endpoint(payload: AnswerRequest):
    try:
        # Get top_k from payload or use config default
        top_k = payload.top_k or retrieval_config["default_top_k"]
        top_k = min(top_k, retrieval_config["max_top_k"])  # Cap at max allowed
        
        # Extract and chunk with smart chunking
        text = extract_text(payload.filepath)
        if chunking_config["use_smart_chunking"]:
            chunks = smart_split_text(
                text, 
                max_tokens=chunking_config["max_tokens"], 
                overlap_tokens=chunking_config["overlap_tokens"]
            )
        else:
            chunks = split_text(text, max_length=chunking_config["max_tokens"])

        # Embed and index
        embeddings = get_embeddings_gemini(chunks)
        index = build_faiss_index(embeddings)

        # Retrieve top-k
        top_chunks = search_faiss_index(index, payload.question, chunks, top_k)

        # Generate answer from top chunks (now with advanced preprocessing)
        context = [chunk for chunk, _ in top_chunks]
        answer = generate_answer(context, payload.question)

        return {
            "status": "success",
            "answer": answer,
            "context_used": context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BulkQARequest(BaseModel):
    documents: str  # Blob URL
    questions: List[str]
    top_k: int = None  # Will use config default if None

@router.post("/api/v1/hackrx/run")
async def bulk_answer_endpoint(payload: BulkQARequest):
    try:
        # Get top_k from payload or use config default
        top_k = payload.top_k or retrieval_config["default_top_k"]
        top_k = min(top_k, retrieval_config["max_top_k"])  # Cap at max allowed
        
        local_path = download_document(payload.documents)
        full_text = extract_text(local_path)
        
        # Use smart chunking for better quality
        if chunking_config["use_smart_chunking"]:
            chunks = smart_split_text(
                full_text, 
                max_tokens=chunking_config["max_tokens"], 
                overlap_tokens=chunking_config["overlap_tokens"]
            )
        else:
            chunks = split_text(full_text, max_length=chunking_config["max_tokens"])
            
        embeddings = get_embeddings_gemini(chunks)
        index = build_faiss_index(embeddings)
        answers = []
        for question in payload.questions:
            top_chunks = search_faiss_index(index, question, chunks, top_k)
            context = [chunk for chunk, _ in top_chunks]
            # Now uses improved answer generation with preprocessing
            answer = generate_answer(context, question)
            answers.append(answer)

        os.remove(local_path)

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))