# app/main.py
import os
import uvicorn
from fastapi import FastAPI
from routes import router

app = FastAPI(title="Document Downloader Service")

app.include_router(router)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Document Processing Service is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "Document Processing API"}

if __name__ == "__main__":
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Important: bind to all interfaces
        port=port,
        reload=False  # Set to False for production
    )
