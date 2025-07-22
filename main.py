# app/main.py
from fastapi import FastAPI
from routes import router

app = FastAPI(title="Document Downloader Service")

app.include_router(router, prefix="/api/v1")
