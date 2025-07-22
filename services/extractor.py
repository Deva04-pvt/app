# app/services/extractor.py

import os
import mimetypes

from pdfminer.high_level import extract_text as extract_pdf
from docx import Document
import email

def extract_text_from_pdf(filepath: str) -> str:
    return extract_pdf(filepath)

def extract_text_from_docx(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_eml(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        msg = email.message_from_file(f)

    body = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            try:
                body.append(part.get_payload(decode=True).decode("utf-8"))
            except Exception:
                pass
    return "\n".join(body)

def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".eml":
        return extract_text_from_eml(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
