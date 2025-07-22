# app/schemas.py

from pydantic import BaseModel, HttpUrl
from typing import List


class RunRequest(BaseModel):
    """Defines the structure of the incoming request payload."""

    documents: HttpUrl  # Pydantic validates that this is a valid URL
    questions: List[str]


class RunResponse(BaseModel):
    """Defines the structure of the final JSON response."""

    answers: List[str]

class DirectRunRequest(BaseModel):
    filepath: str
    questions: List[str]
