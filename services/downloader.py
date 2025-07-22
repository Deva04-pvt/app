# app/services/downloader.py

import os
import uuid
import requests
import mimetypes
from urllib.parse import urlparse

# --- Configuration ---
# Use environment variables for configuration with sensible defaults.
# This makes the app portable and easy to configure in different environments.
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "/tmp/downloads")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))  # Increased default timeout
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".eml",
    ".txt",
}  # Added .eml for emails, .txt as a common case


# --- Custom Exception ---
# Define a specific exception for more granular error handling upstream.
class DownloadError(Exception):
    """Custom exception for download failures."""

    pass


# --- Initialization ---
# Ensure the download directory exists.
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _get_file_extension(response: requests.Response, url: str) -> str:
    """Robustly determines the file extension."""
    # 1. Prioritize Content-Type header
    content_type = response.headers.get("content-type")
    if content_type:
        # Handles cases like "application/pdf; charset=utf-8"
        mime_type = content_type.split(";")[0].strip()
        extension = mimetypes.guess_extension(mime_type)
        if extension and extension in ALLOWED_EXTENSIONS:
            return extension

    # 2. Fallback to parsing the URL path
    path = urlparse(url).path
    _, extension = os.path.splitext(path)
    if extension and extension.lower() in ALLOWED_EXTENSIONS:
        return extension.lower()

    # 3. Default to a generic binary extension if no match found
    return ".bin"


def download_document(url: str) -> str:
    """
    Downloads a document from a URL, validates its type, and saves it locally.

    Args:
        url: The URL of the document to download.

    Returns:
        The local filepath of the downloaded document.

    Raises:
        DownloadError: If the download fails or the file type is not allowed.
    """
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()

            extension = _get_file_extension(response, url)

            if extension not in ALLOWED_EXTENSIONS:
                raise DownloadError(
                    f"File type '{extension}' is not supported. URL: {url}"
                )

            filename = f"{uuid.uuid4()}{extension}"
            filepath = os.path.join(DOWNLOAD_DIR, filename)

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return filepath

    except requests.exceptions.RequestException as e:
        # Wrap the original exception in our custom error
        raise DownloadError(f"Download failed for URL {url}: {e}") from e
