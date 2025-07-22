# app/services/downloader.py

import os
import uuid
import requests
import mimetypes

DOWNLOAD_DIR = "/tmp/downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_document(url: str) -> str:
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type.split(";")[0]) or ".bin"
        filename = f"{uuid.uuid4()}{extension}"
        filepath = os.path.join(DOWNLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return filepath

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed: {e}")
