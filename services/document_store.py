# app/services/document_store.py
import json
from typing import List, Dict


class DocumentStore:
    """A simple in-memory store for text chunks."""

    def __init__(self, chunks: List[str]):
        # The store maps a unique ID (the chunk's index) to the chunk text.
        self._store: Dict[int, str] = {i: chunk for i, chunk in enumerate(chunks)}

    def get_by_ids(self, ids: List[int]) -> List[str]:
        """Retrieves chunks from the store by their IDs."""
        return [self._store.get(id, "") for id in ids]

    def get_all_chunks(self) -> List[str]:
        """Returns all chunks in the store."""
        return list(self._store.values())

    def save(self, filepath: str):
        """Saves the document store to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._store, f)

    @classmethod
    def load(cls, filepath: str):
        """Loads a document store from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # JSON keys are strings, so convert them back to integers
            store_instance = cls(chunks=[])
            store_instance._store = {int(k): v for k, v in data.items()}
            return store_instance
# app/services/document_store.py
import json
from typing import List, Dict


class DocumentStore:
    """A simple in-memory store for text chunks."""

    def __init__(self, chunks: List[str]):
        # The store maps a unique ID (the chunk's index) to the chunk text.
        self._store: Dict[int, str] = {i: chunk for i, chunk in enumerate(chunks)}

    def get_by_ids(self, ids: List[int]) -> List[str]:
        """Retrieves chunks from the store by their IDs."""
        return [self._store.get(id, "") for id in ids]

    def get_all_chunks(self) -> List[str]:
        """Returns all chunks in the store."""
        return list(self._store.values())

    def save(self, filepath: str):
        """Saves the document store to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._store, f)

    @classmethod
    def load(cls, filepath: str):
        """Loads a document store from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # JSON keys are strings, so convert them back to integers
            store_instance = cls(chunks=[])
            store_instance._store = {int(k): v for k, v in data.items()}
            return store_instance
