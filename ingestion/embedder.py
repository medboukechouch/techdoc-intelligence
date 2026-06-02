"""Embedder utilities to create vector embeddings."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

class Embedder:
    """Sentence-transformers embedder wrapper."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for texts."""
        if not texts:
            return []
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]
