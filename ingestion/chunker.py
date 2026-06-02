"""Chunking utilities for documents."""
from __future__ import annotations

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if len(chunk_words) < 10:
            continue
        chunks.append(" ".join(chunk_words))

    return chunks
