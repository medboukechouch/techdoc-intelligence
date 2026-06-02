"""Tests for ingestion package."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestion import ingest_pdf
from ingestion.chunker import chunk_text
from ingestion.parser import parse_pdf

def test_chunk_text_normal_case():
    text = " ".join([f"w{i}" for i in range(120)])
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) >= 2
    assert all(len(c.split()) >= 10 for c in chunks)

def test_chunk_text_overlap():
    text = " ".join([f"w{i}" for i in range(30)])
    chunks = chunk_text(text, chunk_size=15, overlap=5)
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]

def test_chunk_text_filters_short_chunks():
    text = " ".join([f"w{i}" for i in range(9)])
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert chunks == []

def test_parse_pdf_missing_file_raises():
    with pytest.raises(ValueError):
        parse_pdf("does_not_exist.pdf")

def test_ingest_pdf_calls_upsert(monkeypatch):
    monkeypatch.setattr("ingestion.parse_pdf", lambda p: "word " * 40)
    monkeypatch.setattr("ingestion.chunk_text", lambda t: ["a b c d e f g h i j"])

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 384]
    monkeypatch.setattr("ingestion.Embedder", lambda: mock_embedder)

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    monkeypatch.setattr("ingestion.QdrantClient", lambda host, port: mock_client)

    ingest_pdf("sample.pdf", "test_collection", "localhost", 6333)

    assert mock_client.upsert.called
