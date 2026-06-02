"""Tests for MCP server tools."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_server import server

def test_search_docs_returns_expected_format(monkeypatch):
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 384]
    monkeypatch.setattr(server, "Embedder", lambda: mock_embedder)

    mock_client = MagicMock()
    mock_client.search.return_value = [
        SimpleNamespace(payload={"text": "chunk", "source": "file.pdf"}, score=0.9)
    ]
    monkeypatch.setattr(server, "QdrantClient", lambda host, port: mock_client)

    results = server.search_docs("query", top_k=1)
    assert results == [{"text": "chunk", "source": "file.pdf", "score": 0.9}]

def test_add_document_success(monkeypatch):
    monkeypatch.setattr(server, "ingest_pdf", lambda **kwargs: 3)
    monkeypatch.setattr(server.Path, "exists", lambda self: True)

    result = server.add_document("/tmp/sample.pdf")
    assert result["status"] == "success"
    assert result["chunks_ingested"] == 3
    assert result["source"] == "sample.pdf"

def test_add_document_missing_file_raises(monkeypatch):
    monkeypatch.setattr(server.Path, "exists", lambda self: False)
    with pytest.raises(server.ToolError):
        server.add_document("/tmp/missing.pdf")