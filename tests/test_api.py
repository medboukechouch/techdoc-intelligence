"""Tests for API package."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from api.main import app

@pytest.mark.asyncio
async def test_health_connected(monkeypatch):
    mock_client = MagicMock()
    mock_client.get_collections.return_value = {"collections": []}
    monkeypatch.setattr("api.main.QdrantClient", lambda host, port: mock_client)

    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "qdrant": "connected"}

@pytest.mark.asyncio
async def test_ingest_success(monkeypatch):
    monkeypatch.setattr("api.main.ingest_pdf", lambda **kwargs: 2)

    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/ingest", json={"pdf_path": "/tmp/sample.pdf"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["chunks_ingested"] == 2
    assert body["source"] == "sample.pdf"

@pytest.mark.asyncio
async def test_query_sse_stream(monkeypatch):
    async def fake_events(*args, **kwargs):
        yield {"event": "on_node_start", "name": "router_node"}
        yield {"event": "on_node_start", "name": "retrieval_node"}
        yield {"event": "on_node_end", "name": "answer_node", "data": {"output": {"answer": "ok"}}}
        yield {"event": "on_node_end", "name": "critique_node", "data": {"output": {"confidence": 0.9}}}
        yield {
            "event": "on_chain_end",
            "data": {"output": {"answer": "ok", "sources": ["s1"], "confidence": 0.9, "critique": "fine", "question_type": "factual"}},
        }

    monkeypatch.setattr("api.main.graph", SimpleNamespace(astream_events=fake_events))

    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream("POST", "/query", json={"question": "hi"}) as resp:
            body = await resp.aread()

    text = body.decode()
    assert "router_node" in text
    assert "END" in text

@pytest.mark.asyncio
async def test_documents(monkeypatch):
    mock_client = MagicMock()
    mock_client.scroll.side_effect = [
        ([SimpleNamespace(payload={"source": "a.pdf"})], "next"),
        ([SimpleNamespace(payload={"source": "b.pdf"})], None),
    ]
    monkeypatch.setattr("api.main.QdrantClient", lambda host, port: mock_client)

    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.get("/documents")

    assert resp.status_code == 200
    assert resp.json() == {"documents": ["a.pdf", "b.pdf"]}
