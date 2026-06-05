"""Tests for API package."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app

from types import SimpleNamespace # Assure-toi que cet import est bien en haut de ton fichier si ce n'est pas déjà le cas

@pytest.mark.asyncio
async def test_query_sse_stream(monkeypatch):
    async def fake_stream(*args, **kwargs):
        # On simule le nouveau format de réponse de graph.astream()
        yield {"router_node": {"question_type": "factual"}}
        yield {"retrieval_node": {"retrieved_chunks": ["chunk1"], "sources": ["s1"]}}
        yield {"answer_node": {"answer": "ok"}}
        yield {"critique_node": {"confidence": 0.9, "critique": "fine"}}

    # LA CORRECTION EST ICI : on mock 'astream' et non plus 'astream_events'
    monkeypatch.setattr("api.main.graph", SimpleNamespace(astream=fake_stream))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
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

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/documents")

    assert resp.status_code == 200
    assert resp.json() == {"documents": ["a.pdf", "b.pdf"]}
