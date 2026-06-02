"""FastAPI main app entrypoint."""
from __future__ import annotations

import json
import os
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

from agent.graph import graph
from ingestion import ingest_pdf

load_dotenv()

app = FastAPI(title="TechDoc Intelligence API")

class QueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    critique: str
    question_type: str

class IngestRequest(BaseModel):
    pdf_path: str

class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int
    source: str

def _qdrant_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)

def _collection_name(override: Optional[str] = None) -> str:
    return override or os.getenv("COLLECTION_NAME", "techdocs")

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

@app.get("/health")
def health():
    try:
        client = _qdrant_client()
        client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception:
        return {"status": "error", "qdrant": "unreachable"}

@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    collection_name = _collection_name()
    chunks_ingested = ingest_pdf(
        pdf_path=request.pdf_path,
        collection_name=collection_name,
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
    )
    source = os.path.basename(request.pdf_path)
    return IngestResponse(
        status="success",
        chunks_ingested=chunks_ingested,
        source=source,
    )

@app.post("/query")
def query(request: QueryRequest) -> StreamingResponse:
    if request.collection_name:
        os.environ["COLLECTION_NAME"] = request.collection_name

    initial_state = {
        "question": request.question,
        "question_type": "",
        "retrieved_chunks": [],
        "retrieval_scores": [],
        "answer": "",
        "critique": "",
        "confidence": 0.0,
        "retry_count": 0,
        "sources": [],
    }

    async def event_stream() -> AsyncGenerator[str, None]:
        final_state = None
        async for event in graph.astream_events(initial_state, version="v1"):
            event_type = event.get("event")
            node_name = event.get("name") or event.get("metadata", {}).get("node")

            if event_type == "on_node_start" and node_name:
                yield _sse({"node": node_name, "status": "running"})

            if event_type == "on_node_end" and node_name:
                output = event.get("data", {}).get("output")
                if node_name == "answer_node" and isinstance(output, dict):
                    if "answer" in output:
                        yield _sse({"node": "answer_node", "output": output["answer"]})
                if node_name == "critique_node" and isinstance(output, dict):
                    if "confidence" in output:
                        yield _sse(
                            {"node": "critique_node", "output": output["confidence"]}
                        )

            if event_type == "on_chain_end":
                final_state = event.get("data", {}).get("output")

        if final_state is None:
            final_state = initial_state

        response = QueryResponse(
            answer=final_state.get("answer", ""),
            sources=final_state.get("sources", []),
            confidence=float(final_state.get("confidence", 0.0)),
            critique=final_state.get("critique", ""),
            question_type=final_state.get("question_type", ""),
        )
        yield _sse({"node": "END", "result": response.model_dump()})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/documents")
def documents():
    client = _qdrant_client()
    collection_name = _collection_name()
    sources: set[str] = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=next_offset,
        )
        for point in points:
            payload = point.payload or {}
            source = payload.get("source")
            if source:
                sources.add(source)
        if next_offset is None:
            break

    return {"documents": sorted(sources)}
