"""FastMCP server for TechDoc Intelligence."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP, ToolError
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from datasets import Dataset

from ingestion import ingest_pdf
from ingestion.embedder import Embedder

load_dotenv()

SERVER_NAME = "techdoc-mcp"
mcp = FastMCP(SERVER_NAME)

def _qdrant_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)

def _collection_name() -> str:
    return os.getenv("COLLECTION_NAME", "techdocs")

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-flash latest")

def _metric_value(result: Any, key: str) -> float:
    try:
        value = result[key]
    except Exception:
        if hasattr(result, "to_pandas"):
            value = result.to_pandas()[key].iloc[0]
        else:
            raise
    if isinstance(value, list):
        value = value[0]
    return float(value)

@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """Search documents in Qdrant and return chunks with scores."""
    embedder = Embedder()
    vector = embedder.embed([query])[0]

    client = _qdrant_client()
    results = client.search(
        collection_name=_collection_name(),
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )

    payloads: list[dict] = []
    for result in results:
        payload = result.payload or {}
        payloads.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "score": float(result.score or 0.0),
            }
        )
    return payloads

@mcp.tool()
def add_document(pdf_path: str) -> dict:
    """Ingest a PDF into the vector store."""
    path = Path(pdf_path)
    if not path.exists():
        raise ToolError("File does not exist")

    chunks_ingested = ingest_pdf(
        pdf_path=pdf_path,
        collection_name=_collection_name(),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
    )
    return {
        "status": "success",
        "chunks_ingested": chunks_ingested,
        "source": path.name,
    }

@mcp.tool()
def get_document_summary(source_filename: str) -> dict:
    """Summarize a document from stored chunks."""
    client = _qdrant_client()
    collection_name = _collection_name()
    scroll_filter = Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=source_filename))]
    )

    chunks: list[str] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=100,
            offset=next_offset,
            with_payload=True,
        )
        for point in points:
            payload = point.payload or {}
            text = payload.get("text")
            if text:
                chunks.append(text)
        if next_offset is None:
            break

    if not chunks:
        raise ToolError("No chunks found for the given source")

    llm = _get_llm()
    prompt = "Summarize this technical document in 5 bullet points\n\n" + "\n\n".join(chunks)
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"source": source_filename, "summary": response.content.strip()}

@mcp.tool()
def evaluate_last_answer(question: str, answer: str, contexts: list[str]) -> dict:
    """Evaluate a single QA pair using RAGAS metrics."""
    dataset = Dataset.from_dict(
        {"question": [question], "answer": [answer], "contexts": [contexts]}
    )
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    return {
        "faithfulness": _metric_value(results, "faithfulness"),
        "answer_relevancy": _metric_value(results, "answer_relevancy"),
    }

def create_server() -> FastMCP:
    """Return configured FastMCP server."""
    return mcp
