"""Graph nodes for agent reasoning."""
from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient

from agent.state import AgentState
from ingestion.embedder import Embedder

load_dotenv()

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-flash latest")

def router_node(state: AgentState) -> dict:
    """Classify question type."""
    llm = _get_llm()
    system = (
        "You are a classifier. Return only one label: factual, comparative, or summary."
    )
    messages = [SystemMessage(content=system), HumanMessage(content=state["question"])]
    response = llm.invoke(messages)
    label = response.content.strip().lower()
    if label not in {"factual", "comparative", "summary"}:
        label = "factual"
    return {"question_type": label}

def retrieval_node(state: AgentState) -> dict:
    """Retrieve chunks from Qdrant."""
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("COLLECTION_NAME", "techdocs")

    embedder = Embedder()
    vector = embedder.embed([state["question"]])[0]

    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    results = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=5,
        with_payload=True,
    )

    chunks: list[str] = []
    scores: list[float] = []
    sources: list[str] = []
    for result in results:
        payload = result.payload or {}
        chunks.append(payload.get("text", ""))
        scores.append(float(result.score or 0.0))
        sources.append(payload.get("source", ""))

    return {
        "retrieved_chunks": chunks,
        "retrieval_scores": scores,
        "sources": sources,
    }

def validation_node(state: AgentState) -> dict:
    """Validate retrieval quality and optionally reformulate question."""
    good_chunks = sum(score > 0.4 for score in state.get("retrieval_scores", []))
    retry_count = state.get("retry_count", 0)

    if good_chunks >= 3 or retry_count >= 2:
        return {"retry_count": retry_count, "question": state["question"]}

    llm = _get_llm()
    system = "Rephrase the question to improve retrieval while keeping meaning."
    messages = [SystemMessage(content=system), HumanMessage(content=state["question"])]
    response = llm.invoke(messages)
    new_question = response.content.strip()
    return {"retry_count": retry_count + 1, "question": new_question}

def answer_node(state: AgentState) -> dict:
    """Generate answer from retrieved context."""
    llm = _get_llm()
    context = "\n\n".join(
        f"[Source: {src}] {chunk}"
        for src, chunk in zip(state.get("sources", []), state.get("retrieved_chunks", []))
    )
    prompt = (
        "Answer the question using only the context below. "
        "Cite sources in-line like [source].\n\n"
        f"Question: {state['question']}\n\n"
        f"Context:\n{context}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content.strip()}

def critique_node(state: AgentState) -> dict:
    """Critique answer and produce confidence score."""
    llm = _get_llm()
    prompt = (
        "Is this answer fully supported by the context? "
        "Rate confidence 0-1 and explain.\n\n"
        f"Question: {state['question']}\n"
        f"Answer: {state.get('answer', '')}\n\n"
        f"Context:\n{state.get('retrieved_chunks', [])}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    confidence = float(match.group(1)) if match else 0.0
    return {"critique": text, "confidence": confidence}

def should_continue(state: AgentState) -> str:
    good_chunks = sum(score > 0.4 for score in state.get("retrieval_scores", []))
    if good_chunks >= 3 or state.get("retry_count", 0) >= 2:
        return "continue"
    return "retry"
