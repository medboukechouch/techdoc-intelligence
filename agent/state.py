"""Agent state management."""
from __future__ import annotations

from typing import TypedDict

class AgentState(TypedDict):
    """State definition for the LangGraph agent."""
    question: str
    question_type: str
    retrieved_chunks: list[str]
    retrieval_scores: list[float]
    answer: str
    critique: str
    confidence: float
    retry_count: int
    sources: list[str]
