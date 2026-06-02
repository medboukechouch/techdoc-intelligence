"""Tests for agent package."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.graph import graph
from agent.nodes import router_node, validation_node

def test_router_node_classifies_question(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "factual"
    monkeypatch.setattr("agent.nodes.ChatGoogleGenerativeAI", lambda model: mock_llm)

    state = {"question": "What is Qdrant?"}
    result = router_node(state)
    assert result["question_type"] in {"factual", "comparative", "summary"}

def test_validation_node_continue(monkeypatch):
    state = {
        "question": "test",
        "retrieval_scores": [0.5, 0.6, 0.45],
        "retry_count": 0,
    }
    result = validation_node(state)
    assert result["retry_count"] == 0
    assert result["question"] == "test"

def test_validation_node_retry(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "rephrased"
    monkeypatch.setattr("agent.nodes.ChatGoogleGenerativeAI", lambda model: mock_llm)

    state = {
        "question": "test",
        "retrieval_scores": [0.1, 0.2, 0.3],
        "retry_count": 0,
    }
    result = validation_node(state)
    assert result["retry_count"] == 1
    assert result["question"] == "rephrased"

def test_graph_compiles():
    assert graph is not None
