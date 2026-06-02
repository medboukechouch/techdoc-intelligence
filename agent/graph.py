"""Agent reasoning graph utilities."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.nodes import (
    answer_node,
    critique_node,
    retrieval_node,
    router_node,
    should_continue,
    validation_node,
)
from agent.state import AgentState

workflow = StateGraph(AgentState)
workflow.add_node("router_node", router_node)
workflow.add_node("retrieval_node", retrieval_node)
workflow.add_node("validation_node", validation_node)
workflow.add_node("answer_node", answer_node)
workflow.add_node("critique_node", critique_node)

workflow.set_entry_point("router_node")
workflow.add_edge("router_node", "retrieval_node")
workflow.add_edge("retrieval_node", "validation_node")
workflow.add_conditional_edges(
    "validation_node",
    should_continue,
    {
        "continue": "answer_node",
        "retry": "retrieval_node",
    },
)
workflow.add_edge("answer_node", "critique_node")
workflow.add_edge("critique_node", END)

graph = workflow.compile()
