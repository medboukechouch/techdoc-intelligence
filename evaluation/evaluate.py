"""Evaluation scripts for retrieval and QA."""
from __future__ import annotations

import json
import os

from qdrant_client import QdrantClient

from agent.graph import graph
from ingestion.embedder import Embedder

EVAL_DATASET = [
    {
        "question": "What is the purpose of this system?",
        "ground_truth": "placeholder answer 1",
    },
    {
        "question": "How are documents ingested?",
        "ground_truth": "placeholder answer 2",
    },
    {
        "question": "What vector database is used?",
        "ground_truth": "placeholder answer 3",
    },
    {
        "question": "How does the agent validate retrieval?",
        "ground_truth": "placeholder answer 4",
    },
    {
        "question": "What model is used for embeddings?",
        "ground_truth": "placeholder answer 5",
    },
]

def _qdrant_client(qdrant_host: str, qdrant_port: int) -> QdrantClient:
    return QdrantClient(host=qdrant_host, port=qdrant_port)

def run_evaluation(collection_name: str, qdrant_host: str, qdrant_port: int) -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    embedder = Embedder()
    client = _qdrant_client(qdrant_host, qdrant_port)

    samples = []
    for item in EVAL_DATASET:
        question = item["question"]
        query_vector = embedder.embed([question])[0]
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3,
            with_payload=True,
        )
        contexts = [r.payload.get("text", "") for r in results if r.payload]

        state = {
            "question": question,
            "question_type": "",
            "retrieved_chunks": contexts,
            "retrieval_scores": [float(r.score or 0.0) for r in results],
            "answer": "",
            "critique": "",
            "confidence": 0.0,
            "retry_count": 0,
            "sources": [r.payload.get("source", "") for r in results if r.payload],
        }
        final_state = graph.invoke(state)

        samples.append(
            {
                "question": question,
                "answer": final_state.get("answer", ""),
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )

    dataset = Dataset.from_list(samples)
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    return {
        "faithfulness": float(results["faithfulness"][0]),
        "answer_relevancy": float(results["answer_relevancy"][0]),
        "num_samples": len(samples),
    }

if __name__ == "__main__":
    collection_name = os.getenv("COLLECTION_NAME", "techdocs")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    result = run_evaluation(collection_name, qdrant_host, qdrant_port)
    print(json.dumps(result, indent=2))
