# TechDoc Intelligence Platform

[![CI](https://github.com/medboukechouch/techdoc-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/medboukechouch/techdoc-intelligence/actions/workflows/ci.yml)

An agentic RAG system that ingests technical PDF documents and answers engineering queries using LangGraph, Qdrant, and Gemini — with automatic quality evaluation via RAGAS.

## Architecture
The platform is organized into four layers: ingestion (PyMuPDF parsing, chunking, and Qdrant persistence), the agent (a LangGraph 5-node StateGraph with a retry loop), an MCP server (FastMCP tools for search, ingestion, summary, and evaluation), and evaluation (RAGAS metrics for faithfulness and answer relevancy).

```
PDF → parser → chunker → embedder → Qdrant
																				↓
Question → router_node → retrieval_node → validation_node
																↑ retry          ↓ continue
												 answer_node → critique_node → Response
```

## Tech Stack
| Component | Technology |
| --- | --- |
| Vector DB | Qdrant |
| Agent Framework | LangGraph |
| LLM | Gemini 1.5 Flash |
| API | FastAPI (SSE streaming) |
| MCP Server | FastMCP |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Evaluation | RAGAS |
| Containerization | Docker + docker-compose |

## Quick Start
```bash
git clone https://github.com/medboukechouch/techdoc-intelligence
cd techdoc-intelligence
cp .env.example .env
# Add your GEMINI_API_KEY in .env
docker-compose up --build
```

## Usage
1) Ingest a PDF:
```bash
curl -X POST http://localhost:8000/ingest \
	-H "Content-Type: application/json" \
	-d '{"pdf_path": "./data/manual.pdf"}'
```

2) Query the system:
```bash
curl -N -X POST http://localhost:8000/query \
	-H "Content-Type: application/json" \
	-d '{"question": "How do I configure the system?"}'
```

3) Run evaluation:
```bash
python -m evaluation.evaluate
```

## Project Structure
```
techdoc-intelligence/
├── .github/
│   └── workflows/
│       └── ci.yml
├── ingestion/
│   ├── __init__.py
│   ├── parser.py
│   ├── chunker.py
│   └── embedder.py
├── agent/
│   ├── __init__.py
│   ├── state.py
│   ├── nodes.py
│   └── graph.py
├── mcp_server/
│   ├── __init__.py
│   └── server.py
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py
├── api/
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_agent.py
│   ├── test_api.py
│   └── test_mcp.py
├── data/
│   └── .gitkeep
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Why This Architecture
LangGraph provides a stateful graph with explicit control flow and retry loops, which is clearer and safer than ad-hoc chains. Qdrant offers persistent storage and a production-ready REST API, unlike FAISS which is primarily in-memory. RAGAS gives quantitative checks for groundedness and relevancy, moving beyond subjective evaluations.
