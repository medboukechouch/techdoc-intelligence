"""Ingestion package initialiser."""
from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ingestion.chunker import chunk_text
from ingestion.embedder import Embedder
from ingestion.parser import parse_pdf

def ingest_pdf(
	pdf_path: str,
	collection_name: str,
	qdrant_host: str,
	qdrant_port: int,
) -> int:
	"""Parse, chunk, embed, and upsert a PDF into Qdrant."""
	text = parse_pdf(pdf_path)
	chunks = chunk_text(text)
	embedder = Embedder()
	vectors = embedder.embed(chunks)

	client = QdrantClient(host=qdrant_host, port=qdrant_port)
	if not client.collection_exists(collection_name):
		client.create_collection(
			collection_name=collection_name,
			vectors_config=VectorParams(size=384, distance=Distance.COSINE),
		)

	source_name = Path(pdf_path).name
	points: list[PointStruct] = []
	for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
		points.append(
			PointStruct(
				id=idx,
				vector=vector,
				payload={"text": chunk, "source": source_name},
			)
		)

	if points:
		client.upsert(collection_name=collection_name, points=points)
	print(f"Ingested {len(points)} chunks")
	return len(points)
