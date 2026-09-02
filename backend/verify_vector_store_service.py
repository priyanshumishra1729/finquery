"""Direct verification for Exercise 2 - Step 5 vector storage."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.app.services.chunking_service import ChunkingService
from backend.app.services.document_loader import DocumentLoader
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store_service import VectorStoreService, VectorStoreServiceError


def _embedding_to_list(embedding: object) -> list[float]:
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if not isinstance(embedding, list) or not embedding:
        raise AssertionError("Stored embedding is missing or invalid")

    if not all(isinstance(value, (int, float)) for value in embedding):
        raise AssertionError("Stored embedding contains non-numeric values")

    return [float(value) for value in embedding]


def run_verification() -> None:
    print("VectorStoreService import: ok")

    loader = DocumentLoader()
    documents = loader.load_documents()

    chunking_service = ChunkingService(chunk_size=500, chunk_overlap=100)
    chunks = chunking_service.chunk_documents(documents)

    embedding_service = EmbeddingService()
    embedded_chunks = embedding_service.embed_chunks(chunks)

    vector_store = VectorStoreService()
    print("VectorStoreService initialization: ok")
    print(f"Collection: {vector_store.collection_name}")

    vector_store.clear()

    try:
        vector_store.add_chunks(embedded_chunks)
    except VectorStoreServiceError as exc:
        raise AssertionError(f"Vector storage failed: {exc}") from exc

    stored_count = vector_store.count()
    if stored_count != 51:
        raise AssertionError(f"Expected 51 stored vectors, got {stored_count}")

    stored_items = vector_store.collection.get(include=["documents", "embeddings", "metadatas"])
    stored_ids = stored_items.get("ids", [])
    stored_documents = stored_items.get("documents", [])
    stored_embeddings = stored_items.get("embeddings", [])
    stored_metadatas = stored_items.get("metadatas", [])

    if len(stored_ids) != 51:
        raise AssertionError(f"Expected 51 stored ids, got {len(stored_ids)}")

    if len(stored_documents) != 51 or len(stored_embeddings) != 51 or len(stored_metadatas) != 51:
        raise AssertionError("Stored items are missing documents, embeddings, or metadata")

    if len(documents) != 5:
        raise AssertionError(f"Expected 5 documents, got {len(documents)}")

    chunk_counts = defaultdict(int)
    for chunk in chunks:
        chunk_counts[str(chunk["filename"])] += 1

    print()
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Chunks embedded: {len(embedded_chunks)}")
    print(f"Vectors stored: {stored_count}")

    for embedded_chunk, stored_document, stored_embedding, stored_metadata in zip(
        embedded_chunks,
        stored_documents,
        stored_embeddings,
        stored_metadatas,
        strict=True,
    ):
        if stored_document != embedded_chunk["content"]:
            raise AssertionError("Stored document content does not match the original chunk content")
        if stored_metadata.get("filename") != embedded_chunk["filename"]:
            raise AssertionError("Stored filename metadata was not preserved")
        if stored_metadata.get("chunk_index") != embedded_chunk["chunk_index"]:
            raise AssertionError("Stored chunk_index metadata was not preserved")
        stored_embedding_list = _embedding_to_list(stored_embedding)
        if len(stored_embedding_list) != len(embedded_chunk["embedding"]):
            raise AssertionError("Stored embedding dimension does not match the generated embedding")

    sample_metadata = stored_metadatas[0]
    sample_document = stored_documents[0]
    sample_embedding = _embedding_to_list(stored_embeddings[0])

    print()
    print("Sample stored item metadata:")
    print(f"filename: {sample_metadata.get('filename')}")
    print(f"chunk_index: {sample_metadata.get('chunk_index')}")
    print(f"content length: {len(str(sample_document))}")
    print(f"embedding dimension: {len(sample_embedding)}")

    print()
    print("count(): 51")
    print("Persistent path: backend/data/vector_store/")
    print("Storage source: local ChromaDB")
    print("Metadata and content preservation: passed")


if __name__ == "__main__":
    run_verification()
