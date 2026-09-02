"""Direct verification for Exercise 2 - Step 4 embedding generation."""

from __future__ import annotations

from backend.app.services.chunking_service import ChunkingService
from backend.app.services.document_loader import DocumentLoader
from backend.app.services.embedding_service import EmbeddingService, EmbeddingServiceError


def _assert_valid_embedding(embedding: object) -> list[float]:
    if not isinstance(embedding, list):
        raise AssertionError("Embedding is not a Python list")
    if not embedding:
        raise AssertionError("Embedding list is empty")
    if not all(isinstance(value, float) for value in embedding):
        raise AssertionError("Embedding contains non-float values")
    return embedding


def run_verification() -> None:
    print("EmbeddingService import: ok")

    loader = DocumentLoader()
    documents = loader.load_documents()

    chunking_service = ChunkingService(chunk_size=500, chunk_overlap=100)
    chunks = chunking_service.chunk_documents(documents)

    embedding_service = EmbeddingService()
    print("EmbeddingService initialization: ok")
    print(f"Embedding model: {embedding_service.model}")

    if embedding_service.model != "nomic-embed-text":
        raise AssertionError(
            "Embedding model mismatch. Expected 'nomic-embed-text'. "
            f"Received: '{embedding_service.model}'"
        )

    test_text = "Revenue is the total income generated from normal business operations."
    try:
        test_embedding = embedding_service.generate_embedding(test_text)
    except EmbeddingServiceError as exc:
        raise AssertionError(f"Test embedding request failed: {exc}") from exc

    valid_test_embedding = _assert_valid_embedding(test_embedding)

    print()
    print("Test embedding:")
    print("Type: list")
    print(f"Dimensions: {len(valid_test_embedding)}")
    print("Valid floats: yes")

    if len(documents) != 5:
        raise AssertionError(f"Expected 5 documents, got {len(documents)}")

    if len(chunks) != 51:
        raise AssertionError(f"Expected 51 chunks, got {len(chunks)}")

    try:
        embedded_chunks = embedding_service.embed_chunks(chunks)
    except EmbeddingServiceError as exc:
        raise AssertionError(f"Chunk embedding failed: {exc}") from exc

    if len(embedded_chunks) != 51:
        raise AssertionError(f"Expected 51 embedded chunks, got {len(embedded_chunks)}")

    for original_chunk, embedded_chunk in zip(chunks, embedded_chunks, strict=True):
        if embedded_chunk.get("filename") != original_chunk.get("filename"):
            raise AssertionError("Filename was not preserved in embedded chunk")
        if embedded_chunk.get("chunk_index") != original_chunk.get("chunk_index"):
            raise AssertionError("Chunk index was not preserved in embedded chunk")
        if embedded_chunk.get("content") != original_chunk.get("content"):
            raise AssertionError("Chunk content was not preserved in embedded chunk")

        embedding = _assert_valid_embedding(embedded_chunk.get("embedding"))
        if len(embedding) != len(valid_test_embedding):
            raise AssertionError("Inconsistent embedding dimension found across chunks")

    sample_chunk = embedded_chunks[0]
    sample_embedding = _assert_valid_embedding(sample_chunk["embedding"])

    print()
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Chunks embedded: {len(embedded_chunks)}")

    print()
    print("Sample chunk metadata:")
    print(f"filename: {sample_chunk['filename']}")
    print(f"chunk_index: {sample_chunk['chunk_index']}")
    print(f"content length: {len(str(sample_chunk['content']))}")
    print(f"embedding dimension: {len(sample_embedding)}")

    print()
    print("Embeddings source: local Ollama API")


if __name__ == "__main__":
    run_verification()
