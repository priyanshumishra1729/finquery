"""Direct verification for Exercise 3 - Step 1 vector similarity search."""

from __future__ import annotations

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store_service import VectorStoreService, VectorStoreServiceError


def _ensure_result_shape(result: dict[str, object]) -> None:
    required_keys = {"filename", "chunk_index", "content", "distance"}
    missing_keys = required_keys.difference(result.keys())
    if missing_keys:
        raise AssertionError(f"Search result is missing keys: {sorted(missing_keys)}")

    if not isinstance(result["filename"], str) or not result["filename"].strip():
        raise AssertionError("Search result filename is missing or invalid")
    if not isinstance(result["chunk_index"], int):
        raise AssertionError("Search result chunk_index is missing or invalid")
    if not isinstance(result["content"], str) or not result["content"].strip():
        raise AssertionError("Search result content is missing or empty")
    if not isinstance(result["distance"], float):
        raise AssertionError("Search result distance is missing or invalid")


def run_verification() -> None:
    print("VectorStoreService import: ok")

    vector_store = VectorStoreService()
    print(f"Collection: {vector_store.collection_name}")

    stored_vectors = vector_store.count()
    if stored_vectors != 51:
        raise AssertionError(f"Expected 51 stored vectors, got {stored_vectors}")
    print(f"Stored vectors: {stored_vectors}")

    embedding_service = EmbeddingService()
    query_text = "What is the accounting equation?"
    try:
        query_embedding = embedding_service.generate_embedding(query_text)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Query embedding generation failed: {exc}") from exc

    if not isinstance(query_embedding, list) or not query_embedding:
        raise AssertionError("Query embedding is invalid")
    if not all(isinstance(value, float) for value in query_embedding):
        raise AssertionError("Query embedding contains non-float values")

    print()
    print(f'Query:')
    print(f'"{query_text}"')
    print()
    print(f"Query embedding dimension: {len(query_embedding)}")

    try:
        results = vector_store.search(query_embedding=query_embedding, top_k=5)
    except VectorStoreServiceError as exc:
        raise AssertionError(f"Vector search failed: {exc}") from exc

    if len(results) != 5:
        raise AssertionError(f"Expected 5 search results, got {len(results)}")

    for result in results:
        _ensure_result_shape(result)

    distances = [float(result["distance"]) for result in results]
    if distances != sorted(distances):
        raise AssertionError("Search results are not ordered by ascending distance")

    first_content = str(results[0]["content"]).lower()
    if "accounting equation" not in first_content and "assets = liabilities + equity" not in first_content:
        raise AssertionError("Most relevant result does not discuss the accounting equation")

    print()
    print(f"Search results: {len(results)}")
    print()

    for index, result in enumerate(results, start=1):
        content = str(result["content"])
        preview = content.replace("\n", " ")[:220]
        print(f"{index}.")
        print(f"filename: {result['filename']}")
        print(f"chunk_index: {result['chunk_index']}")
        print(f"distance: {float(result['distance']):.6f}")
        print(f"content: {preview}")
        print()

    print("Similarity search: passed")


if __name__ == "__main__":
    run_verification()
