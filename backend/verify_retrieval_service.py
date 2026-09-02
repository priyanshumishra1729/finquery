"""Direct verification for Exercise 3 - Step 2 retrieval service."""

from __future__ import annotations

from backend.app.services.retrieval_service import RetrievalService, RetrievalServiceError


def _ensure_result_shape(result: dict[str, object]) -> None:
    required_keys = {"filename", "chunk_index", "content", "distance"}
    missing_keys = required_keys.difference(result.keys())
    if missing_keys:
        raise AssertionError(f"Retrieved result is missing keys: {sorted(missing_keys)}")

    if not isinstance(result["filename"], str) or not result["filename"].strip():
        raise AssertionError("Retrieved result filename is missing or invalid")
    if not isinstance(result["chunk_index"], int):
        raise AssertionError("Retrieved result chunk_index is missing or invalid")
    if not isinstance(result["content"], str) or not result["content"].strip():
        raise AssertionError("Retrieved result content is missing or empty")
    if not isinstance(result["distance"], float):
        raise AssertionError("Retrieved result distance is missing or invalid")


def run_verification() -> None:
    print("RetrievalService import: ok")

    retrieval_service = RetrievalService()
    print("RetrievalService initialization: ok")

    query = "What is the accounting equation?"
    print()
    print("Query:")
    print(f'"{query}"')

    if retrieval_service.embedding_service.model != "nomic-embed-text":
        raise AssertionError("RetrievalService is not using nomic-embed-text")
    print(f"Query embedding model: {retrieval_service.embedding_service.model}")

    query_embedding = retrieval_service.embedding_service.generate_embedding(query)
    if not isinstance(query_embedding, list) or not query_embedding:
        raise AssertionError("Query embedding is invalid")
    if not all(isinstance(value, float) for value in query_embedding):
        raise AssertionError("Query embedding contains non-float values")
    print(f"Query embedding dimension: {len(query_embedding)}")

    results = retrieval_service.retrieve(query, top_k=5)
    if len(results) != 5:
        raise AssertionError(f"Expected 5 retrieved results, got {len(results)}")

    for result in results:
        _ensure_result_shape(result)

    distances = [float(result["distance"]) for result in results]
    if distances != sorted(distances):
        raise AssertionError("Retrieved results are not ordered by ascending distance")

    first_content = str(results[0]["content"]).lower()
    if "accounting equation" not in first_content and "assets = liabilities + equity" not in first_content:
        raise AssertionError("Top result is not related to the accounting equation")

    print()
    print(f"Retrieved results: {len(results)}")
    print()

    for index, result in enumerate(results, start=1):
        preview = str(result["content"]).replace("\n", " ")[:220]
        print(f"{index}.")
        print(f"filename: {result['filename']}")
        print(f"chunk_index: {result['chunk_index']}")
        print(f"distance: {float(result['distance']):.6f}")
        print(f"content preview: {preview}")
        print()

    trimmed_results = retrieval_service.retrieve(f"  {query}  ", top_k=5)
    if trimmed_results != results:
        raise AssertionError("Whitespace trimming did not produce the same retrieval results")
    print("Whitespace trimming: ok")

    try:
        retrieval_service.retrieve("", top_k=5)
    except RetrievalServiceError:
        print("Empty query rejected: ok")
    else:
        raise AssertionError("Empty query was not rejected")

    try:
        retrieval_service.retrieve("   ", top_k=5)
    except RetrievalServiceError:
        print("Whitespace-only query rejected: ok")
    else:
        raise AssertionError("Whitespace-only query was not rejected")

    print()
    print("RetrievalService: passed")


if __name__ == "__main__":
    run_verification()
