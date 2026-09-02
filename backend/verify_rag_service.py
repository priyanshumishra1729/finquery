"""Direct verification for Exercise 3 - Step 3 RAG context building."""

from __future__ import annotations

from backend.app.services.rag_service import RAGService, RAGServiceError


def run_verification() -> None:
    print("RAGService import: ok")

    rag_service = RAGService()
    print("RAGService initialization: ok")

    query = "What is the accounting equation?"
    print()
    print("Query:")
    print(f'"{query}"')

    if rag_service.retrieval_service.embedding_service.model != "nomic-embed-text":
        raise AssertionError("RAGService is not reusing the nomic-embed-text embedding model")
    print(f"Query embedding model: {rag_service.retrieval_service.embedding_service.model}")

    query_embedding = rag_service.retrieval_service.embedding_service.generate_embedding(query)
    if not isinstance(query_embedding, list) or not query_embedding:
        raise AssertionError("Query embedding is invalid")
    if not all(isinstance(value, float) for value in query_embedding):
        raise AssertionError("Query embedding contains non-float values")
    print(f"Query embedding dimension: {len(query_embedding)}")

    retrieved_chunks = rag_service.retrieval_service.retrieve(query, top_k=5)
    if len(retrieved_chunks) != 5:
        raise AssertionError(f"Expected 5 retrieved chunks, got {len(retrieved_chunks)}")
    print(f"Retrieved chunks: {len(retrieved_chunks)}")

    context = rag_service.build_context(query, top_k=5)
    if not isinstance(context, str) or not context.strip():
        raise AssertionError("Context was not generated")

    print()
    print("Context generated: yes")
    print(f"Context length: {len(context)} characters")
    print()
    print("Context preview:")
    print()
    print(context)
    print()

    if "[FINQUERY CONTEXT]" not in context or "[/FINQUERY CONTEXT]" not in context:
        raise AssertionError("Context delimiters are missing")
    if "[Source:" not in context or "| Chunk:" not in context:
        raise AssertionError("Context metadata is missing")
    if "embedding" in context.lower():
        raise AssertionError("Context should not contain embeddings")
    if "distance" in context.lower():
        raise AssertionError("Context should not contain raw technical distance information")

    first_content = str(retrieved_chunks[0]["content"]).lower()
    if "accounting equation" not in first_content and "assets = liabilities + equity" not in first_content:
        raise AssertionError("Retrieved content does not appear to discuss the accounting equation")

    print("Contains source metadata: yes")
    print("Contains chunk metadata: yes")
    print("Contains financial content: yes")
    print("Contains embeddings: no")

    trimmed_context = rag_service.build_context(f"  {query}  ", top_k=5)
    if trimmed_context != context:
        raise AssertionError("Whitespace trimming did not preserve the same context")
    print("Whitespace trimming: ok")

    try:
        rag_service.build_context("", top_k=5)
    except RAGServiceError:
        print("Empty query rejected: ok")
    else:
        raise AssertionError("Empty query was not rejected")

    try:
        rag_service.build_context("   ", top_k=5)
    except RAGServiceError:
        print("Whitespace-only query rejected: ok")
    else:
        raise AssertionError("Whitespace-only query was not rejected")

    print()
    print("RAGService: passed")


if __name__ == "__main__":
    run_verification()
