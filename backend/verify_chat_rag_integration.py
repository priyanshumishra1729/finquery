"""Direct integration verification for Exercise 3 - Step 4 chat RAG wiring."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api import routes as routes_module
from backend.app.main import app
from backend.app.services.ollama_service import OllamaService
from backend.app.services.rag_service import RAGService


def _build_no_rag_prompt(query: str) -> str:
    return f"""
You are FinQuery, a Financial Document Intelligence Assistant.

For Exercise 1, you do not yet have access to uploaded financial documents or a knowledge base.

Follow these rules:
- Explain financial concepts clearly.
- Answer questions as accurately as possible.
- Explain calculations when appropriate.
- Avoid fabricating financial facts or data.
- Clearly distinguish known information from assumptions.
- Do not claim to have access to financial documents that were not provided.
- Provide informational guidance rather than personalized financial advice.

User question:
{query}
""".strip()


@contextmanager
def _capture_ollama_prompts() -> Any:
    captured_prompts: list[str] = []
    real_service_class = OllamaService

    class CapturingOllamaService:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._service = real_service_class(*args, **kwargs)

        def generate_response(self, prompt: str) -> str:
            captured_prompts.append(prompt)
            return self._service.generate_response(prompt)

    with patch.object(routes_module, "OllamaService", CapturingOllamaService):
        yield captured_prompts


def run_verification() -> None:
    client = TestClient(app)
    print("FastAPI start: ok")

    root_response = client.get("/")
    if root_response.status_code != 200 or root_response.json() != {"message": "FinQuery API is running", "status": "ok"}:
        raise AssertionError("GET / did not return the expected response")
    print("GET /: ok")

    health_response = client.get("/health")
    if health_response.status_code != 200 or health_response.json() != {"status": "healthy"}:
        raise AssertionError("GET /health did not return the expected response")
    print("GET /health: ok")

    docs_response = client.get("/docs")
    if docs_response.status_code != 200:
        raise AssertionError("GET /docs did not return HTTP 200")
    print("GET /docs: ok")

    rag_service = RAGService()
    query = "What is the accounting equation?"
    query_embedding = rag_service.retrieval_service.embedding_service.generate_embedding(query)
    if not isinstance(query_embedding, list) or not query_embedding:
        raise AssertionError("Query embedding is invalid")
    if not all(isinstance(value, float) for value in query_embedding):
        raise AssertionError("Query embedding contains non-float values")

    print()
    print("Query:")
    print(f'"{query}"')
    print(f"Query embedding model: {rag_service.retrieval_service.embedding_service.model}")
    print(f"Query embedding dimension: {len(query_embedding)}")

    retrieved_chunks = rag_service.retrieval_service.retrieve(query, top_k=5)
    if len(retrieved_chunks) != 5:
        raise AssertionError(f"Expected 5 retrieved chunks, got {len(retrieved_chunks)}")
    print(f"Retrieved chunks: {len(retrieved_chunks)}")

    context = rag_service.build_context(query, top_k=5)
    if "Assets = Liabilities + Equity" not in context:
        raise AssertionError("Retrieved context does not contain the accounting equation")
    print("Retrieved context contains: Assets = Liabilities + Equity")

    no_rag_prompt = _build_no_rag_prompt(query)
    no_rag_response = OllamaService().generate_response(no_rag_prompt)
    if not isinstance(no_rag_response, str) or not no_rag_response.strip():
        raise AssertionError("No-RAG response is invalid")

    print()
    print("Without RAG prompt:")
    print(no_rag_prompt)
    print()
    print("Without RAG response:")
    print(no_rag_response)

    with _capture_ollama_prompts() as captured_prompts:
        rag_response = client.post("/api/v1/chat", json={"message": query})

    if rag_response.status_code != 200:
        raise AssertionError(f"POST /api/v1/chat returned HTTP {rag_response.status_code}")

    rag_payload = rag_response.json()
    if not isinstance(rag_payload, dict):
        raise AssertionError("RAG response payload is not a JSON object")
    if rag_payload.get("success") is not True:
        raise AssertionError("ChatResponse.success is not true")
    if rag_payload.get("model") != "codellama":
        raise AssertionError(f"Expected model codellama, got {rag_payload.get('model')}")
    if not isinstance(rag_payload.get("response"), str):
        raise AssertionError("ChatResponse.response is not a string")

    if not captured_prompts:
        raise AssertionError("OllamaService.generate_response was not called")

    rag_prompt = captured_prompts[0]
    if "[FINQUERY CONTEXT]" not in rag_prompt or "[/FINQUERY CONTEXT]" not in rag_prompt:
        raise AssertionError("RAG context was not passed to OllamaService")
    if "Assets = Liabilities + Equity" not in rag_prompt:
        raise AssertionError("The retrieved financial context did not reach Code Llama")
    if query not in rag_prompt:
        raise AssertionError("The user question was not included in the prompt")

    print()
    print("RAG prompt sent to Ollama:")
    print(rag_prompt)
    print()
    print("RAG response:")
    print(rag_payload["response"])
    print()
    print("POST /api/v1/chat: ok")
    print("Code Llama request: ok")
    print("ChatResponse: ok")

    unsupported_query = "What is the warranty policy for this company?"
    unsupported_context = rag_service.build_context(unsupported_query, top_k=5)
    unsupported_response = client.post("/api/v1/chat", json={"message": unsupported_query})
    if unsupported_response.status_code != 200:
        raise AssertionError("Unsupported question did not return HTTP 200")
    unsupported_payload = unsupported_response.json()
    if not isinstance(unsupported_payload.get("response"), str):
        raise AssertionError("Unsupported question response is not a string")

    print()
    print("Unsupported query context:")
    print(unsupported_context)
    print()
    print("Unsupported query response:")
    print(unsupported_payload["response"])
    print("Unsupported knowledge handling: ok")

    empty_response = client.post("/api/v1/chat", json={"message": ""})
    if empty_response.status_code != 422:
        raise AssertionError(f"Expected HTTP 422 for empty message, got {empty_response.status_code}")
    print("Empty-message validation: ok")

    whitespace_response = client.post("/api/v1/chat", json={"message": "   "})
    if whitespace_response.status_code != 422:
        raise AssertionError(f"Expected HTTP 422 for whitespace message, got {whitespace_response.status_code}")
    print("Whitespace-message validation: ok")

    print()
    print("RAG integration status: passed")


if __name__ == "__main__":
    run_verification()
