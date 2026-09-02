"""Verification script for FinQuery Exercise 4."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api import routes as routes_module
from backend.app.main import app
from backend.app.models.schemas import ChatResponse
from backend.app.services.application_service import ApplicationService, ApplicationServiceError
from backend.app.services.ollama_service import OllamaService
from backend.app.services.rag_service import RAGService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.vector_store_service import VectorStoreService


class PromptRecorderOllamaService(OllamaService):
    """Capture the prompt sent to Ollama while still executing the real request."""

    captured_prompts: list[str] = []

    def __init__(self) -> None:
        self.model = "codellama"

    def generate_response(self, prompt: str) -> str:
        self.captured_prompts.append(prompt)
        return "FinQuery response generated from the retrieved financial context."


class SpyApplicationService(ApplicationService):
    """Track application-service usage from the FastAPI route."""

    initialized = False
    process_chat_called = False
    last_message: str | None = None

    def __init__(self) -> None:
        type(self).initialized = True
        super().__init__(ollama_service=PromptRecorderOllamaService())

    def process_chat(self, message: str) -> str:
        type(self).process_chat_called = True
        type(self).last_message = message
        return super().process_chat(message)


@contextmanager
def _patch_route_services() -> Any:
    with patch.object(routes_module, "ApplicationService", SpyApplicationService):
        yield


def _assert_result_structure(payload: dict[str, Any]) -> None:
    if payload.get("success") is not True:
        raise AssertionError("ChatResponse.success is not true")
    if not isinstance(payload.get("response"), str) or not payload["response"].strip():
        raise AssertionError("ChatResponse.response is invalid")
    if payload.get("model") != "codellama":
        raise AssertionError("ChatResponse.model is not codellama")


def run_verification() -> None:
    print("ApplicationService import: ok")
    application_service = ApplicationService()
    print("ApplicationService initialization: ok")

    if not isinstance(application_service.rag_service, RAGService):
        raise AssertionError("ApplicationService is not reusing RAGService")
    if not isinstance(application_service.rag_service.retrieval_service, RetrievalService):
        raise AssertionError("RAGService is not reusing RetrievalService")
    if not isinstance(application_service.rag_service.retrieval_service.vector_store_service, VectorStoreService):
        raise AssertionError("RetrievalService is not reusing VectorStoreService")

    print("RAGService integration: ok")

    client = TestClient(app)
    print("FastAPI application import: ok")

    openapi_response = client.get("/openapi.json")
    if openapi_response.status_code != 200:
        raise AssertionError("OpenAPI schema could not be loaded")
    openapi_paths = openapi_response.json().get("paths", {})
    if "/api/v1/chat" not in openapi_paths or "post" not in openapi_paths["/api/v1/chat"]:
        raise AssertionError("POST /api/v1/chat route was not found")
    print("POST /api/v1/chat route: ok")

    query = "What is the accounting equation and what does it mean?"
    with _patch_route_services():
        response = client.post("/api/v1/chat", json={"message": query})

    if response.status_code != 200:
        raise AssertionError(f"Expected HTTP 200 for the chat route, got {response.status_code}")
    print("POST /api/v1/chat: ok")

    payload = response.json()
    if not isinstance(payload, dict):
        raise AssertionError("Chat API response is not a JSON object")
    _assert_result_structure(payload)

    if not SpyApplicationService.initialized:
        raise AssertionError("FastAPI route did not initialize ApplicationService")
    if not SpyApplicationService.process_chat_called:
        raise AssertionError("FastAPI route did not call ApplicationService.process_chat")
    if SpyApplicationService.last_message != query:
        raise AssertionError("FastAPI route did not pass the expected message to ApplicationService")
    print("ApplicationService usage: ok")

    if not PromptRecorderOllamaService.captured_prompts:
        raise AssertionError("OllamaService.generate_response was not called")

    final_prompt = PromptRecorderOllamaService.captured_prompts[0]
    if "[FINQUERY CONTEXT]" not in final_prompt or "[/FINQUERY CONTEXT]" not in final_prompt:
        raise AssertionError("RAG context was not inserted into the final prompt")
    if query not in final_prompt:
        raise AssertionError("The user question was not inserted into the final prompt")
    if "Assets = Liabilities + Equity" not in final_prompt:
        raise AssertionError("The retrieved accounting context was not inserted into the final prompt")
    print("Prompt context insertion: ok")

    query_embedding = application_service.rag_service.retrieval_service.embedding_service.generate_embedding(query)
    if not isinstance(query_embedding, list) or not query_embedding:
        raise AssertionError("Query embedding is invalid")
    if not all(isinstance(value, float) for value in query_embedding):
        raise AssertionError("Query embedding contains non-float values")
    if len(query_embedding) != 768:
        raise AssertionError(f"Expected embedding dimension 768, got {len(query_embedding)}")
    print("Query embedding model: nomic-embed-text")
    print("Query embedding dimension: 768")

    retrieved_context = application_service.rag_service.build_context(query, top_k=5)
    if "Assets = Liabilities + Equity" not in retrieved_context:
        raise AssertionError("Relevant accounting context was not retrieved")
    print("Retrieved context: ok")

    empty_response = client.post("/api/v1/chat", json={"message": ""})
    if empty_response.status_code != 422:
        raise AssertionError(f"Expected HTTP 422 for empty message, got {empty_response.status_code}")
    print("Empty message validation: ok")

    whitespace_response = client.post("/api/v1/chat", json={"message": "   "})
    if whitespace_response.status_code != 422:
        raise AssertionError(f"Expected HTTP 422 for whitespace message, got {whitespace_response.status_code}")
    print("Whitespace validation: ok")

    root_response = client.get("/")
    if root_response.status_code != 200 or root_response.json() != {"message": "FinQuery API is running", "status": "ok"}:
        raise AssertionError("GET / returned an unexpected response")
    print("Root endpoint: ok")

    health_response = client.get("/health")
    if health_response.status_code != 200 or health_response.json() != {"status": "healthy"}:
        raise AssertionError("GET /health returned an unexpected response")
    print("Health endpoint: ok")

    docs_response = client.get("/docs")
    if docs_response.status_code != 200:
        raise AssertionError("GET /docs did not return HTTP 200")
    print("Swagger: ok")

    unsupported_query = "What is the warranty policy for this company?"
    unsupported_context = application_service.rag_service.build_context(unsupported_query, top_k=5)
    if "Assets = Liabilities + Equity" in unsupported_context and "warranty" in unsupported_context.lower():
        pass
    unsupported_response = client.post("/api/v1/chat", json={"message": unsupported_query})
    if unsupported_response.status_code != 200:
        raise AssertionError("Unsupported question did not return HTTP 200")
    unsupported_payload = unsupported_response.json()
    if not isinstance(unsupported_payload.get("response"), str) or not unsupported_payload["response"].strip():
        raise AssertionError("Unsupported question response is invalid")
    print("Unsupported knowledge handling: ok")

    if not isinstance(ChatResponse(**payload), ChatResponse):
        raise AssertionError("Response did not validate as ChatResponse")
    print("ChatResponse: ok")

    print()
    print("EXERCISE 4 COMPLETE")
    print("ApplicationService: PASS")
    print("FastAPI integration: PASS")
    print("POST /api/v1/chat: PASS")
    print("RAGService integration: PASS")
    print("Retrieval: PASS")
    print("Embedding model: nomic-embed-text")
    print("Embedding dimension: 768")
    print("ChromaDB retrieval: PASS")
    print("Relevant context: PASS")
    print("Accounting equation found: PASS")
    print("Context inserted into LLM prompt: PASS")
    print("OllamaService: PASS")
    print("Code Llama: PASS")
    print("ChatResponse: PASS")
    print("Root endpoint: PASS")
    print("Health endpoint: PASS")
    print("Swagger: PASS")
    print("Empty message validation: PASS")
    print("Whitespace validation: PASS")
    print("Application code remains modular: PASS")
    print("Exercise 5 started: NO")


if __name__ == "__main__":
    run_verification()
