"""Application orchestration service for the FinQuery chat workflow."""

from __future__ import annotations

from typing import Any

from backend.app.services.ollama_service import OllamaService, OllamaServiceError
from backend.app.services.rag_service import RAGService, RAGServiceError


class ApplicationServiceError(Exception):
    """Raised when chat orchestration cannot complete successfully."""


class ApplicationService:
    """Orchestrate RAG context building and LLM response generation."""

    def __init__(
        self,
        rag_service: RAGService | None = None,
        ollama_service: OllamaService | None = None,
    ) -> None:
        self.rag_service = rag_service or RAGService()
        self.ollama_service = ollama_service or OllamaService(timeout=1800.0)

    def process_chat(self, message: str) -> str:
        """Build RAG context, create the FinQuery prompt, and generate a response."""
        return self.process_chat_with_context(message)["answer"]

    def process_chat_with_context(self, message: str) -> dict[str, Any]:
        """Return the answer together with retrieved context and prompt metadata."""
        normalized_message = self._normalize_message(message)

        try:
            context = self.rag_service.build_context(normalized_message, top_k=5)
        except RAGServiceError as exc:
            raise ApplicationServiceError("Unable to build the RAG context.") from exc

        prompt = self._build_prompt(normalized_message, context)

        try:
            metadata = self.ollama_service.generate_response_with_metadata(prompt)
        except OllamaServiceError as exc:
            raise ApplicationServiceError("Unable to generate a response from the language model.") from exc

        return {
            "answer": metadata["response"],
            "context": context,
            "prompt": prompt,
            "metadata": metadata,
        }

    @staticmethod
    def _normalize_message(message: str) -> str:
        if not isinstance(message, str):
            raise ApplicationServiceError("Message must be a string.")

        normalized_message = message.strip()
        if not normalized_message:
            raise ApplicationServiceError("Message cannot be empty or whitespace only.")

        return normalized_message

    @staticmethod
    def _build_prompt(message: str, context: str) -> str:
        return f"""
You are FinQuery, a Financial Document Intelligence Assistant.

Use the supplied financial context to answer the user's question.

Rules:
- Use the supplied financial context when it is relevant.
- Answer clearly and accurately.
- Explain financial concepts clearly.
- Explain calculations when appropriate.
- Avoid fabricating financial facts.
- State when the supplied context is insufficient.
- Do not claim access to documents that were not supplied.
- Provide educational and informational guidance rather than personalized financial advice.

{context}

USER QUESTION:
{message}
""".strip()
