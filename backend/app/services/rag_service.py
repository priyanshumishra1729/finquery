"""RAG context service for turning retrieved chunks into a prompt-ready context."""

from __future__ import annotations

from backend.app.services.retrieval_service import RetrievalService, RetrievalServiceError


class RAGServiceError(Exception):
    """Raised when a RAG context cannot be built."""


class RAGService:
    """Build a clean context string from retrieved knowledge-base chunks."""

    def __init__(self, retrieval_service: RetrievalService | None = None) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()

    def build_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve the most relevant chunks and format them as a context string."""
        normalized_query = self._normalize_query(query)
        validated_top_k = self._validate_top_k(top_k)

        try:
            retrieved_chunks = self.retrieval_service.retrieve(normalized_query, top_k=validated_top_k)
        except RetrievalServiceError as exc:
            raise RAGServiceError("Unable to build context from the knowledge base.") from exc

        return self._format_context(retrieved_chunks)

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str):
            raise RAGServiceError("Query must be a string.")

        normalized_query = query.strip()
        if not normalized_query:
            raise RAGServiceError("Query cannot be empty or whitespace only.")

        return normalized_query

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if not isinstance(top_k, int) or top_k <= 0:
            raise RAGServiceError("top_k must be a positive integer.")

        return top_k

    @staticmethod
    def _format_context(retrieved_chunks: list[dict[str, object]]) -> str:
        if not retrieved_chunks:
            return (
                "[FINQUERY CONTEXT]\n"
                "No relevant information was found in the knowledge base.\n"
                "[/FINQUERY CONTEXT]"
            )

        sections: list[str] = ["[FINQUERY CONTEXT]"]
        for chunk in retrieved_chunks:
            filename = str(chunk.get("filename", "")).strip() or "unknown_source"
            chunk_index = chunk.get("chunk_index")
            content = str(chunk.get("content", "")).strip()

            sections.append(f"[Source: {filename} | Chunk: {chunk_index}]")
            sections.append(content)
            sections.append("")

        sections.append("[/FINQUERY CONTEXT]")
        return "\n".join(sections).rstrip()
