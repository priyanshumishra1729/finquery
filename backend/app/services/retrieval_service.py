"""Retrieval service for turning user queries into relevant FinQuery chunks."""

from __future__ import annotations

from backend.app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from backend.app.services.vector_store_service import VectorStoreService, VectorStoreServiceError


class RetrievalServiceError(Exception):
    """Raised when retrieval cannot complete successfully."""


class RetrievalService:
    """Coordinate query embedding and vector similarity search."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store_service: VectorStoreService | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """Retrieve relevant chunks for the given user query."""
        normalized_query = self._normalize_query(query)
        validated_top_k = self._validate_top_k(top_k)

        try:
            query_embedding = self.embedding_service.generate_embedding(normalized_query)
        except EmbeddingServiceError as exc:
            raise RetrievalServiceError("Unable to generate a query embedding.") from exc

        try:
            return self.vector_store_service.search(query_embedding=query_embedding, top_k=validated_top_k)
        except VectorStoreServiceError as exc:
            raise RetrievalServiceError("Unable to retrieve relevant chunks from the vector store.") from exc

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str):
            raise RetrievalServiceError("Query must be a string.")

        normalized_query = query.strip()
        if not normalized_query:
            raise RetrievalServiceError("Query cannot be empty or whitespace only.")

        return normalized_query

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if not isinstance(top_k, int) or top_k <= 0:
            raise RetrievalServiceError("top_k must be a positive integer.")

        return top_k
