"""Embedding service for generating vectors from text chunks via Ollama."""

from __future__ import annotations

from typing import Any

import httpx

from backend.app.core.config import settings


class EmbeddingServiceError(Exception):
    """Raised when embeddings cannot be generated from Ollama."""


class EmbeddingService:
    """Generate embeddings using the local Ollama embedding API."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_EMBED_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_embedding(self, text: str) -> list[float]:
        """Generate one embedding vector for the provided text."""
        cleaned_text = text.strip()
        if not cleaned_text:
            raise EmbeddingServiceError("Cannot generate an embedding from empty text.")

        payload: dict[str, Any] = {
            "model": self.model,
            "input": cleaned_text,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise EmbeddingServiceError("Unable to connect to local Ollama. Please ensure Ollama is running.") from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingServiceError("Ollama embedding request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingServiceError("Ollama returned an unsuccessful status for embeddings.") from exc
        except httpx.HTTPError as exc:
            raise EmbeddingServiceError("An error occurred while requesting embeddings from Ollama.") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingServiceError("Ollama returned malformed JSON for embeddings.") from exc

        embedding = self._extract_embedding(data)
        if not embedding:
            raise EmbeddingServiceError("Ollama response did not include a non-empty embedding vector.")

        return embedding

    def embed_chunks(self, chunks: list[dict[str, str | int]]) -> list[dict[str, object]]:
        """Attach embeddings to each chunk while preserving chunk metadata and text."""
        embedded_chunks: list[dict[str, object]] = []

        for chunk in chunks:
            filename = str(chunk.get("filename", "")).strip()
            chunk_index = chunk.get("chunk_index")
            content = str(chunk.get("content", "")).strip()

            if not filename:
                raise EmbeddingServiceError("A chunk is missing its filename.")
            if not isinstance(chunk_index, int):
                raise EmbeddingServiceError(f"Chunk index is missing or invalid for file: {filename}")
            if not content:
                raise EmbeddingServiceError(f"Chunk content is empty for file: {filename}, index: {chunk_index}")

            embedding = self.generate_embedding(content)
            embedded_chunks.append(
                {
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "content": content,
                    "embedding": embedding,
                }
            )

        return embedded_chunks

    @staticmethod
    def _extract_embedding(data: Any) -> list[float]:
        """Extract and validate one embedding vector from Ollama response data."""
        if not isinstance(data, dict):
            raise EmbeddingServiceError("Ollama embedding response format is invalid.")

        if "embeddings" in data:
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or not embeddings:
                raise EmbeddingServiceError("Ollama embedding response is missing embeddings data.")

            first_embedding = embeddings[0]
            if not isinstance(first_embedding, list):
                raise EmbeddingServiceError("Ollama embedding vector format is invalid.")

            return EmbeddingService._validate_embedding_values(first_embedding)

        if "embedding" in data:
            legacy_embedding = data.get("embedding")
            if not isinstance(legacy_embedding, list):
                raise EmbeddingServiceError("Ollama embedding vector format is invalid.")

            return EmbeddingService._validate_embedding_values(legacy_embedding)

        raise EmbeddingServiceError("Ollama response is missing embedding values.")

    @staticmethod
    def _validate_embedding_values(values: list[Any]) -> list[float]:
        """Ensure embedding values are numeric and return them as floats."""
        embedding: list[float] = []
        for value in values:
            if not isinstance(value, (int, float)):
                raise EmbeddingServiceError("Embedding vector contains non-numeric values.")
            embedding.append(float(value))

        return embedding
