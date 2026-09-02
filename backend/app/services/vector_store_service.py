"""Vector storage service for FinQuery knowledge-base chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


class VectorStoreServiceError(Exception):
    """Raised when vector storage operations fail."""


class VectorStoreService:
    """Persist embedded chunks in a local ChromaDB collection."""

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name: str = "finquery_knowledge_base",
    ) -> None:
        self.persist_directory = persist_directory or self._default_persist_directory()
        self.collection_name = collection_name

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:
            raise VectorStoreServiceError("Unable to initialize the local vector store.") from exc

    def add_chunks(self, chunks: list[dict[str, object]]) -> None:
        """Store embedded chunks in ChromaDB using deterministic chunk IDs."""
        if not chunks:
            raise VectorStoreServiceError("No chunks were provided for storage.")

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            filename = str(chunk.get("filename", "")).strip()
            chunk_index = chunk.get("chunk_index")
            content = str(chunk.get("content", "")).strip()
            embedding = chunk.get("embedding")

            if not filename:
                raise VectorStoreServiceError("A chunk is missing its filename.")
            if not isinstance(chunk_index, int):
                raise VectorStoreServiceError(f"Chunk index is missing or invalid for file: {filename}")
            if not content:
                raise VectorStoreServiceError(f"Chunk content is empty for file: {filename}, index: {chunk_index}")

            validated_embedding = self._validate_embedding(embedding, filename, chunk_index)

            ids.append(f"{filename}_{chunk_index}")
            embeddings.append(validated_embedding)
            documents.append(content)
            metadatas.append({"filename": filename, "chunk_index": chunk_index})

        try:
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        except Exception as exc:
            raise VectorStoreServiceError("Unable to store embedded chunks in ChromaDB.") from exc

    def count(self) -> int:
        """Return the number of stored vectors."""
        try:
            return int(self.collection.count())
        except Exception as exc:
            raise VectorStoreServiceError("Unable to read the vector store count.") from exc

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, object]]:
        """Return the most relevant chunks for a query embedding."""
        validated_query_embedding = self._validate_query_embedding(query_embedding)
        validated_top_k = self._validate_top_k(top_k)

        try:
            results = self.collection.query(
                query_embeddings=[validated_query_embedding],
                n_results=validated_top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreServiceError("Unable to search the vector store.") from exc

        return self._format_search_results(results)

    def clear(self) -> None:
        """Remove the stored knowledge-base collection and recreate it empty."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        try:
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:
            raise VectorStoreServiceError("Unable to clear the vector store.") from exc

    @staticmethod
    def _default_persist_directory() -> Path:
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "backend" / "data" / "vector_store"

    @staticmethod
    def _validate_embedding(
        embedding: object,
        filename: str,
        chunk_index: int,
    ) -> list[float]:
        if not isinstance(embedding, list) or not embedding:
            raise VectorStoreServiceError(
                f"Chunk embedding is missing or empty for file: {filename}, index: {chunk_index}"
            )

        validated_embedding: list[float] = []
        for value in embedding:
            if not isinstance(value, (int, float)):
                raise VectorStoreServiceError(
                    f"Chunk embedding contains invalid values for file: {filename}, index: {chunk_index}"
                )
            validated_embedding.append(float(value))

        return validated_embedding

    @staticmethod
    def _validate_query_embedding(query_embedding: list[float]) -> list[float]:
        if not isinstance(query_embedding, list) or not query_embedding:
            raise VectorStoreServiceError("Query embedding cannot be empty.")

        validated_query_embedding: list[float] = []
        for value in query_embedding:
            if not isinstance(value, (int, float)):
                raise VectorStoreServiceError("Query embedding contains invalid values.")
            validated_query_embedding.append(float(value))

        return validated_query_embedding

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if not isinstance(top_k, int) or top_k <= 0:
            raise VectorStoreServiceError("top_k must be a positive integer.")

        return top_k

    @staticmethod
    def _format_search_results(results: dict[str, Any]) -> list[dict[str, object]]:
        ids = results.get("ids", [[]])
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        distances = results.get("distances", [[]])

        if not ids or not documents or not metadatas or not distances:
            return []

        formatted_results: list[dict[str, object]] = []
        for index, _ in enumerate(ids[0]):
            metadata = metadatas[0][index] if metadatas[0] else None
            document = documents[0][index] if documents[0] else None
            distance = distances[0][index] if distances[0] else None

            if not isinstance(metadata, dict) or not isinstance(document, str) or distance is None:
                continue

            formatted_results.append(
                {
                    "filename": metadata.get("filename"),
                    "chunk_index": metadata.get("chunk_index"),
                    "content": document,
                    "distance": float(distance),
                }
            )

        return formatted_results