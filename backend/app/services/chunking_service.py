"""Text chunking service for FinQuery knowledge-base documents."""


class ChunkingServiceError(Exception):
    """Raised when documents cannot be chunked."""


class ChunkingService:
    """Split documents into overlapping text chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ChunkingServiceError("Chunk size must be greater than zero.")
        if chunk_overlap < 0:
            raise ChunkingServiceError("Chunk overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ChunkingServiceError("Chunk overlap must be smaller than chunk size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: list[dict[str, str]]) -> list[dict[str, str | int]]:
        """Create overlapping chunks from loaded documents."""
        chunks: list[dict[str, str | int]] = []

        for document in documents:
            filename = document.get("filename", "").strip()
            content = document.get("content", "").strip()

            if not filename:
                raise ChunkingServiceError("A document is missing its filename.")
            if not content:
                raise ChunkingServiceError(f"Document has no readable content: {filename}")

            for chunk_index, chunk_text in enumerate(self._chunk_text(content)):
                chunks.append(
                    {
                        "filename": filename,
                        "chunk_index": chunk_index,
                        "content": chunk_text,
                    }
                )

        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start += step

        return chunks
