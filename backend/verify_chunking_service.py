"""Direct verification for Exercise 2 - Step 3 chunking service."""

from __future__ import annotations

from collections import Counter, defaultdict

from app.services.chunking_service import ChunkingService
from app.services.document_loader import DocumentLoader


def _shared_boundary_overlap(left: str, right: str, max_overlap: int) -> int:
    """Return the longest suffix/prefix overlap length between two chunks."""
    max_len = min(len(left), len(right), max_overlap)
    for overlap_len in range(max_len, 0, -1):
        if left[-overlap_len:] == right[:overlap_len]:
            return overlap_len
    return 0


def run_verification() -> None:
    loader = DocumentLoader()
    documents = loader.load_documents()
    if len(documents) != 5:
        raise AssertionError(f"Expected 5 documents, got {len(documents)}")
    print(f"DocumentLoader: {len(documents)} documents loaded")

    chunk_service = ChunkingService(chunk_size=500, chunk_overlap=100)
    print("ChunkingService: initialized")

    chunks = chunk_service.chunk_documents(documents)

    if len(chunks) <= 5:
        raise AssertionError("Expected more than 5 chunks in total")

    chunks_per_file: Counter[str] = Counter()
    indexes_per_file: dict[str, list[int]] = defaultdict(list)

    for chunk in chunks:
        if "filename" not in chunk or "chunk_index" not in chunk or "content" not in chunk:
            raise AssertionError("Chunk missing one of required keys: filename, chunk_index, content")

        filename = str(chunk["filename"])
        chunk_index = int(chunk["chunk_index"])
        content = str(chunk["content"])

        if not content.strip():
            raise AssertionError(f"Chunk has empty content for file: {filename}")

        chunks_per_file[filename] += 1
        indexes_per_file[filename].append(chunk_index)

    expected_filenames = sorted(doc["filename"] for doc in documents)

    for filename in expected_filenames:
        if chunks_per_file[filename] < 1:
            raise AssertionError(f"No chunks generated for file: {filename}")

        sorted_indexes = sorted(indexes_per_file[filename])
        expected_indexes = list(range(len(sorted_indexes)))
        if sorted_indexes != expected_indexes:
            raise AssertionError(
                f"Chunk indexes are not deterministic from 0 for {filename}. "
                f"Got {sorted_indexes}, expected {expected_indexes}"
            )

    print()
    print(f"Total chunks: {len(chunks)}")
    print()

    for filename in expected_filenames:
        print(f"{filename}: {chunks_per_file[filename]} chunks")

    sample_filename = expected_filenames[0]
    sample_chunk = next(
        chunk
        for chunk in chunks
        if chunk["filename"] == sample_filename and int(chunk["chunk_index"]) == 0
    )

    print()
    print("Sample chunk:")
    print(sample_chunk)

    overlap_verified = False
    overlap_details = ""

    chunks_by_file: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_file[str(chunk["filename"])].append(chunk)

    for filename in expected_filenames:
        ordered_chunks = sorted(chunks_by_file[filename], key=lambda item: int(item["chunk_index"]))
        if len(ordered_chunks) < 2:
            continue

        first_text = str(ordered_chunks[0]["content"])
        second_text = str(ordered_chunks[1]["content"])
        overlap_len = _shared_boundary_overlap(first_text, second_text, chunk_service.chunk_overlap)

        if overlap_len > 0:
            overlap_verified = True
            overlap_details = f"{filename} (overlap length: {overlap_len})"
            break

    if not overlap_verified:
        raise AssertionError("Chunk overlap verification failed")

    print()
    print(f"Chunk overlap verification: passed ({overlap_details})")


if __name__ == "__main__":
    run_verification()
