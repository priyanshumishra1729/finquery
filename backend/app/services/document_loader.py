"""Document loader for the FinQuery knowledge base."""

from pathlib import Path


class DocumentLoaderError(Exception):
    """Raised when knowledge-base documents cannot be loaded."""


class DocumentLoader:
    """Load educational text documents from the FinQuery knowledge base."""

    def __init__(self, knowledge_base_path: Path | None = None) -> None:
        self.knowledge_base_path = knowledge_base_path or self._default_knowledge_base_path()

    def load_documents(self) -> list[dict[str, str]]:
        """Load all .txt documents in deterministic filename order."""
        if not self.knowledge_base_path.exists():
            raise DocumentLoaderError("Knowledge-base directory was not found.")

        if not self.knowledge_base_path.is_dir():
            raise DocumentLoaderError("Knowledge-base path is not a directory.")

        text_files = sorted(self.knowledge_base_path.glob("*.txt"))
        if not text_files:
            raise DocumentLoaderError("No text documents were found in the knowledge base.")

        documents: list[dict[str, str]] = []
        for file_path in text_files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise DocumentLoaderError(f"Unable to read knowledge-base document: {file_path.name}") from exc

            documents.append(
                {
                    "filename": file_path.name,
                    "content": content,
                }
            )

        return documents

    @staticmethod
    def _default_knowledge_base_path() -> Path:
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "backend" / "data" / "knowledge_base"
