"""Document loading and chunking functionality."""

from pathlib import Path
from typing import List
from llama_index.core import SimpleDirectoryReader, Document


class DocumentLoaderException(Exception):
    """Custom exception for document loading errors."""

    def __init__(self, message: str, path: Path):
        """
        Initialize the exception with a message and optional path.

        Args:
            message: Error message describing what went wrong
            path: Optional path to the file/directory that caused the error
        """
        self.message = message
        self.path = path
        # Include path in the message if provided for better error context
        full_message = f"{message}: {path}" if path else message
        super().__init__(full_message)


class DocumentLoader:
    """Handle text file reading and chunking."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document loader.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: List[Document] = []

    def load(self, source_path: Path) -> None:
        """
        Load text present in source_path, this path can be a directory
        with supported files (txt, pdf, or md).

        Args:
            source_path: A path to a file or directory
        Returns:
            documents: a list of Documents loaded by llama-index.SimpleDirectoryReader
        """
        if not source_path.exists():
            raise DocumentLoaderException("Source path does not exist", source_path)

        if source_path.is_file() and source_path.name[-3:] in ["pdf", ".md", "txt"]:
            reader = SimpleDirectoryReader(input_files=[source_path])
            self.documents.extend(reader.load_data())

        else:
            reader = SimpleDirectoryReader(input_dir=source_path)
            self.documents.extend(reader.load_data())

    def chunk_text(self, text: str) -> List[str]:  # type: ignore
        """
        Split text into chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        # TODO: Implement text chunking logic
        pass
