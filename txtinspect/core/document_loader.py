"""Document loading and chunking functionality."""

from pathlib import Path
from typing import List
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser.text import TokenTextSplitter


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
        self.splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load(self, source_path: Path) -> List[Document]:
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

        documents: List[Document] = []

        if source_path.is_file() and source_path.suffix in [".pdf", ".md", ".txt"]:
            reader = SimpleDirectoryReader(input_files=[source_path])
            documents.extend(reader.load_data())

        else:
            reader = SimpleDirectoryReader(input_dir=source_path)
            documents.extend(reader.load_data())

        return documents

    def chunk(self, documents: List[Document]) -> List[str]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of text chunks as strings

        Raises:
            ValueError: If no documents are provided
        """
        if len(documents) == 0:
            raise ValueError("No documents provided for chunking")

        chunks: List[str] = []

        for document in documents:
            # Get nodes from splitter and extract text
            nodes = self.splitter.split_text(document.text)
            # Directly extend with nodes iterable to avoid unnecessary list creation
            chunks.extend(nodes)

        return chunks
