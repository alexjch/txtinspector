import os
from txtinspect.config import config
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding  # type: ignore[import-untyped]
from llama_index.core.node_parser import SentenceSplitter


class DocumentIndexer:
    """
    Handles document ingestion and indexing using LlamaIndex with Ollama models.
    This class configures the LLM and embedding models, loads documents from a directory,
    and creates a vector store index for efficient semantic search and retrieval.
    The indexer uses Ollama for both text generation and embeddings, requiring an
    Ollama service running at the specified base_url.
    """

    def __init__(
        self,
        embed_model: str = config.embed_model,
        base_url: str = config.llm_base_url,
        chunk_size: int = 512,
        chunk_overlap: int = 10,
        show_progress: bool = False,
    ) -> None:
        """
        Initialize the DocumentIndexer with Ollama model configurations.

        Args:
            embed_model: Name of the Ollama embedding model (e.g., 'nomic-embed-text').
            base_url: URL where Ollama service is running (e.g., 'http://localhost:11434').
            chunk_size: Maximum size of text chunks passed to the SentenceSplitter, controlling how
                large each document segment can be before splitting (default is 512).
            chunk_overlap: Number of characters (or tokens, depending on splitter configuration)
                that should overlap between consecutive chunks to preserve context (default is 10).
            show_progress: Display progress during indexing.
        """
        self.show_progress = show_progress
        # Initialize and store the embedding model for indexing operations.
        self.embed_model = OllamaEmbedding(
            model_name=embed_model,
            base_url=base_url,
        )
        # Initialize and store the text splitter to avoid recreating it in multiple places.
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def create_index(self, source: str) -> VectorStoreIndex:
        """
        Loads documents from a directory and creates a vector store index.
        The index enables semantic search by converting documents into embeddings
        and storing them for efficient similarity-based retrieval.

        Args:
            source: Path to directory containing documents to index

        Returns:
            VectorStoreIndex: The created vector store index built from the documents.

        Raises:
            ValueError: If the source path does not exist.
            TimeoutError: If Ollama service times out during query
            Exception: If document loading or indexing fails
        """
        # Validate that the source directory exists before attempting to load
        if not source or not source.strip() or not os.path.isdir(source):
            raise ValueError(
                f"Argument source is invalid, the path: {source} "
                "does not exist or is not a directory."
            )

        # Load all documents from the specified directory
        # SimpleDirectoryReader automatically detects and parses common file formats
        # Wrap in try-except to provide clearer error messages if loading fails.
        try:
            documents = SimpleDirectoryReader(source).load_data()
        except FileNotFoundError as exc:
            # Provide a clear error when the source directory cannot be found.
            raise FileNotFoundError(
                f"Source directory '{source}' was not found while loading documents."
            ) from exc
        except PermissionError as exc:
            # Provide a clear error when permissions prevent reading the source directory.
            raise PermissionError(
                f"Insufficient permissions to read source directory '{source}'."
            ) from exc
        except OSError as exc:
            # Handle other OS-level errors related to file system access.
            raise OSError(
                f"OS error occurred while loading documents from source directory '{source}'."
            ) from exc
        except ValueError as exc:
            # Handle parsing/format-related issues raised during document loading.
            raise ValueError(
                f"Failed to parse one or more documents in source directory '{source}'."
            ) from exc

        try:
            index = VectorStoreIndex.from_documents(
                documents=documents,
                embed_model=self.embed_model,
                show_progress=self.show_progress,
                transformations=[self.text_splitter],
            )
        except TimeoutError as exc:
            # Provide a clear error when the underlying LLM or embedding service times out.
            raise TimeoutError(
                "Timed out while creating the vector store index from loaded documents."
            ) from exc
        except ValueError as exc:
            # Handle configuration or data validation issues within the indexing pipeline.
            raise ValueError(
                "Invalid data or configuration encountered while creating the vector store index."
            ) from exc
        except RuntimeError as exc:
            # Handle generic runtime failures from the indexing components.
            raise RuntimeError(
                "A runtime error occurred while creating vector store index from loaded documents."
            ) from exc

        return index
