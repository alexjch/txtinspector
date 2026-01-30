import os
from txtinspect.config import config
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding  # type: ignore[import-untyped]
from llama_index.llms.ollama import Ollama  # type: ignore[import-untyped]
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
        llm_model: str = config.llm_model,
        embedding_model: str = config.embedding_model,
        base_url: str = config.llm_base_url,
        chunk_size: int = 512,
        chunk_overlap: int = 10,
        show_progress: bool = False,
    ) -> None:
        """
        Initialize the DocumentIndexer with Ollama model configurations.

        Args:
            llm_model: Name of the Ollama LLM model to use (e.g., 'llama2', 'mistral')
            embedding_model: Name of the Ollama embedding model (e.g., 'nomic-embed-text')
            base_url: URL where Ollama service is running (e.g., 'http://localhost:11434')
            show_progress: Display progress during indexing
        """
        self.show_progress = show_progress
        self.text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Configure global LlamaIndex settings with Ollama models
        # These settings will be used by all index operations
        Settings.llm = Ollama(model=llm_model, base_url=base_url)
        Settings.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=base_url,
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
        except Exception as exc:
            # Raise a user-friendly exception while preserving the original traceback.
            raise Exception(f"Failed to load documents from source directory '{source}'.") from exc

        # Create vector store index from documents using configured embedding model
        # Documents are chunked and embedded automatically
        # Wrap in try-except to surface indexing failures with clear context.
        try:
            index = VectorStoreIndex.from_documents(
                documents, show_progress=self.show_progress, transformations=[self.text_splitter]
            )
        except Exception as exc:
            # Raise a user-friendly exception that indicates indexing failure.
            raise Exception("Failed to create vector store index from loaded documents.") from exc

        return index
