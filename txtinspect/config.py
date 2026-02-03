"""Configuration management for txtinspect."""

import os


class Config:
    """Application configuration."""

    # Default values
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50
    DEFAULT_EMBED_MODEL = "nomic-embed-text"
    DEFAULT_LLM_MODEL = "gemma:2b"
    DEFAULT_LLM_BASE_URL = "http://localhost:11434"
    DEFAULT_VECTOR_STORE = "chroma"
    DEFAULT_PERSIST_DIR = "./chroma_db"
    DEFAULT_TOP_K = 5

    def __init__(self) -> None:
        """Initialize configuration from environment variables or defaults."""
        # Load chunking configuration from environment or use defaults
        self.chunk_size = int(os.getenv("TXTINSPECT_CHUNK_SIZE", self.DEFAULT_CHUNK_SIZE))
        self.chunk_overlap = int(os.getenv("TXTINSPECT_CHUNK_OVERLAP", self.DEFAULT_CHUNK_OVERLAP))
        # Load embedding model configuration
        self.embed_model = os.getenv("TXTINSPECT_EMBEDDING_MODEL", self.DEFAULT_EMBED_MODEL)
        # Load LLM configuration from environment or use defaults
        self.llm_model = os.getenv("TXTINSPECT_LLM_MODEL", self.DEFAULT_LLM_MODEL)
        self.llm_base_url = os.getenv("TXTINSPECT_LLM_BASE_URL", self.DEFAULT_LLM_BASE_URL)
        # Load vector store configuration
        self.vector_store = os.getenv("TXTINSPECT_VECTOR_STORE", self.DEFAULT_VECTOR_STORE)
        self.persist_dir = os.getenv("TXTINSPECT_PERSIST_DIR", self.DEFAULT_PERSIST_DIR)
        self.top_k = int(os.getenv("TXTINSPECT_TOP_K", self.DEFAULT_TOP_K))


# Global config instance
config = Config()
