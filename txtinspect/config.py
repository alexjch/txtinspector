"""Configuration management for txtinspect."""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration."""

    # Default values
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
    DEFAULT_LLM_MODEL = "llama3.2"
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
        self.embedding_model = os.getenv("TXTINSPECT_EMBEDDING_MODEL", self.DEFAULT_EMBEDDING_MODEL)
        # Load LLM configuration from environment or use defaults
        self.llm_model = os.getenv("TXTINSPECT_LLM_MODEL", self.DEFAULT_LLM_MODEL)
        self.llm_base_url = os.getenv("TXTINSPECT_LLM_BASE_URL", self.DEFAULT_LLM_BASE_URL)
        # Load vector store configuration
        self.vector_store = os.getenv("TXTINSPECT_VECTOR_STORE", self.DEFAULT_VECTOR_STORE)
        self.persist_dir = os.getenv("TXTINSPECT_PERSIST_DIR", self.DEFAULT_PERSIST_DIR)
        self.top_k = int(os.getenv("TXTINSPECT_TOP_K", self.DEFAULT_TOP_K))

    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from a TOML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Config instance
        """
        # TODO: Implement TOML configuration loading
        return cls()


# Global config instance
config = Config()
