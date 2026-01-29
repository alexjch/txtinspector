"""Text vectorization service."""

from typing import List
from txtinspect.config import config
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class EmbeddingService:
    """Handle text embedding generation."""

    def __init__(self, model_name: str = "default") -> None:
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the embedding model
        """
        self.model_name = model_name
        self.llm_base_url = config.llm_base_url
        self.embedding_model = config.embedding_model

        # Session
        self.session = self._new_session()

    def _new_session(self):
        """
        Create a requests session with retry logic.
        Uses urllib3.util.Retry for automatic retries on connection failures.
        Configured with exponential backoff to handle transient network issues.

        Returns:
            requests.Session: Configured session with retry strategy
        """

        # Configure retry strategy with exponential backoff
        # Retries on connection errors, timeouts, and specific HTTP status codes
        retry_strategy = Retry(
            total=3,  # Maximum number of retry attempts
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP codes to retry
            allowed_methods=["POST", "GET"],  # Methods that can be retried
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()

        # Apply retry strategy to both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def embed_text(self, text: str) -> List[float]:  # type: ignore
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # TODO: Implement embedding generation
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:  # type: ignore
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        # TODO: Implement batch embedding generation
        pass
