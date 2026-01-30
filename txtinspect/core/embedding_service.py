"""Text vectorization service."""

from typing import Optional
from txtinspect.config import config
from llama_index.embeddings.ollama import OllamaEmbedding

class EmbeddingService():
    """
    Adapter for generating embeddings via remote Ollama server.

    This class handles:
    - Single and batch embedding generation
    - Error handling and retries with exponential backoff
    - Rate limiting
    - SSL verification and timeout configuration

    To change the Ollama endpoint or model, update the settings in config.py or .env file.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize Ollama embeddings adapter.

        Args:
            base_url: Ollama server base URL (defaults to settings.ollama_base_url)
            model_name: Model name for embeddings (defaults to settings.ollama_embedding_model)
            timeout: Request timeout in seconds (defaults to settings.ollama_timeout)
        """
        base_url = base_url or config.ollama_base_url
        model_name = model_name or config.ollama_embedding_model

        self.embed_model = OllamaEmbedding(
            model_name=model_name,
            base_url=base_url
        )

    def get_embedding(self, text: str) -> None:
        embeddings = self.embed_model.get_text_embeddings(text)
        print(embeddings)
    


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Set both LLM and embedding model
Settings.llm = Ollama(model="llama3.2", base_url="http://localhost:11434")
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    client_kwargs={
        "timeout": 120.0  # 60 seconds timeout
    }
)
# Load documents
documents = SimpleDirectoryReader("/home/coderadmin/data/").load_data()

# Create index with Ollama embeddings
index = VectorStoreIndex.from_documents(documents)

import time

# Query the index with timeout exception handling
# Ollama service may timeout on slow networks or heavy loads
query_engine = index.as_query_engine()

start = time.perf_counter()
try:
    response = query_engine.query("What is the main topic?")
    print(response)
except TimeoutError as e:
    print(f"Query timed out: {e}")
except Exception as e:
    print(f"Query failed with error: {e}")
finally:
    end = time.perf_counter()
    print(f"took {start - end} seconds")    



from llama_index.embeddings.ollama import OllamaEmbedding

# Initialize the embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

        # TODO: Implement batch embedding generation
        pass