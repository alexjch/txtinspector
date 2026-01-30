"""Tests for the DocumentIndexer class."""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Mock llama_index modules before importing DocumentIndexer
# This prevents network calls to Ollama during testing
sys.modules["llama_index"] = Mock()
sys.modules["llama_index.core"] = Mock()
sys.modules["llama_index.core.node_parser"] = Mock()
sys.modules["llama_index.embeddings"] = Mock()
sys.modules["llama_index.embeddings.ollama"] = Mock()
sys.modules["llama_index.llms"] = Mock()
sys.modules["llama_index.llms.ollama"] = Mock()

from txtinspect.core.document_indexer import DocumentIndexer  # noqa: E402


class TestDocumentIndexer(unittest.TestCase):
    """Test cases for DocumentIndexer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test documents
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.txt"
        self.test_file.write_text("This is a test document for indexing.")

        # Patch the llama_index imports to prevent actual initialization
        self.patcher_ollama = patch("txtinspect.core.document_indexer.Ollama")
        self.patcher_embedding = patch("txtinspect.core.document_indexer.OllamaEmbedding")
        self.patcher_settings = patch("txtinspect.core.document_indexer.Settings")
        self.patcher_splitter = patch("txtinspect.core.document_indexer.SentenceSplitter")

        self.mock_ollama = self.patcher_ollama.start()
        self.mock_embedding = self.patcher_embedding.start()
        self.mock_settings = self.patcher_settings.start()
        self.mock_splitter = self.patcher_splitter.start()

        # Configure mock return values
        self.mock_ollama.return_value = MagicMock()
        self.mock_embedding.return_value = MagicMock()
        self.mock_splitter.return_value = MagicMock()

    def tearDown(self):
        """Clean up test fixtures."""
        # Stop all patchers
        self.patcher_ollama.stop()
        self.patcher_embedding.stop()
        self.patcher_settings.stop()
        self.patcher_splitter.stop()

        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_with_defaults(self):
        """Test DocumentIndexer initialization with default configuration."""
        _ = DocumentIndexer()

        # Verify that Ollama and OllamaEmbedding were called during initialization
        self.mock_ollama.assert_called_once()
        self.mock_embedding.assert_called_once()
        self.mock_splitter.assert_called_once()

    def test_initialization_with_custom_models(self):
        """Test DocumentIndexer initialization with custom model configurations."""
        custom_llm = "custom-llm-model"
        custom_embedding = "custom-embedding-model"
        custom_url = "http://custom-server:8080"

        _ = DocumentIndexer(
            llm_model=custom_llm, embedding_model=custom_embedding, base_url=custom_url
        )

        # Verify Ollama was initialized with custom parameters
        self.mock_ollama.assert_called_once_with(model=custom_llm, base_url=custom_url)

        # Verify OllamaEmbedding was initialized with custom parameters
        self.mock_embedding.assert_called_once_with(
            model_name=custom_embedding, base_url=custom_url
        )

    @patch("txtinspect.core.document_indexer.VectorStoreIndex")
    @patch("txtinspect.core.document_indexer.SimpleDirectoryReader")
    def test_create_index_success(self, mock_reader, mock_vector_store):
        """Test successful index creation with valid document directory."""
        # Set up mocks
        mock_documents = [MagicMock(text="Document 1"), MagicMock(text="Document 2")]
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = mock_documents
        mock_reader.return_value = mock_reader_instance

        mock_index = MagicMock()
        mock_vector_store.from_documents.return_value = mock_index

        # Create indexer and generate index
        indexer = DocumentIndexer()
        result = indexer.create_index(self.temp_dir)

        # Verify SimpleDirectoryReader was called with correct path
        mock_reader.assert_called_once_with(self.temp_dir)
        mock_reader_instance.load_data.assert_called_once()

        # Verify VectorStoreIndex was created from documents with correct parameters
        call_args = mock_vector_store.from_documents.call_args
        self.assertEqual(call_args[0][0], mock_documents)
        self.assertIn("show_progress", call_args[1])
        self.assertIn("transformations", call_args[1])

        # Verify the returned index matches the mock
        self.assertEqual(result, mock_index)

    def test_create_index_nonexistent_directory(self):
        """Test that ValueError is raised when source directory does not exist."""
        nonexistent_path = "/path/that/does/not/exist"

        indexer = DocumentIndexer()

        with self.assertRaises(ValueError) as context:
            indexer.create_index(nonexistent_path)

        # Verify error message contains the path
        self.assertIn(nonexistent_path, str(context.exception))
        self.assertIn("does not exist or is not a directory", str(context.exception))

    @patch("txtinspect.core.document_indexer.VectorStoreIndex")
    @patch("txtinspect.core.document_indexer.SimpleDirectoryReader")
    def test_create_index_empty_directory(self, mock_reader, mock_vector_store):
        """Test index creation with an empty directory."""
        # Create empty directory
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()

        # Mock empty document list
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = []
        mock_reader.return_value = mock_reader_instance

        mock_index = MagicMock()
        mock_vector_store.from_documents.return_value = mock_index

        indexer = DocumentIndexer()
        result = indexer.create_index(str(empty_dir))

        # Verify VectorStoreIndex was still called even with empty documents
        call_args = mock_vector_store.from_documents.call_args
        self.assertEqual(call_args[0][0], [])
        self.assertIn("show_progress", call_args[1])
        self.assertEqual(result, mock_index)

    @patch("txtinspect.core.document_indexer.VectorStoreIndex")
    @patch("txtinspect.core.document_indexer.SimpleDirectoryReader")
    def test_create_index_reader_exception(self, mock_reader, mock_vector_store):
        """Test that exceptions from SimpleDirectoryReader are wrapped with context."""
        # Configure mock to raise an exception
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.side_effect = Exception("Failed to load documents")
        mock_reader.return_value = mock_reader_instance

        indexer = DocumentIndexer()

        with self.assertRaises(Exception) as context:
            indexer.create_index(self.temp_dir)

        # Check for wrapped exception message
        self.assertIn("Failed to load documents from source directory", str(context.exception))

    @patch("txtinspect.core.document_indexer.VectorStoreIndex")
    @patch("txtinspect.core.document_indexer.SimpleDirectoryReader")
    def test_create_index_vectorstore_exception(self, mock_reader, mock_vector_store):
        """Test that exceptions from VectorStoreIndex are wrapped with context."""
        # Set up reader mock to succeed
        mock_documents = [MagicMock(text="Document 1")]
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = mock_documents
        mock_reader.return_value = mock_reader_instance

        # Configure VectorStoreIndex to raise an exception
        mock_vector_store.from_documents.side_effect = Exception("Indexing failed")

        indexer = DocumentIndexer()

        with self.assertRaises(Exception) as context:
            indexer.create_index(self.temp_dir)

        self.assertIn("Failed to create vector", str(context.exception))

    @patch("txtinspect.core.document_indexer.VectorStoreIndex")
    @patch("txtinspect.core.document_indexer.SimpleDirectoryReader")
    def test_create_index_ollama_timeout(self, mock_reader, mock_vector_store):
        """Test handling of timeout when connecting to Ollama service."""
        # Set up reader mock to succeed
        mock_documents = [MagicMock(text="Document 1")]
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = mock_documents
        mock_reader.return_value = mock_reader_instance

        # Simulate timeout error during indexing
        mock_vector_store.from_documents.side_effect = TimeoutError("Ollama service timeout")

        indexer = DocumentIndexer()

        with self.assertRaises(Exception) as context:
            indexer.create_index(self.temp_dir)

        self.assertIn("Failed to create vector store index", str(context.exception))

    @patch("txtinspect.core.document_indexer.VectorStoreIndex")
    @patch("txtinspect.core.document_indexer.SimpleDirectoryReader")
    def test_create_index_with_file_path(self, mock_reader, mock_vector_store):
        """Test create_index with a direct file path instead of directory."""
        # Set up mocks
        mock_documents = [MagicMock(text="Document content")]
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = mock_documents
        mock_reader.return_value = mock_reader_instance

        mock_index = MagicMock()
        mock_vector_store.from_documents.return_value = mock_index

        indexer = DocumentIndexer()
        with self.assertRaises(ValueError) as context:
            _ = indexer.create_index(str(self.test_file))

        self.assertIn("Argument source is invalid", str(context.exception))

    def test_settings_configuration(self):
        """Test that Settings are properly configured during initialization."""
        custom_llm = "test-llm"
        custom_embedding = "test-embedding"
        custom_url = "http://test:9999"

        _ = DocumentIndexer(
            llm_model=custom_llm, embedding_model=custom_embedding, base_url=custom_url
        )

        # Verify Settings.llm was assigned
        self.mock_ollama.assert_called_once()

        # Verify Settings.embed_model was assigned
        self.mock_embedding.assert_called_once()

    def test_initialization_with_custom_chunk_params(self):
        """Test DocumentIndexer initialization with custom chunk parameters."""
        custom_chunk_size = 1024
        custom_chunk_overlap = 50

        _ = DocumentIndexer(chunk_size=custom_chunk_size, chunk_overlap=custom_chunk_overlap)

        # Verify SentenceSplitter was initialized with custom parameters
        self.mock_splitter.assert_called_once_with(
            chunk_size=custom_chunk_size, chunk_overlap=custom_chunk_overlap
        )

    def test_initialization_with_show_progress(self):
        """Test DocumentIndexer initialization with show_progress enabled."""
        indexer = DocumentIndexer(show_progress=True)

        # Verify show_progress is stored
        self.assertTrue(indexer.show_progress)

    def test_create_index_empty_source(self):
        """Test that ValueError is raised when source is empty string."""
        indexer = DocumentIndexer()

        with self.assertRaises(ValueError) as context:
            indexer.create_index("")

        self.assertIn("Argument source is invalid", str(context.exception))

    def test_create_index_none_source(self):
        """Test that ValueError is raised when source is None."""
        indexer = DocumentIndexer()

        with self.assertRaises(ValueError) as context:
            indexer.create_index(None)

        self.assertIn("Argument source is invalid", str(context.exception))

    def test_create_index_whitespace_source(self):
        """Test that ValueError is raised when source is whitespace only."""
        indexer = DocumentIndexer()

        with self.assertRaises(ValueError) as context:
            indexer.create_index("   ")

        self.assertIn("Argument source is invalid", str(context.exception))


if __name__ == "__main__":
    unittest.main()
