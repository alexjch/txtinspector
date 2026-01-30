"""Tests for the index command."""

import sys
import unittest
import argparse
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Mock llama_index before importing
sys.modules["llama_index"] = Mock()
sys.modules["llama_index.core"] = Mock()
sys.modules["llama_index.core.node_parser"] = Mock()
sys.modules["llama_index.embeddings"] = Mock()
sys.modules["llama_index.embeddings.ollama"] = Mock()
sys.modules["llama_index.llms"] = Mock()
sys.modules["llama_index.llms.ollama"] = Mock()

from txtinspect.commands.index import index_and_chat  # noqa: E402


class TestIndexCommand(unittest.TestCase):
    """Test cases for index_and_chat function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory with test documents
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.txt"
        self.test_file.write_text("This is a test document.")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_command_missing_source(self):
        """Test that ValueError is raised when source argument is missing."""
        args = argparse.Namespace()
        with self.assertRaises(ValueError) as context:
            index_and_chat(args)
        self.assertIn("source directory must be provided", str(context.exception))

    @patch("builtins.input", side_effect=["exit"])
    @patch("txtinspect.commands.index.DocumentIndexer")
    def test_index_command_with_defaults(self, mock_indexer_class, mock_input):
        """Test index_and_chat with default parameters."""
        args = argparse.Namespace(
            source=self.temp_dir, chunk_size=512, chunk_overlap=10, progress=False
        )

        mock_indexer = MagicMock()
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_indexer.create_index.return_value = mock_index
        mock_indexer_class.return_value = mock_indexer

        index_and_chat(args)

        # Verify DocumentIndexer was instantiated
        mock_indexer_class.assert_called_once()

        # Verify create_index was called with source path
        mock_indexer.create_index.assert_called_once_with(self.temp_dir)

        # Verify query engine was created
        mock_index.as_query_engine.assert_called_once()

    @patch("builtins.input", side_effect=["exit"])
    @patch("txtinspect.commands.index.DocumentIndexer")
    def test_index_command_with_custom_params(self, mock_indexer_class, mock_input):
        """Test index_and_chat with custom chunk parameters."""
        custom_chunk_size = 1024
        custom_chunk_overlap = 50

        args = argparse.Namespace(
            source=self.temp_dir,
            chunk_size=custom_chunk_size,
            chunk_overlap=custom_chunk_overlap,
            progress=True,
        )

        mock_indexer = MagicMock()
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_indexer.create_index.return_value = mock_index
        mock_indexer_class.return_value = mock_indexer

        index_and_chat(args)

        # Verify DocumentIndexer was initialized with custom parameters
        call_kwargs = mock_indexer_class.call_args[1]
        self.assertEqual(call_kwargs["chunk_size"], custom_chunk_size)
        self.assertEqual(call_kwargs["chunk_overlap"], custom_chunk_overlap)
        self.assertTrue(call_kwargs["show_progress"])

    @patch("txtinspect.commands.index.DocumentIndexer")
    def test_index_command_invalid_directory(self, mock_indexer_class):
        """Test that ValueError is properly propagated for invalid directory."""
        args = argparse.Namespace(
            source="/nonexistent/path", chunk_size=512, chunk_overlap=10, progress=False
        )

        mock_indexer = MagicMock()
        mock_indexer.create_index.side_effect = ValueError("Directory does not exist")
        mock_indexer_class.return_value = mock_indexer

        with self.assertRaises(ValueError):
            index_and_chat(args)

    @patch("txtinspect.commands.index.DocumentIndexer")
    def test_index_command_timeout_error(self, mock_indexer_class):
        """Test handling of TimeoutError from Ollama service."""
        args = argparse.Namespace(
            source=self.temp_dir, chunk_size=512, chunk_overlap=10, progress=False
        )

        mock_indexer = MagicMock()
        mock_indexer.create_index.side_effect = TimeoutError("Ollama service timeout")
        mock_indexer_class.return_value = mock_indexer

        with self.assertRaises(TimeoutError):
            index_and_chat(args)

    @patch("txtinspect.commands.index.DocumentIndexer")
    def test_index_command_general_exception(self, mock_indexer_class):
        """Test handling of general exceptions during indexing."""
        args = argparse.Namespace(
            source=self.temp_dir, chunk_size=512, chunk_overlap=10, progress=False
        )

        mock_indexer = MagicMock()
        mock_indexer.create_index.side_effect = Exception("Indexing failed")
        mock_indexer_class.return_value = mock_indexer

        with self.assertRaises(Exception) as context:
            index_and_chat(args)

        self.assertIn("Indexing failed", str(context.exception))

    @patch("builtins.input", side_effect=["exit"])
    @patch("txtinspect.commands.index.DocumentIndexer")
    def test_index_command_with_progress_flag(self, mock_indexer_class, mock_input):
        """Test that progress flag is properly passed to DocumentIndexer."""
        args = argparse.Namespace(
            source=self.temp_dir, chunk_size=512, chunk_overlap=10, progress=True
        )

        mock_indexer = MagicMock()
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_indexer.create_index.return_value = mock_index
        mock_indexer_class.return_value = mock_indexer

        index_and_chat(args)

        # Verify show_progress was set to True
        call_kwargs = mock_indexer_class.call_args[1]
        self.assertTrue(call_kwargs["show_progress"])


if __name__ == "__main__":
    unittest.main()
