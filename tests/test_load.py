"""Tests for the load command."""

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
sys.modules["llama_index.core.node_parser.text"] = Mock()

from txtinspect.commands.load import load_command  # noqa: E402
from txtinspect.core.document_loader import DocumentLoader, DocumentLoaderException  # noqa: E402


class TestLoadCommand(unittest.TestCase):
    """Test cases for load_command function."""

    def setUp(self):
        """Set up test fixtures."""
        # Creates a temporary directory that must be manually cleaned up
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.txt"
        self.test_file.write_text("This is a test document.")

    def tearDown(self):
        """Clean up test fixtures by removing temporary directory."""
        # Remove the temporary directory and its contents after each test
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_command_missing_source(self):
        """Test that ValueError is raised when source argument is missing."""
        args = argparse.Namespace()
        with self.assertRaises(ValueError) as context:
            load_command(args)
        self.assertIn("source path must be provided", str(context.exception))

    def test_load_command_with_valid_file(self):
        """Test load_command with a valid file path."""
        args = argparse.Namespace(source=str(self.test_file), chunk_size=512)
        with patch("txtinspect.commands.load.DocumentLoader") as mock_loader:
            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            load_command(args)

            mock_loader.assert_called_once()
            mock_loader_instance.load.assert_called_once_with(self.test_file)

    def test_load_command_with_directory(self):
        """Test load_command with a directory path."""
        args = argparse.Namespace(source=self.temp_dir, chunk_size=256)
        with patch("txtinspect.commands.load.DocumentLoader") as mock_loader:
            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            load_command(args)

            mock_loader.assert_called_once()
            mock_loader_instance.load.assert_called_once_with(Path(self.temp_dir))

    def test_load_command_uses_config_chunk_size(self):
        """Test that load_command uses config chunk size when not overridden."""
        args = argparse.Namespace(source=str(self.test_file))
        with (
            patch("txtinspect.commands.load.DocumentLoader") as mock_loader,
            patch("txtinspect.commands.load.config") as mock_config,
        ):
            mock_config.chunk_size = 1024
            mock_config.chunk_overlap = 100
            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            load_command(args)

            mock_loader.assert_called_once_with(chunk_size=1024, chunk_overlap=100)

    def test_load_command_overrides_chunk_size(self):
        """Test that load_command overrides config chunk size when provided."""
        args = argparse.Namespace(source=str(self.test_file), chunk_size=2048)
        with (
            patch("txtinspect.commands.load.DocumentLoader") as mock_loader,
            patch("txtinspect.commands.load.config") as mock_config,
        ):
            mock_config.chunk_size = 512
            mock_config.chunk_overlap = 50
            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            load_command(args)

            mock_loader.assert_called_once_with(chunk_size=2048, chunk_overlap=50)

    def test_load_command_prints_status(self):
        """Test that load_command prints loading status."""
        args = argparse.Namespace(source=str(self.test_file), chunk_size=512)
        with (
            patch("txtinspect.commands.load.DocumentLoader"),
            patch("builtins.print") as mock_print,
        ):
            load_command(args)

            # Check that print was called with expected messages
            calls = [str(call) for call in mock_print.call_args_list]
            self.assertTrue(any("Loading documents from:" in str(call) for call in calls))
            self.assertTrue(any("Chunk size:" in str(call) for call in calls))


class TestDocumentLoader(unittest.TestCase):
    """Test cases for DocumentLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_txt_file = Path(self.temp_dir) / "test.txt"
        self.test_txt_file.write_text("This is a test document with some content.")
        self.test_md_file = Path(self.temp_dir) / "test.md"
        self.test_md_file.write_text("# Test Markdown\n\nThis is markdown content.")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_loader_initialization(self):
        """Test DocumentLoader initializes with correct parameters."""
        loader = DocumentLoader(chunk_size=1024, chunk_overlap=100)
        self.assertIsNotNone(loader.splitter)

    def test_loader_default_initialization(self):
        """Test DocumentLoader initializes with default parameters."""
        loader = DocumentLoader()
        self.assertIsNotNone(loader.splitter)

    @patch("txtinspect.core.document_loader.SimpleDirectoryReader")
    def test_load_single_file(self, mock_reader):
        """Test loading a single file returns documents."""
        mock_reader_instance = MagicMock()
        mock_doc = MagicMock()
        mock_doc.text = "Sample document text"
        mock_reader_instance.load_data.return_value = [mock_doc]
        mock_reader.return_value = mock_reader_instance

        loader = DocumentLoader()
        documents = loader.load(self.test_txt_file)

        mock_reader.assert_called_once_with(input_files=[self.test_txt_file])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, "Sample document text")

    @patch("txtinspect.core.document_loader.SimpleDirectoryReader")
    def test_load_directory(self, mock_reader):
        """Test loading files from a directory returns documents."""
        mock_reader_instance = MagicMock()
        mock_docs = [MagicMock(), MagicMock()]
        mock_reader_instance.load_data.return_value = mock_docs
        mock_reader.return_value = mock_reader_instance

        loader = DocumentLoader()
        documents = loader.load(Path(self.temp_dir))

        mock_reader.assert_called_once_with(input_dir=Path(self.temp_dir))
        self.assertEqual(len(documents), 2)

    def test_load_nonexistent_path(self):
        """Test that loading nonexistent path raises exception."""
        loader = DocumentLoader()
        nonexistent_path = Path(self.temp_dir) / "nonexistent.txt"

        with self.assertRaises(DocumentLoaderException) as context:
            loader.load(nonexistent_path)

        self.assertIn("Source path does not exist", str(context.exception))
        self.assertEqual(context.exception.path, nonexistent_path)

    @patch("txtinspect.core.document_loader.SimpleDirectoryReader")
    def test_load_multiple_calls_return_separate_documents(self, mock_reader):
        """Test that multiple load calls return separate document lists."""
        mock_reader_instance = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.text = "Document 1"
        mock_doc2 = MagicMock()
        mock_doc2.text = "Document 2"

        loader = DocumentLoader()

        # First load
        mock_reader_instance.load_data.return_value = [mock_doc1]
        mock_reader.return_value = mock_reader_instance
        documents1 = loader.load(self.test_txt_file)

        # Second load
        mock_reader_instance.load_data.return_value = [mock_doc2]
        documents2 = loader.load(self.test_md_file)

        self.assertEqual(len(documents1), 1)
        self.assertEqual(len(documents2), 1)
        self.assertEqual(documents1[0].text, "Document 1")
        self.assertEqual(documents2[0].text, "Document 2")

    @patch("txtinspect.core.document_loader.TokenTextSplitter")
    @patch("txtinspect.core.document_loader.SimpleDirectoryReader")
    def test_chunk_with_documents(self, mock_reader, mock_splitter):
        """Test chunking text from documents."""
        # Setup mock documents
        mock_reader_instance = MagicMock()
        mock_doc = MagicMock()
        mock_doc.text = "This is a test document with enough text to be chunked."
        mock_reader_instance.load_data.return_value = [mock_doc]
        mock_reader.return_value = mock_reader_instance

        # Setup mock splitter
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text = MagicMock(return_value=["chunk1", "chunk2"])
        mock_splitter.return_value = mock_splitter_instance

        loader = DocumentLoader(chunk_size=512, chunk_overlap=50)
        documents = loader.load(self.test_txt_file)
        chunks = loader.chunk(documents)

        self.assertEqual(len(chunks), 2)
        self.assertIn("chunk1", chunks)
        self.assertIn("chunk2", chunks)

    def test_chunk_without_documents(self):
        """Test that chunking with empty document list raises ValueError."""
        loader = DocumentLoader()

        with self.assertRaises(ValueError) as context:
            loader.chunk([])

        self.assertIn("No documents", str(context.exception))

    @patch("txtinspect.core.document_loader.TokenTextSplitter")
    @patch("txtinspect.core.document_loader.SimpleDirectoryReader")
    def test_chunk_multiple_documents(self, mock_reader, mock_splitter):
        """Test chunking text from multiple documents."""
        # Setup mock documents
        mock_reader_instance = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.text = "First document text"
        mock_doc2 = MagicMock()
        mock_doc2.text = "Second document text"
        mock_reader_instance.load_data.return_value = [mock_doc1, mock_doc2]
        mock_reader.return_value = mock_reader_instance

        # Setup mock splitter to return different chunks for each call
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text = MagicMock(
            side_effect=[["chunk1", "chunk2"], ["chunk3"]]
        )
        mock_splitter.return_value = mock_splitter_instance

        loader = DocumentLoader()
        documents = loader.load(Path(self.temp_dir))
        chunks = loader.chunk(documents)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(mock_splitter_instance.split_text.call_count, 2)

    def test_document_loader_exception_attributes(self):
        """Test DocumentLoaderException stores message and path correctly."""
        test_path = Path("/test/path")
        exc = DocumentLoaderException("Test error", test_path)

        self.assertEqual(exc.message, "Test error")
        self.assertEqual(exc.path, test_path)
        self.assertIn("Test error", str(exc))
        self.assertIn(str(test_path), str(exc))


if __name__ == "__main__":
    unittest.main()
