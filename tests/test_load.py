"""Tests for the load command."""

import sys
import unittest
import argparse
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Mock llama_index before importing load_command
sys.modules["llama_index"] = Mock()
sys.modules["llama_index.core"] = Mock()

from txtinspect.commands.load import load_command  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
