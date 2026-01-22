"""Load command implementation."""

import argparse
from pathlib import Path


def load_command(args: argparse.Namespace) -> None:
    """
    Handle the load command to ingest documents into the vector store.

    Args:
        args: Parsed command-line arguments
    """
    source_path = Path(args.source)
    chunk_size = args.chunk_size

    print(f"Loading documents from: {source_path}")
    print(f"Chunk size: {chunk_size}")

    if not source_path.exists():
        print(f"Error: Source path '{source_path}' does not exist")
        return

    # TODO: Implement document loading logic
    # 1. Discover files (txt, md, pdf)
    # 2. Load and chunk documents
    # 3. Generate embeddings
    # 4. Store in vector database

    print("Loading functionality not yet implemented")
