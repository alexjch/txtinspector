"""Load command implementation."""

import argparse
from pathlib import Path
from txtinspect.core.document_loader import DocumentLoader
from txtinspect.core.embedding_service import EmbeddingService
from txtinspect.config import config


def load_command(args: argparse.Namespace) -> None:
    """
    Handle the load command to ingest documents into the vector store.

    Args:
        args: Parsed command-line arguments
    """
    # Validate required arguments are present
    if not hasattr(args, "source") or args.source is None:
        raise ValueError("Missing required argument: source path must be provided")

    # Determine the value of chunk_size. The configuration provides a default
    # that should be superseded only if a non-zero value is provided in the arguments.
    chunk_size = config.chunk_size
    if hasattr(args, "chunk_size") and args.chunk_size > 0:
        chunk_size = args.chunk_size

    source_path = Path(args.source)

    print(f"Loading documents from: {source_path}")
    print(f"Chunk size: {chunk_size}")

    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=config.chunk_overlap)
    # 1. Discover and load files from source_path (txt, md, pdf)
    documents = loader.load(source_path)
    # 2. Load and chunk documents
    _ = loader.chunk(documents)
    # 3. Generate embeddings
    _ = EmbeddingService(model_name=config.embedding_model, llm_base_url=config.llm_base_url)
    # 4. Store in vector database

    print("Loading functionality not yet implemented")
