"""Index and interactive chat command implementation."""

import argparse
from pathlib import Path
from txtinspect.core.document_indexer import DocumentIndexer
from txtinspect.config import config


def index_and_chat(args: argparse.Namespace) -> None:
    """
    Index documents and start an interactive chat session for querying.

    Args:
        args: Parsed command-line arguments containing:
            - source: Directory path containing documents to index
            - chunk_size: Size of text chunks (optional, default: 512)
            - chunk_overlap: Overlap between chunks (optional, default: 10)
            - progress: Whether to show progress (optional, default: False)
    """
    # Validate required argument
    if not hasattr(args, "source") or args.source is None:
        raise ValueError("Missing required argument: source directory must be provided")

    source_path = Path(args.source)

    # Get chunk parameters with defaults
    chunk_size = getattr(args, "chunk_size", config.DEFAULT_CHUNK_SIZE)
    chunk_overlap = getattr(args, "chunk_overlap", config.DEFAULT_CHUNK_OVERLAP)
    show_progress = getattr(args, "progress", False)

    print("=" * 60)
    print("Text Inspector - RAG Document Q&A")
    print("=" * 60)
    print(f"\nIndexing documents from: {source_path}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print(f"Show progress: {show_progress}\n")

    try:
        # Initialize the document indexer with configuration
        indexer = DocumentIndexer(
            llm_model=config.llm_model,
            embedding_model=config.embedding_model,
            base_url=config.llm_base_url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            show_progress=show_progress,
        )

        # Create the index from documents in the source directory
        index = indexer.create_index(str(source_path))

        print(f"\n✓ Successfully indexed documents from {source_path}")
        print("=" * 60)
        print("\nStarting interactive chat session...")
        print("Type your questions below (or 'exit', 'quit', 'q' to quit)\n")

        # Create query engine from the index
        query_engine = index.as_query_engine()

        # Interactive chat loop
        while True:
            try:
                # Get user input
                question = input("\nYou: ").strip()

                # Check for exit commands
                if question.lower() in ["exit", "quit", "q", ""]:
                    print("\nGoodbye!")
                    break

                # Query the engine
                print("\nAssistant: ", end="", flush=True)
                response = query_engine.query(question)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error processing query: {e}")
                print("Please try again or type 'exit' to quit.")

    except ValueError as e:
        print(f"\n✗ Error: {e}")
        raise
    except TimeoutError as e:
        print(f"\n✗ Timeout: {e}")
        print("Make sure Ollama service is running and accessible")
        raise
    except Exception as e:
        print(f"\n✗ Failed to index documents: {e}")
        raise
