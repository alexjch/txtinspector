"""Query command implementation."""

import argparse


def query_command(args: argparse.Namespace) -> None:
    """
    Handle the query command to ask questions about loaded documents.

    Args:
        args: Parsed command-line arguments
    """
    question = args.question
    top_k = args.top_k
    model = args.model

    print(f"Question: {question}")
    print(f"Retrieving top {top_k} chunks")
    print(f"Using model: {model}")

    # TODO: Implement query logic
    # 1. Embed the question
    # 2. Perform similarity search
    # 3. Retrieve relevant chunks
    # 4. Generate response using LLM

    print("Query functionality not yet implemented")
