"""Main application entry point for txtinspect RAG system."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text Inspector - A RAG application for document Q&A",
        epilog="After indexing, you'll enter an interactive chat session to query your documents.",
    )

    parser.add_argument("source", help="Directory containing documents to index")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Text chunk size for document splitting (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=10, help="Overlap between text chunks (default: 10)"
    )
    parser.add_argument("--progress", action="store_true", help="Show progress during indexing")

    return parser.parse_args()


def main() -> None:
    """Main function to index documents and start interactive query session."""
    args = parse_args()

    from txtinspect.commands.index import index_and_chat

    index_and_chat(args)


if __name__ == "__main__":
    main()
