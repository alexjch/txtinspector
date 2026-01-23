"""Command routing implementation for the txtinspect CLI application."""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text Inspector - A RAG application for document Q&A"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load documents into the vector store")
    load_parser.add_argument("--source", required=True, help="File or directory to load")
    load_parser.add_argument("--chunk-size", type=int, default=0, help="Text chunk size")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask questions about loaded documents")
    query_parser.add_argument("--question", required=True, help="Question to ask")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve",
    )
    query_parser.add_argument("--model", default="llama3", help="LLM model to use")

    # Index command
    index_parser = subparsers.add_parser("index", help="Manage the vector database")
    index_subparsers = index_parser.add_subparsers(
        dest="index_command", help="Index management commands"
    )
    index_subparsers.add_parser("list", help="Show indexed documents")
    index_subparsers.add_parser("clear", help="Reset the index")
    index_subparsers.add_parser("stats", help="Display index statistics")

    args = parser.parse_args()

    if not args.command or args.command not in ["load", "query", "index"]:
        parser.print_help()
        sys.exit(1)

    return args


def main() -> None:
    """Main function to handle CLI arguments and route commands."""
    args = parse_args()

    # Route to appropriate command handler
    if args.command == "load":
        from txtinspect.commands.load import load_command

        load_command(args)
    elif args.command == "query":
        from txtinspect.commands.query import query_command

        query_command(args)
    elif args.command == "index":
        from txtinspect.commands.index import index_command

        index_command(args)


if __name__ == "__main__":
    main()
