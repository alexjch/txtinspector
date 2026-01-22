# Text Inspector

A Retrieval-Augmented Generation (RAG) application for loading text documents and performing question-answering tasks.

## Features

- **Document Loading**: Ingest text files (txt, md, pdf) into a vector database
- **Question Answering**: Ask questions about your documents and get AI-generated responses
- **Index Management**: List, clear, and view statistics about your document index

## Installation

```bash
# Install the package
make install

# Or for development
make dev
```

## Usage

### Load Documents

Load documents from a file or directory:

```bash
txtinspect load --source ./documents --chunk-size 512
```

### Query Documents

Ask questions about loaded documents:

```bash
txtinspect query --question "What is the main topic?" --top-k 5 --model llama3
```

### Manage Index

View indexed documents:

```bash
txtinspect index list
```

View index statistics:

```bash
txtinspect index stats
```

Clear the index:

```bash
txtinspect index clear
```

## Configuration

Configuration can be set via environment variables:

- `TXTINSPECT_CHUNK_SIZE`: Text chunk size (default: 512)
- `TXTINSPECT_CHUNK_OVERLAP`: Chunk overlap (default: 50)
- `TXTINSPECT_EMBEDDING_MODEL`: Embedding model name
- `TXTINSPECT_LLM_MODEL`: LLM model name (default: llama3)
- `TXTINSPECT_LLM_BASE_URL`: LLM server URL (default: http://localhost:11434)
- `TXTINSPECT_PERSIST_DIR`: Vector store directory (default: ./chroma_db)
- `TXTINSPECT_TOP_K`: Number of chunks to retrieve (default: 5)

## Architecture

The application consists of:

- **CLI Interface**: Command-line interface for user interaction
- **Commands**: Load, query, and index management commands
- **Core Modules**: Document loading, embedding, vector storage, and LLM services
- **Configuration**: Centralized configuration management

## Technology Stack

- **CLI Framework**: argparse
- **Embeddings**: llama-index
- **Vector Store**: ChromaDB
- **LLM**: Ollama (hosted separately)
- **Configuration**: Environment variables / TOML

## Development

Run tests:

```bash
make test
```

Format code:

```bash
make format
```

Lint code:

```bash
make lint
```

## License

MIT
