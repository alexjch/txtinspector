# Text Inspector

A Retrieval-Augmented Generation (RAG) application for loading text documents and performing question-answering tasks.

## Features

- **Document Loading**: Ingest text files (txt, md, pdf) into a vector database
- **Question Answering**: Ask questions about your documents and get AI-generated responses
- **Index Management**: List, clear, and view statistics about your document index

## Pre-reqs
```
# default models
ollama pull nomic-embed-text
ollama pull gemma:2b
```

## Installation

```bash
# Install the package
make install

# Or for development
make dev
```

## Usage

### Load and query documents

Load documents from a directory:

```bash
txtinspect --chunk-size 512 ./documents
```

### Query Documents

Once the document was successfully loaded you can ask questions about the content:

```bash
You: what is this document about?
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

Default values are defined in [config.py](./txtinspect/config.py)


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

## TODO

- [ ] Use pipeline instead of step by step "transformations".
- [ ] Use the async API.
- [ ] Use the streamming API.
- [ ] Expand source argument to grab documents from URL pdf, html, txt.
