# Text Inspector

A Retrieval-Augmented Generation (RAG) application for loading text documents and performing question-answering tasks using local LLMs via Ollama.

## Features

- **Document Indexing**: Ingest and index text documents from a directory
- **Interactive Chat**: Ask questions about your documents in an interactive chat session
- **Streaming Responses**: Get real-time streaming responses from the AI
- **Configurable Chunking**: Customize document splitting with chunk size and overlap parameters
- **Progress Tracking**: Optional progress display during document indexing

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

### Basic Usage

Index documents and start an interactive chat session:

```bash
txtinspect ./documents
```

Or with Python module:

```bash
python -m txtinspect ./documents
```

### Advanced Options

Customize chunking parameters and show progress:

```bash
txtinspect --chunk-size 1024 --chunk-overlap 50 --progress ./documents
```

### Interactive Chat

After indexing, you'll enter an interactive chat session:

```
You: what is this document about?
Assistant: [AI response will stream here...]

You: exit
```

Type `exit`, `quit`, or `q` to end the session.

## Configuration

Configuration can be set via environment variables:

- `TXTINSPECT_CHUNK_SIZE`: Text chunk size (default: 512)
- `TXTINSPECT_CHUNK_OVERLAP`: Chunk overlap (default: 10)
- `TXTINSPECT_EMBEDDING_MODEL`: Embedding model name (default: nomic-embed-text)
- `TXTINSPECT_LLM_MODEL`: LLM model name (default: gemma:2b)
- `TXTINSPECT_LLM_BASE_URL`: LLM server URL (default: http://localhost:11434)
- `TXTINSPECT_VECTOR_STORE`: Vector store backend (default: chroma)
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

- [ ] Improve CLI UX
    - [x] ~~Use the async API?~~ There's no good place where this can be added, maybe handling first time model pull?
    - [x] ~~Use the streaming API?~~ Implemented - responses now stream in real-time
    - [ ] Use click for chat?
- [ ] Expand source argument to grab documents from URL (pdf, html, txt)
- [ ] Persist index to disk
- [ ] Test using specific documents (e.g., ToS) where chunking is more delicate than chopping text by length
- [x] ~~Use pipeline instead of step by step "transformations"~~ Best when using an external Vector database. Keeping VectorStoreIndex for simplicity
- [ ] CPU only performance improvements
- [ ] Index documentation
