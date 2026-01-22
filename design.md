## Design Document: Text Inspector RAG Application

### Overview
A Retrieval-Augmented Generation (RAG) application for loading text documents and performing question-answering tasks.

### Architecture

#### Components
1. **Entry Point** (`main.py`)
    - Command-line interface
    - Argument parsing
    - Command routing

2. **Commands Module** (`commands/`)
    - `load.py` - Document ingestion and processing
    - `query.py` - Question answering interface
    - `index.py` - Vector store management

3. **Core Modules**
    - `document_loader.py` - Text file reading and chunking
    - `embedding_service.py` - Text vectorization
    - `vector_store.py` - Storage and retrieval interface
    - `llm_service.py` - Language model integration
    - `config.py` - Configuration management

### Command Specifications

#### `load` Command
- **Purpose**: Ingest text documents into the vector store
- **Arguments**: 
  - `--source`: File or directory (auto detect), file type is derived from file extension (txt, md, pdf).
  - `--chunk-size`: Text chunking parameters

#### `query` Command
- **Purpose**: Ask questions about loaded documents
- **Arguments**:
  - `--question`: Query string
  - `--top-k`: Number of relevant chunks to retrieve
  - `--model`: LLM model selection

#### `index` Command
- **Purpose**: Manage the vector database
- **Subcommands**:
  - `list`: Show indexed documents
  - `clear`: Reset the index
  - `stats`: Display index statistics

### Data Flow
1. User loads documents → Text chunking → Embedding generation → Vector store
2. User queries → Query embedding → Similarity search → Context retrieval → LLM generation → Response

### Technology Stack (Proposed)
- CLI Framework: argparse
- Embeddings: llama-index
- Vector Store: ChromaDB
- LLM: ollama hosted in a different server
- Configuration: TOML

### File Structure
```
txtinspector/
├── .flake8
├── .github/
│   └── copilot-instructions.md
├── .gitignore
├── txtinspect/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── config.py
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   ├── query.py
│   │   └── index.py
│   └── core/
│       ├── __init__.py
│       ├── document_loader.py
│       ├── embedding_service.py
│       ├── vector_store.py
│       └── llm_service.py
├── tests/
│   └── __init__.py
│   └── test_stub.py
├── design.md
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

### Configuration Options
- Embedding model selection
- Vector store backend
- LLM provider and model
- Chunk size and overlap
- Storage location
