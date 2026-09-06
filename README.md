# RAG Pipeline

A local retrieval-augmented generation pipeline: ingest documents into a local
Chroma vector store using local sentence-transformer embeddings, then answer
questions with Claude grounded in the retrieved context.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in ANTHROPIC_API_KEY
```

## Usage

Ingest a file or a directory of `.txt`, `.md`, or `.pdf` files:

```bash
python cli.py ingest data/
```

Ask a question:

```bash
python cli.py query "What does this document say about X?"
```

## How it works

1. **Loader** (`rag/loader.py`) reads files and splits them into overlapping
   chunks on paragraph boundaries.
2. **Store** (`rag/store.py`) embeds chunks locally with
   `sentence-transformers/all-MiniLM-L6-v2` and persists them in a local
   Chroma collection (`chroma_db/`).
3. **Pipeline** (`rag/pipeline.py`) embeds the question, retrieves the
   top-k most similar chunks, and asks Claude to answer using only that
   context, citing sources.

## Swapping components

- **Vector store**: replace `rag/store.py` with a client for Pinecone,
  Weaviate, pgvector, etc. — it only needs `add_documents` and `query`.
- **Embeddings**: swap the `SentenceTransformerEmbeddingFunction` for
  Voyage AI (Anthropic's recommended embeddings partner) or another provider.
- **Chunking**: tune `CHUNK_SIZE` / `CHUNK_OVERLAP` in `rag/config.py`, or
  replace the paragraph-based splitter in `rag/loader.py` with a
  token-aware one for large-scale ingestion.
