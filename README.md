# RAG Pipeline

A local retrieval-augmented generation pipeline built over a real corpus: the
dev.to writing of [@pgmpofu](https://dev.to/pgmpofu) — 26 posts on AppSec, SAST
tooling, ML-based secrets detection, and RAG design.

Documents are chunked, embedded locally with sentence-transformers, and stored
in a local Chroma collection. Claude answers questions grounded in the retrieved
chunks and cites the posts it drew from.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in ANTHROPIC_API_KEY
```

## Usage

Ingest the dev.to corpus:

```bash
python cli.py ingest-devto pgmpofu
```

Ingest local `.txt`, `.md`, or `.pdf` files as well:

```bash
python cli.py ingest data/
```

Ask a question:

```bash
python cli.py query "Why did I choose Random Forest over deep learning?"
```

Clear the collection:

```bash
python cli.py reset
```

## How it works

1. **Sources** — `rag/devto.py` pulls posts from the public dev.to API
   (`body_markdown`), and `rag/loader.py` reads local files. Both emit the same
   chunk shape, so the rest of the pipeline doesn't care where text came from.
2. **Chunking** — splits on paragraph boundaries up to ~800 chars with overlap.
   dev.to chunks are prefixed with the post title so a standalone chunk stays
   retrievable by topic even when its body never restates the subject.
3. **Store** (`rag/store.py`) — embeds chunks locally with
   `sentence-transformers/all-MiniLM-L6-v2` and upserts them into a persistent
   Chroma collection (`chroma_db/`). Chunk IDs are derived from
   source + index, so re-ingesting updates rather than duplicates.
4. **Pipeline** (`rag/pipeline.py`) — retrieves the top-k chunks and asks Claude
   to answer using only that context, citing post titles and URLs.

## Notes and next steps

- **Retrieved text is data, not instructions.** The system prompt tells the model
  to treat context as reference material only. Anything ingested from a remote
  source should be regarded as untrusted input.
- **Retrieval currently favours a single post.** For a narrow question that is
  correct, but comparative questions ("how did my SAST and secrets tools differ?")
  can have all k slots taken by one article. Per-source diversity or MMR
  re-ranking would fix this.
- **Swappable pieces.** The vector store only needs `add_documents` / `query`,
  so Pinecone, Weaviate, or pgvector can drop in. Embeddings can move to Voyage
  AI (Anthropic's recommended partner) if you want hosted quality over local
  cost. Chunk sizing lives in `rag/config.py`.
