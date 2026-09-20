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

Score retrieval against the labelled set:

```bash
python cli.py eval                 # metrics + any misses
python cli.py eval --sweep         # compare MMR lambda values
python cli.py eval --verbose       # per-question ranks
```

Run the chunking tests:

```bash
python tests/test_chunking.py
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
4. **Retrieval** (`rag/store.py`) — pulls `top_k * 5` candidates, then re-ranks
   them with maximal marginal relevance (MMR) before handing back `top_k`.
   Each step picks the candidate maximising
   `λ·sim(query, chunk) − (1−λ)·max sim(chunk, already_selected)`, so a chunk
   near-identical to one already chosen must be substantially more relevant to
   earn a slot.
5. **Pipeline** (`rag/pipeline.py`) — asks Claude to answer using only the
   retrieved context, citing post titles and URLs.

## Retrieval diversity

Plain top-k similarity lets one post monopolise every slot, because adjacent
chunks of the same article are near-duplicates of each other. MMR fixes this
without a per-source quota, which would have damaged narrow questions whose
answer genuinely lives in a single post. Measured over the dev.to corpus at
`top_k=5`:

| Question | Distinct posts, λ=1.0 | Distinct posts, λ=0.6 |
| --- | --- | --- |
| How did my SAST scanner and secrets detector differ? | 2/5 | 5/5 |
| What tradeoffs did I make across all my security tools? | 4/5 | 5/5 |
| What did I learn about false positives and suppression? | 1/5 | 3/5 |
| Why Random Forest instead of deep learning? *(narrow)* | 1/5 | 1/5 |
| Why did I split chunks on paragraph boundaries? *(narrow)* | 1/5 | 1/5 |

Comparative questions gain breadth; narrow ones correctly stay concentrated in
the one post that answers them. Tune per query with `--lambda`:

```bash
python cli.py query "How do my tools compare?" --lambda 0.4   # more diverse
python cli.py query "Why Random Forest?" --lambda 1.0          # pure relevance
```

## Evaluation

`eval/questions.json` holds 26 labelled questions, each mapped to the post(s)
that genuinely answer it. The questions are deliberately paraphrased rather than
echoing post titles, so the metrics measure retrieval rather than title matching.

Current baseline (`top_k=5`, `lambda=0.6`, 535 chunks):

| Metric | Value |
| --- | --- |
| hit@1 | 0.462 |
| recall@5 | 0.885 |
| MRR | 0.625 |
| distinct posts per result set | 3.85 / 5 |

Sweeping the MMR lambda:

| lambda | hit@1 | recall@5 | MRR | distinct |
| --- | --- | --- | --- | --- |
| 0.2 | 0.462 | 0.885 | 0.618 | 4.23 |
| 0.4 | 0.462 | 0.846 | 0.612 | 4.15 |
| 0.6 | 0.462 | 0.885 | 0.625 | 3.85 |
| 0.8 | 0.462 | 0.885 | **0.639** | 3.15 |
| 1.0 | 0.462 | 0.885 | 0.619 | 2.54 |

Two things to read from this. `hit@1` is identical at every lambda, which is
expected and a useful correctness check: MMR's first pick is always pure
relevance, so re-ranking cannot change the top result. And diversity is close to
free — pushing distinct posts from 2.54 to 3.85 costs nothing measurable in
recall.

The MRR spread (0.612–0.639) is too small to call a winner across 26 questions.
The deeper limitation is that **every labelled question here is a narrow one**,
answerable from a single post, so this set cannot measure the case MMR exists
for. Settling the lambda properly needs comparative questions with multi-post
ground truth.

Known misses at the current baseline:

- *"revisited a codebase I had not touched in years"* retrieves the "13 years of
  software engineering" post — surface-level year matching beating topic.
- *"keep a scanner from drowning developers in noise"* retrieves the adoption
  post, which is arguably a correct answer the label is too strict to accept.
- *"encode a well-known vulnerability list as configuration"* misses the OWASP
  YAML post entirely. The question deliberately avoids the words "OWASP" and
  "YAML", so keyword search would not rescue it either; this is a genuine
  semantic gap and a case for a stronger embedding model.

## Notes and next steps

- **Retrieved text is data, not instructions.** The system prompt tells the model
  to treat context as reference material only. Anything ingested from a remote
  source should be regarded as untrusted input.
- **Hybrid retrieval is the next structural gap.** Dense embeddings cannot
  represent opaque identifiers. `AES-GCM` and `AES-ECB` retrieve their post at
  rank 1, but the literal `AKIAIOSFODNN7EXAMPLE7` — present verbatim in exactly
  one post — is not retrieved at all. Adding BM25 alongside the vector search
  would close this for exact terms.
- **Markdown-aware chunking.** Splitting on blank lines tears fenced code
  blocks: 2 of 18 chunks in one post end with unbalanced ``` fences.
- **Incremental ingest.** Every run refetches all 26 posts; `edited_at` from the
  API would let unchanged ones be skipped.
- **Unused metadata.** The dev.to API returns `tag_list`, `published_at` and
  `reading_time_minutes`; none is stored, so filtered queries aren't possible.
- **Swappable pieces.** The vector store only needs `add_documents` / `query`,
  so Pinecone, Weaviate, or pgvector can drop in. Embeddings can move to Voyage
  AI (Anthropic's recommended partner) if you want hosted quality over local
  cost. Chunk sizing lives in `rag/config.py`.
