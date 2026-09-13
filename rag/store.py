import hashlib

import chromadb
import numpy as np
from chromadb.utils import embedding_functions

from .config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    FETCH_K_MULTIPLIER,
    MMR_LAMBDA,
    PERSIST_DIR,
)

_embedding_fn = None
_collection = None


def _embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=_embedding_function()
        )
    return _collection


def _doc_id(source: str, chunk_index: int) -> str:
    return hashlib.sha1(f"{source}:{chunk_index}".encode()).hexdigest()


def add_documents(documents: list[dict]) -> int:
    if not documents:
        return 0

    collection = get_collection()
    collection.upsert(
        ids=[_doc_id(d["source"], d["chunk_index"]) for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[
            {
                "source": d["source"],
                "chunk_index": d["chunk_index"],
                "title": d.get("title", ""),
                "url": d.get("url", ""),
            }
            for d in documents
        ],
    )
    return len(documents)


def _mmr_select(
    query_embedding: np.ndarray,
    candidates: np.ndarray,
    top_k: int,
    lambda_mult: float,
) -> list[int]:
    """Maximal marginal relevance: pick chunks that are relevant to the query but
    not redundant with each other.

    Each step picks the candidate maximising
        lambda * sim(query, candidate) - (1 - lambda) * max sim(candidate, already_picked)
    so a chunk near-identical to one already chosen has to be substantially more
    relevant to earn a slot. Embeddings are L2-normalised, so a dot product is
    the cosine similarity.
    """
    # numpy's Accelerate BLAS backend on macOS raises spurious divide-by-zero,
    # overflow and invalid flags from matmul even on clean finite input, so the
    # warnings are suppressed here and the result is checked explicitly instead —
    # a genuine numerical failure still surfaces rather than being swallowed.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        relevance = candidates @ query_embedding
        pairwise = candidates @ candidates.T

    if not (np.isfinite(relevance).all() and np.isfinite(pairwise).all()):
        raise ValueError("Non-finite similarity scores; the stored embeddings look corrupt.")

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best = max(
                remaining,
                key=lambda i: lambda_mult * relevance[i]
                - (1 - lambda_mult) * pairwise[i, selected].max(),
            )
        selected.append(best)
        remaining.remove(best)

    return selected


def query(
    text: str,
    top_k: int,
    lambda_mult: float = MMR_LAMBDA,
) -> list[dict]:
    """Retrieve top_k chunks, MMR-reranked from a wider candidate pool.

    lambda_mult of 1.0 is pure relevance (no diversity); lower values trade
    relevance for variety across the retrieved set.
    """
    collection = get_collection()
    fetch_k = min(top_k * FETCH_K_MULTIPLIER, collection.count())

    results = collection.query(
        query_texts=[text],
        n_results=max(fetch_k, top_k),
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    hits = [
        {
            "text": doc,
            "source": meta["source"],
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "distance": distance,
        }
        for doc, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

    if lambda_mult >= 1.0 or len(hits) <= top_k:
        return hits[:top_k]

    query_embedding = np.asarray(_embedding_function()([text])[0], dtype=np.float32)
    candidates = np.asarray(results["embeddings"][0], dtype=np.float32)

    return [hits[i] for i in _mmr_select(query_embedding, candidates, top_k, lambda_mult)]


def count() -> int:
    return get_collection().count()


def reset():
    global _collection
    chromadb.PersistentClient(path=PERSIST_DIR).delete_collection(COLLECTION_NAME)
    _collection = None
