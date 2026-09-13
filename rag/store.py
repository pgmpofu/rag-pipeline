import hashlib

import chromadb
from chromadb.utils import embedding_functions

from .config import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR


def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )


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


def query(text: str, top_k: int) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[text], n_results=top_k)

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta["source"],
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "distance": distance,
            }
        )
    return hits


def count() -> int:
    return get_collection().count()


def reset():
    chromadb.PersistentClient(path=PERSIST_DIR).delete_collection(COLLECTION_NAME)
