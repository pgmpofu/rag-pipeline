import argparse
from pathlib import Path

from rag import devto, pipeline, store
from rag.loader import load_and_chunk


def cmd_ingest(args):
    documents = load_and_chunk(Path(args.path))
    count = store.add_documents(documents)
    print(f"Ingested {count} chunks from {args.path}")


def cmd_ingest_devto(args):
    documents = devto.load_and_chunk(args.username)
    articles = len({d["source"] for d in documents})
    count = store.add_documents(documents)
    print(f"Ingested {count} chunks from {articles} dev.to posts by {args.username}")


def cmd_query(args):
    result = pipeline.answer(args.question, top_k=args.top_k)
    print(result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        location = f" — {src['url']}" if src["url"] else ""
        print(f"  - {src['title']}{location} (distance={src['distance']:.4f})")


def cmd_reset(args):
    store.reset()
    print("Cleared the collection")


def main():
    parser = argparse.ArgumentParser(description="Local RAG pipeline")
    subparsers = parser.add_subparsers(required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file or directory")
    ingest_parser.add_argument("path", help="Path to a file or directory to ingest")
    ingest_parser.set_defaults(func=cmd_ingest)

    devto_parser = subparsers.add_parser("ingest-devto", help="Ingest a dev.to author's posts")
    devto_parser.add_argument("username", help="dev.to username, e.g. pgmpofu")
    devto_parser.set_defaults(func=cmd_ingest_devto)

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    query_parser.set_defaults(func=cmd_query)

    reset_parser = subparsers.add_parser("reset", help="Delete everything in the collection")
    reset_parser.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
