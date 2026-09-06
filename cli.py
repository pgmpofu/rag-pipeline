import argparse
from pathlib import Path

from rag import pipeline, store
from rag.loader import load_and_chunk


def cmd_ingest(args):
    documents = load_and_chunk(Path(args.path))
    count = store.add_documents(documents)
    print(f"Ingested {count} chunks from {args.path}")


def cmd_query(args):
    result = pipeline.answer(args.question, top_k=args.top_k)
    print(result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"  - {src['source']} (distance={src['distance']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Local RAG pipeline")
    subparsers = parser.add_subparsers(required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file or directory")
    ingest_parser.add_argument("path", help="Path to a file or directory to ingest")
    ingest_parser.set_defaults(func=cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    query_parser.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
