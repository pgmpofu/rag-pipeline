import argparse
from pathlib import Path

from rag import devto, evaluate, pipeline, store
from rag.config import MMR_LAMBDA, TOP_K
from rag.loader import load_and_chunk

LAMBDA_SWEEP = [0.2, 0.4, 0.6, 0.8, 1.0]


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
    result = pipeline.answer(args.question, top_k=args.top_k, lambda_mult=args.lambda_mult)
    print(result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        location = f" — {src['url']}" if src["url"] else ""
        print(f"  - {src['title']}{location} (distance={src['distance']:.4f})")


def cmd_reset(args):
    store.reset()
    print("Cleared the collection")


def cmd_eval(args):
    questions = evaluate.load_questions()
    if args.sweep:
        print(evaluate.format_sweep(evaluate.sweep(questions, LAMBDA_SWEEP, top_k=args.top_k)))
    else:
        report = evaluate.evaluate(questions, top_k=args.top_k, lambda_mult=args.lambda_mult)
        print(evaluate.format_report(report, verbose=args.verbose))


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
    query_parser.add_argument(
        "--lambda",
        dest="lambda_mult",
        type=float,
        default=MMR_LAMBDA,
        help="MMR relevance/diversity tradeoff: 1.0 is pure relevance, lower is more diverse",
    )
    query_parser.set_defaults(func=cmd_query)

    reset_parser = subparsers.add_parser("reset", help="Delete everything in the collection")
    reset_parser.set_defaults(func=cmd_reset)

    eval_parser = subparsers.add_parser("eval", help="Score retrieval against the labelled set")
    eval_parser.add_argument("--top-k", type=int, default=TOP_K)
    eval_parser.add_argument("--lambda", dest="lambda_mult", type=float, default=MMR_LAMBDA)
    eval_parser.add_argument(
        "--sweep", action="store_true", help="Score across a range of lambda values"
    )
    eval_parser.add_argument("--verbose", action="store_true", help="Show every question's rank")
    eval_parser.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
