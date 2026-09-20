"""Retrieval evaluation.

Without this, tuning is guesswork: you change the chunk size or the MMR lambda,
eyeball a couple of queries, and convince yourself it improved. These metrics
make a change either measurably better or measurably worse.

Metrics, over a labelled set of question -> acceptable post(s):
  hit@1     fraction whose first result is a correct post
  recall@k  fraction with a correct post anywhere in the top k
  MRR       mean reciprocal rank of the first correct post (misses score 0)
  distinct  mean number of distinct posts per result set (diversity, not accuracy)
"""

from __future__ import annotations

import json
from pathlib import Path

from . import store
from .config import MMR_LAMBDA, TOP_K

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "questions.json"


def load_questions(path: Path | None = None) -> list[dict]:
    return json.loads(Path(path or QUESTIONS_PATH).read_text(encoding="utf-8"))


def evaluate(
    questions: list[dict],
    top_k: int = TOP_K,
    lambda_mult: float = MMR_LAMBDA,
) -> dict:
    results = []
    for item in questions:
        hits = store.query(item["question"], top_k=top_k, lambda_mult=lambda_mult)
        titles = [hit["title"] for hit in hits]
        expected = set(item["expected"])
        rank = next((i + 1 for i, t in enumerate(titles) if t in expected), None)

        results.append(
            {
                "question": item["question"],
                "expected": item["expected"],
                "titles": titles,
                "rank": rank,
                "distinct": len(set(titles)),
            }
        )

    total = len(results) or 1
    return {
        "questions": len(results),
        "top_k": top_k,
        "lambda": lambda_mult,
        "hit@1": sum(r["rank"] == 1 for r in results) / total,
        "recall": sum(r["rank"] is not None for r in results) / total,
        "mrr": sum(1 / r["rank"] for r in results if r["rank"]) / total,
        "distinct": sum(r["distinct"] for r in results) / total,
        "results": results,
    }


def sweep(
    questions: list[dict],
    lambdas: list[float],
    top_k: int = TOP_K,
) -> list[dict]:
    return [evaluate(questions, top_k=top_k, lambda_mult=lam) for lam in lambdas]


def format_report(report: dict, verbose: bool = False) -> str:
    lines = [
        f"{report['questions']} questions | top_k={report['top_k']} | lambda={report['lambda']}",
        f"  hit@1            {report['hit@1']:.3f}",
        f"  recall@{report['top_k']}         {report['recall']:.3f}",
        f"  MRR              {report['mrr']:.3f}",
        f"  distinct posts   {report['distinct']:.2f} of {report['top_k']}",
    ]

    misses = [r for r in report["results"] if r["rank"] is None]
    if misses:
        lines.append(f"\n  {len(misses)} miss(es):")
        for miss in misses:
            lines.append(f"    - {miss['question']}")
            lines.append(f"        wanted: {miss['expected'][0]}")
            lines.append(f"        got   : {miss['titles'][0]}")

    if verbose:
        lines.append("\n  per-question rank:")
        for r in report["results"]:
            rank = r["rank"] or "-"
            lines.append(f"    {str(rank):>3}  {r['question'][:68]}")

    return "\n".join(lines)


def format_sweep(reports: list[dict]) -> str:
    lines = [
        "  lambda   hit@1   recall    MRR   distinct",
        "  ------   -----   ------   ----   --------",
    ]
    for r in reports:
        lines.append(
            f"  {r['lambda']:>6.1f}   {r['hit@1']:.3f}   {r['recall']:.3f}  "
            f"{r['mrr']:.3f}   {r['distinct']:>5.2f}"
        )
    return "\n".join(lines)
