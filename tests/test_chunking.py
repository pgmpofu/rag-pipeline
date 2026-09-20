"""Chunking invariants. Run with: python tests/test_chunking.py

Plain asserts rather than pytest so the suite needs no extra dependency.
"""

import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.loader import chunk_text  # noqa: E402


def _paragraphs(count, words=45):
    random.seed(0)
    return "\n\n".join(
        " ".join("".join(random.choices(string.ascii_lowercase, k=6)) for _ in range(words))
        for _ in range(count)
    )


def test_respects_chunk_size():
    chunks = chunk_text(_paragraphs(8), chunk_size=400, overlap=60)
    oversized = [len(c) for c in chunks if len(c) > 400]
    assert not oversized, f"chunks exceeded chunk_size: {oversized}"


def test_adjacent_chunks_actually_overlap():
    chunks = chunk_text(_paragraphs(8), chunk_size=400, overlap=60)
    assert len(chunks) > 1, "test needs multiple chunks to be meaningful"

    for i, (first, second) in enumerate(zip(chunks, chunks[1:])):
        shared = next(
            (n for n in range(min(len(first), len(second)), 0, -1) if first[-n:] == second[:n]),
            0,
        )
        assert shared > 0, f"boundary {i} has no overlap (this was the original bug)"


def test_zero_overlap_is_honoured():
    chunks = chunk_text(_paragraphs(8), chunk_size=400, overlap=0)
    for first, second in zip(chunks, chunks[1:]):
        assert not second.startswith(first[-20:]), "overlap=0 should not repeat text"


def test_oversized_paragraph_is_split():
    chunks = chunk_text("x" * 5000, chunk_size=400, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_empty_and_whitespace_input():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  \n\n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("A short note.") == ["A short note."]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
