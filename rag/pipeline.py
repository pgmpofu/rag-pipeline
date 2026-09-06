from anthropic import Anthropic

from . import store
from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, TOP_K

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "If the context does not contain the answer, say so plainly instead of guessing. "
    "Cite the source file for each claim you make when possible."
)


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {hit['source']}]\n{hit['text']}" for hit in hits
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )


def answer(question: str, top_k: int = TOP_K) -> dict:
    hits = store.query(question, top_k=top_k)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )

    return {
        "answer": response.content[0].text,
        "sources": [{"source": h["source"], "distance": h["distance"]} for h in hits],
    }
