import time

import requests

from .loader import chunk_text

API_ROOT = "https://dev.to/api"
REQUEST_DELAY = 0.35


def list_articles(username: str) -> list[dict]:
    response = requests.get(
        f"{API_ROOT}/articles",
        params={"username": username, "per_page": 100},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_article(article_id: int) -> dict:
    response = requests.get(f"{API_ROOT}/articles/{article_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def load_and_chunk(username: str) -> list[dict]:
    """Fetch a dev.to author's posts and split them into chunks ready for indexing."""
    documents = []

    for summary in list_articles(username):
        article = fetch_article(summary["id"])
        time.sleep(REQUEST_DELAY)

        body = article.get("body_markdown") or ""
        if not body.strip():
            continue

        # Prefixing each chunk with the title keeps standalone chunks retrievable
        # by topic even when the body text never restates what the post is about.
        title = article["title"]
        for i, chunk in enumerate(chunk_text(body)):
            documents.append(
                {
                    "text": f"# {title}\n\n{chunk}",
                    "source": f"dev.to/{username}#{article['id']}",
                    "chunk_index": i,
                    "title": title,
                    "url": article["url"],
                }
            )

    return documents
