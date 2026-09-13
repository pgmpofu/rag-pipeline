from pathlib import Path

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def discover_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        p for p in source.rglob("*") if p.suffix in SUPPORTED_EXTENSIONS and p.is_file()
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue

        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i : i + chunk_size])
            current = ""

    if current:
        chunks.append(current)

    return chunks


def load_and_chunk(source: Path) -> list[dict]:
    """Read local files and split them into chunks ready for indexing."""
    documents = []
    for file_path in discover_files(source):
        text = read_file(file_path)
        if not text.strip():
            continue
        for i, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "text": chunk,
                    "source": str(file_path),
                    "chunk_index": i,
                    "title": file_path.name,
                    "url": "",
                }
            )
    return documents
