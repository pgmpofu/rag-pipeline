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


def _overlap_tail(chunk: str, overlap: int) -> str:
    """The trailing slice of a chunk to repeat at the start of the next one."""
    if overlap <= 0 or not chunk:
        return ""

    tail = chunk[-overlap:]
    # Snap forward to a word boundary so the repeat never starts mid-word.
    space = tail.find(" ")
    return tail[space + 1 :].strip() if space != -1 else tail.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of at most chunk_size, each repeating roughly
    `overlap` characters of the previous one.

    The overlap matters because a sentence that falls across a boundary is
    otherwise retrievable from neither side.
    """
    overlap = max(0, min(overlap, chunk_size // 2))
    # Leave room for the repeated tail plus the "\n\n" joining it to the body.
    budget = chunk_size - overlap - 2

    blocks: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(para) <= budget:
            blocks.append(para)
        else:
            blocks.extend(para[i : i + budget] for i in range(0, len(para), budget))

    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        tail = _overlap_tail(current, overlap)
        current = f"{tail}\n\n{block}" if tail else block

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
