import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = os.environ.get("PERSIST_DIR", "chroma_db")
COLLECTION_NAME = "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5
