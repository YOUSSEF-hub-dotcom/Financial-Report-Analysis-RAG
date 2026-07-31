"""
Centralized configuration for the Financial RAG system.

Loads environment variables from .env and exposes typed constants
used across all modules.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- Paths ---
PROJECT_ROOT: Path = _PROJECT_ROOT
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(Path.home() / "Financial_RAG" / "data")))
LOG_DIR: Path = PROJECT_ROOT / "logs"

# --- API Keys ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

# --- MongoDB ---
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB: str = os.getenv("MONGODB_DB", "financial_rag")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "raw_chunks")

# --- Qdrant ---
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "financial_vectors")
QDRANT_PATH: str = os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant_db"))
EMBEDDING_DIM: int = 768

# --- Redis ---
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- Chunking ---
CHUNK_MIN_TOKENS: int = 512
CHUNK_MAX_TOKENS: int = 768
CHUNK_OVERLAP_RATIO: float = 0.125  # 12.5% overlap (midpoint of 10-15%)

# --- LLM ---
GROQ_PRIMARY_MODEL: str = "llama-3.1-8b-instant"
GROQ_FALLBACK_MODEL: str = "gemma2-9b-it"
LLM_TEMPERATURE: float = 0.0
LLM_SEED: int = 42
LLM_MAX_TOKENS: int = 1524
LLM_HISTORY_K: int = 3

# --- Supported tickers ---
SUPPORTED_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA"]
