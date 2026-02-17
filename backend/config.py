from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# data/ is outside backend/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
# Add in config.py
NUMPY_EMB_PATH = Path(os.getenv("NUMPY_EMB_PATH", DATA_DIR / "embeddings.npy"))
NUMPY_META_PATH = Path(os.getenv("NUMPY_META_PATH", DATA_DIR / "embeddings_meta.json"))


CHUNKS_JSONL = Path(os.getenv("CHUNKS_JSONL", DATA_DIR / "chunks.jsonl"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", DATA_DIR / "faiss.index"))
FAISS_META_PATH = Path(os.getenv("FAISS_META_PATH", DATA_DIR / "faiss_meta.json"))

# Embedding model (local)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
BUNDLE_NEIGHBORS = os.getenv("BUNDLE_NEIGHBORS", "true").lower() == "true"
BUNDLE_DEPS = os.getenv("BUNDLE_DEPS", "true").lower() == "true"

# Answering mode: "extractive" (no external LLM) by default.
ANSWER_MODE = os.getenv("ANSWER_MODE", "extractive")
