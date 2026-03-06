from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import (
    DATA_DIR, CHUNKS_JSONL, EMBEDDING_MODEL_NAME,
    NUMPY_EMB_PATH, NUMPY_META_PATH
)
from backend.retrieval.load_chunks import load_chunks_map
from backend.retrieval.numpy_store import NumpyVectorStore


def main():
    chunks_map = load_chunks_map(Path(CHUNKS_JSONL))
    chunk_ids: List[str] = list(chunks_map.keys())

    # BGE-M3 retrieval format: passages should be prefixed
    texts: List[str] = [
        "passage: " + chunks_map[cid]["text"]
        for cid in chunk_ids
    ]

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # dot product = cosine similarity
    ).astype(np.float32)

    meta: Dict[str, Any] = {
        "model_name": EMBEDDING_MODEL_NAME,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "text_prefix": "passage: ",
    }

    NumpyVectorStore.save(
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        meta=meta,
        emb_path=Path(NUMPY_EMB_PATH),
        meta_path=Path(NUMPY_META_PATH),
    )

    print("Saved embeddings to:")
    print(" -", NUMPY_EMB_PATH)
    print(" -", NUMPY_META_PATH)
    print("Chunks:", len(chunk_ids))
    print("Model:", EMBEDDING_MODEL_NAME)


if __name__ == "__main__":
    main()