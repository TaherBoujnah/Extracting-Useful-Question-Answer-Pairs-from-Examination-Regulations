from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np


class NumpyVectorStore:
    """
    Stores normalized embeddings in a .npy file and metadata in .json.
    Retrieval = cosine similarity via dot product (since vectors are normalized).
    """

    def __init__(self, embeddings: np.ndarray, chunk_ids: List[str], meta: Dict[str, Any]):
        self.embeddings = embeddings.astype(np.float32)   # shape: (N, D), normalized
        self.chunk_ids = chunk_ids
        self.meta = meta

    @staticmethod
    def save(embeddings: np.ndarray, chunk_ids: List[str], meta: Dict[str, Any],
             emb_path: Path, meta_path: Path):
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_path), embeddings.astype(np.float32))
        meta_out = {**meta, "chunk_ids": chunk_ids, "shape": list(embeddings.shape)}
        meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(emb_path: Path, meta_path: Path) -> "NumpyVectorStore":
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        embeddings = np.load(str(emb_path)).astype(np.float32)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunk_ids = meta["chunk_ids"]
        return NumpyVectorStore(embeddings=embeddings, chunk_ids=chunk_ids, meta=meta)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        query_vec must be normalized (shape: (D,) or (1,D)).
        Returns list of (chunk_id, score) where score is cosine similarity (higher better).
        """
        if query_vec.ndim == 2:
            query_vec = query_vec[0]
        query_vec = query_vec.astype(np.float32)

        # cosine sim for normalized vectors = dot product
        sims = self.embeddings @ query_vec  # (N,)
        if top_k >= len(self.chunk_ids):
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]

        return [(self.chunk_ids[i], float(sims[i])) for i in top_idx]
