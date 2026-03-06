# backend/evaluation/retrieve_chunk.py
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.evaluation.judgefree_metrics import content_tokens


@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: dict[str, Any]


class BM25Index:
    """
    Pure-python BM25 over chunk texts.
    Enough for 117 chunks and avoids sklearn/scipy.
    """
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.docs_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}

        for ch in chunks:
            toks = content_tokens(ch.text)
            self.docs_tokens.append(toks)
            self.doc_len.append(len(toks))
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

        self.avgdl = (sum(self.doc_len) / max(1, len(self.doc_len))) if self.doc_len else 0.0
        n = len(self.chunks)

        for t, dft in self.df.items():
            # classic BM25 idf
            self.idf[t] = math.log(1 + (n - dft + 0.5) / (dft + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        q = content_tokens(query)
        if not q:
            return 0.0
        toks = self.docs_tokens[doc_idx]
        if not toks:
            return 0.0
        freqs: dict[str, int] = {}
        for t in toks:
            freqs[t] = freqs.get(t, 0) + 1

        dl = self.doc_len[doc_idx]
        denom_norm = self.k1 * (1 - self.b + self.b * (dl / max(1e-9, self.avgdl)))

        s = 0.0
        for term in q:
            if term not in freqs:
                continue
            tf = freqs[term]
            idf = self.idf.get(term, 0.0)
            s += idf * (tf * (self.k1 + 1)) / (tf + denom_norm)
        return s

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self.chunks))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def load_chunks(chunks_path: Path) -> list[Chunk]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks: list[Chunk] = []
    if chunks_path.suffix.lower() == ".jsonl":
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunk_id = obj.get("chunk_id") or obj.get("id") or obj.get("chunkId") or "unknown"
            text = obj.get("text") or obj.get("content") or obj.get("context") or ""
            meta = {k: v for k, v in obj.items() if k not in ("text", "content", "context")}
            chunks.append(Chunk(chunk_id=chunk_id, text=text, meta=meta))
    else:
        obj = json.loads(chunks_path.read_text(encoding="utf-8"))
        for ch in obj:
            chunk_id = ch.get("chunk_id") or ch.get("id") or "unknown"
            text = ch.get("text") or ch.get("content") or ""
            meta = {k: v for k, v in ch.items() if k not in ("text", "content")}
            chunks.append(Chunk(chunk_id=chunk_id, text=text, meta=meta))

    # drop empties
    chunks = [c for c in chunks if c.text.strip()]
    if not chunks:
        raise RuntimeError("No chunks loaded (all empty). Check your chunks.jsonl format.")
    return chunks
