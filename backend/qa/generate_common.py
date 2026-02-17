# backend/qa/generate_common.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Chunk:
    chunk_id: str
    text: str
    degree_level: str = "Unknown"
    program: str = "Unknown"


def _norm(s: str) -> str:
    return (s or "").strip()


def load_chunks_from_jsonl(path: Path) -> List[Chunk]:
    """
    Loads chunks from JSONL. Each line is a JSON dict.
    Expected fields (flexible):
      - chunk_id / id
      - text / chunk / content
      - degree_level / degree
      - program / course
    """
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    out: List[Chunk] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)

        cid = _norm(str(r.get("chunk_id") or r.get("id") or f"chunk_{i}"))
        text = _norm(r.get("text") or r.get("chunk") or r.get("content") or "")
        if not text:
            continue

        degree = _norm(r.get("degree_level") or r.get("degree") or "Unknown")
        program = _norm(r.get("program") or r.get("course") or "Unknown")

        out.append(Chunk(chunk_id=cid, text=text, degree_level=degree or "Unknown", program=program or "Unknown"))

    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def best_k_by_overlap(query: str, chunks: List[Chunk], k: int) -> List[Chunk]:
    scored = [(jaccard(query, c.text), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]
