# backend/evaluation/filter.py
"""
Filter + deduplicate generated FAQ JSONL.

What it does:
- Reads a JSONL of {"question": ..., "answer": ...} objects (extra fields allowed).
- Optional quality filters (min lengths).
- Deduplicates questions using token-level Jaccard similarity.
- Keeps the "best" item among duplicates (by a simple quality score: longer answer + non-empty source if present).

Usage (PowerShell):
  python -m backend.evaluation.filter `
    --input "data\generated\faqs_hybrid__qwen2.5_7b.jsonl" `
    --output "data\generated\faqs_hybrid__qwen2.5_7b_clean.jsonl" `
    --dup_question_jaccard_threshold 0.85 `
    --min_answer_length 40

Notes:
- This is judge-free: no LLM required.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return _WORD_RE.findall(s)


def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    a = set(a_tokens)
    b = set(b_tokens)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def safe_get_str(obj: Dict[str, Any], key: str) -> str:
    v = obj.get(key, "")
    return v if isinstance(v, str) else ""


def quality_score(item: Dict[str, Any]) -> float:
    """
    Heuristic to keep the better sample when duplicates exist.
    You can tweak this anytime without breaking file format.
    """
    q = safe_get_str(item, "question")
    a = safe_get_str(item, "answer")

    # Some generators may store sources under different keys
    src = safe_get_str(item, "source_excerpt") or safe_get_str(item, "source") or safe_get_str(item, "excerpt")

    score = 0.0
    score += min(len(q), 300) / 300.0          # prefer non-trivial questions
    score += min(len(a), 1200) / 1200.0        # prefer informative answers
    score += 0.25 if src.strip() else 0.0      # prefer items with any source text
    return score


@dataclass
class FilterStats:
    read: int = 0
    written: int = 0
    dropped_bad: int = 0
    dropped_dupe: int = 0
    replaced_dupe: int = 0


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            # ignore malformed lines
            continue


def is_good_item(
    item: Dict[str, Any],
    min_question_length: int,
    min_answer_length: int,
) -> bool:
    q = safe_get_str(item, "question").strip()
    a = safe_get_str(item, "answer").strip()
    if len(q) < min_question_length:
        return False
    if len(a) < min_answer_length:
        return False
    # avoid super short "?" style questions
    if len(tokenize(q)) < 3:
        return False
    return True


def dedup_items(
    items: List[Dict[str, Any]],
    dup_question_jaccard_threshold: float,
    stats: FilterStats,
) -> List[Dict[str, Any]]:
    """
    Greedy dedup:
    - Keep a growing list of accepted items.
    - For each new item, compare to existing using question Jaccard.
    - If duplicate, keep the one with higher quality_score.
    """
    kept: List[Dict[str, Any]] = []
    kept_tokens: List[List[str]] = []

    for item in items:
        q = safe_get_str(item, "question")
        q_tokens = tokenize(q)

        best_match_idx: Optional[int] = None
        best_sim: float = 0.0

        for i, kt in enumerate(kept_tokens):
            sim = jaccard(q_tokens, kt)
            if sim > best_sim:
                best_sim = sim
                best_match_idx = i

        if best_match_idx is None or best_sim < dup_question_jaccard_threshold:
            kept.append(item)
            kept_tokens.append(q_tokens)
            continue

        # duplicate found
        stats.dropped_dupe += 1
        existing = kept[best_match_idx]
        if quality_score(item) > quality_score(existing):
            kept[best_match_idx] = item
            kept_tokens[best_match_idx] = q_tokens
            stats.replaced_dupe += 1

    return kept


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter & deduplicate FAQ JSONL")
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument(
        "--dup_question_jaccard_threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold to treat questions as duplicates (0..1).",
    )
    ap.add_argument("--min_question_length", type=int, default=12, help="Minimum number of characters in question.")
    ap.add_argument("--min_answer_length", type=int, default=30, help="Minimum number of characters in answer.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for debugging (0=all).")

    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        raise FileNotFoundError(f"Input JSONL not found: {inp}")

    stats = FilterStats()

    raw: List[Dict[str, Any]] = []
    for obj in iter_jsonl(inp):
        stats.read += 1
        if args.limit and len(raw) >= args.limit:
            break
        if not is_good_item(obj, args.min_question_length, args.min_answer_length):
            stats.dropped_bad += 1
            continue
        raw.append(obj)

    deduped = dedup_items(raw, args.dup_question_jaccard_threshold, stats)
    stats.written = len(deduped)

    write_jsonl(out, deduped)

    print("✅ filter.py done")
    print(f" - input:  {inp}")
    print(f" - output: {out}")
    print(f" - read: {stats.read}")
    print(f" - kept after quality filter: {len(raw)} (dropped_bad={stats.dropped_bad})")
    print(f" - written after dedup: {stats.written} (dropped_dupe={stats.dropped_dupe}, replaced_dupe={stats.replaced_dupe})")
    print(f" - dup_question_jaccard_threshold={args.dup_question_jaccard_threshold}")


if __name__ == "__main__":
    main()

