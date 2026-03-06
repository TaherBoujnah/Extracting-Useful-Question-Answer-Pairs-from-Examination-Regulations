import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def quality_score(qa: Dict) -> float:
    q = (qa.get("question") or "").strip()
    a = (qa.get("answer") or "").strip()

    score = 0.0
    score += min(len(a), 500) / 500.0
    score += 0.2 if q.endswith("?") else 0.0
    score += 0.1 if any(ch.isdigit() for ch in a) else 0.0
    score -= 0.3 if len(a) < 40 else 0.0
    return score


def semantic_deduplicate(embeddings, scores, threshold=0.92):
    order = np.argsort(-scores)  # best quality first
    kept_indices = []

    for idx in order:
        if not kept_indices:
            kept_indices.append(idx)
            continue

        kept_vecs = embeddings[kept_indices]
        sims = kept_vecs @ embeddings[idx]  # cosine similarity (normalized)

        if np.max(sims) < threshold:
            kept_indices.append(idx)

    return sorted(kept_indices)


def main():
    parser = argparse.ArgumentParser(description="Semantic deduplication of FAQs (no size limits)")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--embed_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--cos_threshold", type=float, default=0.92)

    args = parser.parse_args()

    print("Loading dataset...")
    rows = read_jsonl(Path(args.input_jsonl))
    rows = [r for r in rows if r.get("question") and r.get("answer")]

    print(f"Loaded {len(rows)} questions")

    questions = [r["question"].strip() for r in rows]
    scores = np.array([quality_score(r) for r in rows], dtype=np.float32)

    print("Loading embedding model...")
    model = SentenceTransformer(args.embed_model)

    print("Encoding questions into embedding matrix...")
    embeddings = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)

    print("Performing semantic deduplication...")
    kept = semantic_deduplicate(embeddings, scores, threshold=args.cos_threshold)

    dedup_rows = [rows[i] for i in kept]

    write_jsonl(Path(args.out_jsonl), dedup_rows)

    print("✅ Deduplication complete")
    print(f"Original: {len(rows)}")
    print(f"After dedup: {len(dedup_rows)}")
    print(f"Cosine threshold: {args.cos_threshold}")
    print(f"Saved to: {args.out_jsonl}")


if __name__ == "__main__":
    main()


