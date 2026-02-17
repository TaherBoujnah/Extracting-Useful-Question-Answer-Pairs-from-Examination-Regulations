# backend/qa/generate_slow_hq.py
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from backend.qa.ollama_client import ollama_generate
from backend.qa.generate_common import Chunk, best_k_by_overlap, load_chunks_from_jsonl, write_json, write_jsonl, jaccard


def qa_prompt(rule_text: str, degree_level: str, program: str) -> str:
    return f"""
Du bist ein Assistent, der aus Prüfungsordnungen sehr hochwertige FAQ-Frage-Antwort-Paare erzeugt.

Kontext:
- Studienniveau: {degree_level}
- Studiengang/Programm: {program}

Regel-/Textauszug:
\"\"\"{rule_text}\"\"\"

Aufgabe:
1) Formuliere eine konkrete, nützliche Frage (Studierendenperspektive).
2) Beantworte sie ausschließlich anhand des Auszugs (keine Spekulation).
3) Antwort 1–4 Sätze, klar und präzise.

Gib NUR gültiges JSON zurück:
{{"question": "...", "answer": "..."}}
""".strip()


def parse_one_json(raw: str) -> Dict[str, str] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except Exception:
        return None
    q = (obj.get("question") or "").strip()
    a = (obj.get("answer") or "").strip()
    if not q or not a:
        return None
    return {"question": q, "answer": a}


def generate_slow_hq(
    model: str,
    chunks_path: Path,
    out_path: Path,
    seed_limit: int = 10,
    max_context_words: int = 1400,
    max_rules_per_seed: int = 30,
    max_qas_per_seed: int = 6,
    temperature: float = 0.05,
    num_predict: int = 260,
    dup_question_jaccard_threshold: float = 0.85,
    checkpoint_every_seeds: int = 1,
) -> Dict[str, Any]:
    chunks = load_chunks_from_jsonl(chunks_path)
    if not chunks:
        raise RuntimeError(f"No chunks loaded from {chunks_path}")

    rng = random.Random(1337)
    seeds = rng.sample(chunks, k=min(seed_limit, len(chunks)))

    out_rows: List[Dict[str, Any]] = []
    seen_questions: List[str] = []

    t0 = time.time()

    for si, seed in enumerate(seeds, start=1):
        pool = [c for c in chunks if c.degree_level == seed.degree_level and c.program == seed.program]
        if len(pool) < max_rules_per_seed:
            pool = chunks

        rules = best_k_by_overlap(seed.text, pool, k=max_rules_per_seed)

        added_for_seed = 0
        for rule in rules:
            if added_for_seed >= max_qas_per_seed:
                break

            rule_text = " ".join(rule.text.split()[:max_context_words])
            raw = ollama_generate(
                model=model,
                prompt=qa_prompt(rule_text, seed.degree_level, seed.program),
                temperature=temperature,
                num_predict=num_predict,
            )
            parsed = parse_one_json(raw)
            if not parsed:
                continue

            q = parsed["question"]
            if any(jaccard(q, prev) >= dup_question_jaccard_threshold for prev in seen_questions):
                continue

            seen_questions.append(q)
            out_rows.append(
                {
                    "question": parsed["question"],
                    "answer": parsed["answer"],
                    "degree_level": seed.degree_level,
                    "program": seed.program,
                    "source_chunk_id": rule.chunk_id,
                }
            )
            added_for_seed += 1

        # checkpoints (helpful if it takes forever)
        if checkpoint_every_seeds and (si % checkpoint_every_seeds == 0):
            write_jsonl(out_path, out_rows)

    runtime = round(time.time() - t0, 2)

    write_jsonl(out_path, out_rows)
    meta = {
        "model": model,
        "strategy": "slow_hq",
        "runtime_seconds": runtime,
        "written_qas": len(out_rows),
        "settings": {
            "seed_limit": seed_limit,
            "max_context_words": max_context_words,
            "max_rules_per_seed": max_rules_per_seed,
            "max_qas_per_seed": max_qas_per_seed,
            "temperature": temperature,
            "num_predict": num_predict,
            "checkpoint_every_seeds": checkpoint_every_seeds,
        },
        "chunks_path": str(chunks_path),
        "output": str(out_path),
    }
    write_json(out_path.with_suffix(out_path.suffix + ".meta.json"), meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--out", default=None, help="Output JSONL path. Defaults to data/generated/<auto>.jsonl")
    ap.add_argument("--seed_limit", type=int, default=10)
    ap.add_argument("--max_context_words", type=int, default=1400)
    ap.add_argument("--max_rules_per_seed", type=int, default=30)
    ap.add_argument("--max_qas_per_seed", type=int, default=6)
    ap.add_argument("--checkpoint_every_seeds", type=int, default=1)
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    if args.out:
        out_path = Path(args.out)
    else:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        out_path = Path("data/generated") / f"faqs_slow_hq__{safe_model}.jsonl"

    meta = generate_slow_hq(
        model=args.model,
        chunks_path=chunks_path,
        out_path=out_path,
        seed_limit=args.seed_limit,
        max_context_words=args.max_context_words,
        max_rules_per_seed=args.max_rules_per_seed,
        max_qas_per_seed=args.max_qas_per_seed,
        checkpoint_every_seeds=args.checkpoint_every_seeds,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
