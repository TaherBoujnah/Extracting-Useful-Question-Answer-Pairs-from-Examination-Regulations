# backend/evaluation/layer1_jsonl.py
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.qa.ollama_client import ollama_generate  # your existing client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# ----------------------------
# Text utils
# ----------------------------
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+", re.UNICODE)

GER_STOP = {
    "und", "oder", "der", "die", "das", "ein", "eine", "einer", "eines", "den", "dem", "des",
    "im", "in", "am", "an", "auf", "für", "von", "mit", "zu", "zum", "zur", "aus", "bei",
    "ist", "sind", "war", "waren", "wird", "werden", "kann", "können", "muss", "müssen",
    "darf", "dürfen", "nicht", "nur", "auch", "als", "wenn", "dann", "dass", "da", "wie",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "man",
}


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in WORD_RE.findall(text or "")]
    out: List[str] = []
    for t in toks:
        if len(t) <= 2:
            continue
        if t in GER_STOP:
            continue
        out.append(t)
    return out


# ----------------------------
# Hashed bag-of-words cosine (pure python)
# ----------------------------
def hashed_bow(tokens: List[str], dim: int = 2048) -> Dict[int, float]:
    v: Dict[int, float] = {}
    for tok in tokens:
        h = int(md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        v[idx] = v.get(idx, 0.0) + sign
    return v


def cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    if len(a) > len(b):
        a, b = b, a
    for i, va in a.items():
        vb = b.get(i)
        if vb is not None:
            dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


# ----------------------------
# Scoring helpers
# ----------------------------
def token_f1(pred: str, gold: str) -> float:
    pt = tokenize(pred)
    gt = tokenize(gold)
    if not pt or not gt:
        return 0.0

    used: Dict[str, int] = {}
    overlap = 0
    gcounts: Dict[str, int] = {}
    for t in gt:
        gcounts[t] = gcounts.get(t, 0) + 1

    for t in pt:
        c = gcounts.get(t, 0)
        u = used.get(t, 0)
        if u < c:
            overlap += 1
            used[t] = u + 1

    prec = overlap / max(1, len(pt))
    rec = overlap / max(1, len(gt))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def hallucination_rate(answer: str, chunk_text: str) -> float:
    at = tokenize(answer)
    if not at:
        return 0.0
    ct = set(tokenize(chunk_text))
    missing = sum(1 for t in at if t not in ct)
    return missing / max(1, len(at))


# ----------------------------
# Loading data
# ----------------------------
def load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")
    chunks: List[Dict[str, Any]] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "id" not in obj:
            obj["id"] = obj.get("chunk_id") or obj.get("chunkId") or "unknown"
        if "text" not in obj:
            obj["text"] = obj.get("content") or obj.get("context") or ""
        chunks.append(obj)
    return chunks


def _try_load_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def load_generated_qas(
    jsonl_path: Path, limit_qas: Optional[int]
) -> Tuple[List[Dict[str, Any]], Optional[float], Dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {jsonl_path}")

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    gen_meta: Dict[str, Any] = {}
    qas: List[Dict[str, Any]] = []

    # --- sidecar meta discovery (includes your "<file>.jsonl.meta.json") ---
    sidecars = [
        jsonl_path.with_name(jsonl_path.stem + "__meta.json"),
        jsonl_path.with_name(jsonl_path.stem + ".meta.json"),
        jsonl_path.with_suffix(".meta.json"),
        jsonl_path.with_name(jsonl_path.name + ".meta.json"),  # ✅ foo.jsonl.meta.json
    ]

    loaded_sidecar = None
    for sc in sidecars:
        if sc.exists():
            loaded_sidecar = sc
            try:
                meta_obj = json.loads(sc.read_text(encoding="utf-8"))
                if isinstance(meta_obj, dict):
                    gen_meta = meta_obj
                else:
                    print(f"⚠️ meta file is not a dict: {sc} (type={type(meta_obj)})")
            except Exception as e:
                print(f"⚠️ failed to read meta file {sc}: {e}")
            break

    if loaded_sidecar:
        print(f"🧾 loaded meta sidecar: {loaded_sidecar}")
        if isinstance(gen_meta, dict):
            print(f"🧾 meta keys: {sorted(list(gen_meta.keys()))[:50]}")

    # --- first-line metadata dict in JSONL (optional) ---
    start_idx = 0
    if lines:
        try:
            first = json.loads(lines[0])
            if isinstance(first, dict) and (
                "runtime_seconds" in first
                or "generation_runtime_seconds" in first
                or "settings" in first
                or "strategy" in first
                or "model" in first
            ):
                # Merge: first-line overrides sidecar
                gen_meta = {**gen_meta, **first}
                start_idx = 1
                print("🧾 loaded meta from first JSONL line (merged)")
        except Exception:
            pass

    # --- parse QA objects ---
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        q = obj.get("question") or obj.get("q")
        a = obj.get("answer") or obj.get("a")
        if not q or not a:
            continue
        obj["question"] = str(q).strip()
        obj["answer"] = str(a).strip()
        obj["program"] = obj.get("program") or obj.get("degree_program") or obj.get("track") or "unknown"
        obj["degree_level"] = obj.get("degree_level") or obj.get("level") or "unknown"
        qas.append(obj)
        if limit_qas and len(qas) >= limit_qas:
            break

    # --- robust runtime extraction ---
    gen_runtime = None
    if isinstance(gen_meta, dict):
        # try several likely keys
        candidates = [
            gen_meta.get("runtime_seconds"),
            gen_meta.get("generation_runtime_seconds"),
            gen_meta.get("generation_seconds"),
            gen_meta.get("total_seconds"),
            gen_meta.get("runtime"),
            gen_meta.get("time_seconds"),
        ]
        val = next((c for c in candidates if c is not None), None)

        # allow nested shapes like {"timing": {"runtime_seconds": ...}}
        if val is None:
            timing = gen_meta.get("timing")
            if isinstance(timing, dict):
                val = timing.get("runtime_seconds") or timing.get("generation_runtime_seconds")

        if val is not None:
            try:
                if isinstance(val, str):
                    val = val.strip()
                    if val.endswith("s"):
                        val = val[:-1].strip()
                gen_runtime = float(val)
            except Exception as e:
                print(f"⚠️ could not parse runtime value {val!r} into float: {e}")
                gen_runtime = None

    print(f"🧾 parsed generation runtime: {gen_runtime}")
    return qas, gen_runtime, (gen_meta if isinstance(gen_meta, dict) else {})



# ----------------------------
# Retrieval
# ----------------------------
@dataclass
class Retriever:
    chunk_ids: List[str]
    chunk_texts: List[str]
    chunk_vecs: List[Dict[int, float]]
    dim: int = 2048

    @classmethod
    def from_chunks(cls, chunks: List[Dict[str, Any]], dim: int = 2048) -> "Retriever":
        ids: List[str] = []
        texts: List[str] = []
        vecs: List[Dict[int, float]] = []
        for c in chunks:
            cid = str(c.get("id", "unknown"))
            txt = str(c.get("text", "") or "")
            ids.append(cid)
            texts.append(txt)
            vecs.append(hashed_bow(tokenize(txt), dim=dim))
        return cls(ids, texts, vecs, dim=dim)

    def top1(self, query: str) -> Tuple[str, str, float]:
        qv = hashed_bow(tokenize(query), dim=self.dim)
        best_i = 0
        best_s = -1.0
        for i, cv in enumerate(self.chunk_vecs):
            s = cosine_sparse(qv, cv)
            if s > best_s:
                best_s = s
                best_i = i
        return self.chunk_ids[best_i], self.chunk_texts[best_i], float(best_s)


# ----------------------------
# Roundtrip
# ----------------------------
def roundtrip_answer(model: str, question: str, context: str) -> str:
    prompt = (
        "Du bist ein Hochschul-FAQ Assistent.\n"
        "Beantworte die Frage NUR mit Informationen aus dem Kontext.\n"
        "Wenn der Kontext nicht reicht, sage: \"Das geht aus dem Kontext nicht hervor.\".\n\n"
        f"KONTEXT:\n{context}\n\n"
        f"FRAGE:\n{question}\n\n"
        "ANTWORT:"
    )
    return ollama_generate(model=model, prompt=prompt, temperature=0.1, num_predict=220)


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_jsonl(
    input_path: Path,
    output_path: Path,
    model_tag: str,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    limit_qas: Optional[int] = None,
    roundtrip_model: Optional[str] = None,
    roundtrip_ok_threshold: float = 0.60,
    success_f1_threshold: float = 0.20,
    bow_dim: int = 2048,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    chunks = load_chunks(chunks_path)
    retriever = Retriever.from_chunks(chunks, dim=bow_dim)

    qas, gen_runtime, gen_meta = load_generated_qas(input_path, limit_qas)

    print(f"🔎 Judge-free eval | {model_tag}")
    print(f" - qas: {len(qas)} (limit_qas={limit_qas})")
    print(f" - chunks: {len(chunks)}")
    print(f" - retrieval: hashed BOW cosine (dim={bow_dim})")
    print(f" - generation_runtime_seconds: {gen_runtime}")

    per_qa: List[Dict[str, Any]] = []
    used_chunks = set()

    sum_cos = 0.0
    sum_f1_to_chunk = 0.0
    sum_hall = 0.0
    success = 0

    rt_f1s: List[float] = []
    rt_ok = 0
    rt_attempts = 0

    for qa in qas:
        q = qa["question"]
        a = qa["answer"]

        cid, ctxt, cos = retriever.top1(q)
        used_chunks.add(cid)

        f1_chunk = token_f1(a, ctxt)
        hall = hallucination_rate(a, ctxt)
        ok = 1 if f1_chunk >= success_f1_threshold else 0

        sum_cos += cos
        sum_f1_to_chunk += f1_chunk
        sum_hall += hall
        success += ok

        row: Dict[str, Any] = {
            "question": q,
            "answer": a,
            "program": qa.get("program", "unknown"),
            "degree_level": qa.get("degree_level", "unknown"),
            "retrieved_chunk_id": cid,
            "retrieval_cosine": round(cos, 4),
            "token_f1_to_retrieved_chunk": round(f1_chunk, 4),
            "hallucination_rate": round(hall, 4),
            "success": bool(ok),
        }

        if roundtrip_model:
            rt_attempts += 1
            pred = roundtrip_answer(roundtrip_model, q, ctxt)
            rt_f1 = token_f1(pred, a)
            rt_f1s.append(rt_f1)
            is_ok = rt_f1 >= roundtrip_ok_threshold
            if is_ok:
                rt_ok += 1
            row["roundtrip_answer"] = pred
            row["roundtrip_f1"] = round(rt_f1, 4)
            row["roundtrip_ok"] = bool(is_ok)

        per_qa.append(row)

    n = max(1, len(qas))
    avg_cos = sum_cos / n
    avg_f1 = sum_f1_to_chunk / n
    avg_hall = sum_hall / n
    accuracy_success_rate = success / n
    coverage = len(used_chunks) / max(1, len(chunks))

    eval_runtime = time.perf_counter() - t0
    total_runtime = (gen_runtime + eval_runtime) if gen_runtime is not None else None

    out: Dict[str, Any] = {
        "model_tag": model_tag,
        "num_qas": len(qas),
        "chunks_total": len(chunks),
        "limits": {"limit_qas": limit_qas},

        "generation_runtime_seconds": gen_runtime,
        "eval_runtime_seconds": round(eval_runtime, 2),
        "total_runtime_seconds": round(total_runtime, 2) if total_runtime is not None else None,

        "accuracy_success_rate": round(accuracy_success_rate, 4),
        "coverage": round(coverage, 4),
        "hallucination_rate": round(avg_hall, 4),
        "avg_retrieval_cosine": round(avg_cos, 4),
        "avg_token_f1": round(avg_f1, 4),

        "roundtrip_model": roundtrip_model,
        "roundtrip_ok_rate": round(rt_ok / rt_attempts, 4) if rt_attempts > 0 else None,
        "avg_roundtrip_f1": round(sum(rt_f1s) / len(rt_f1s), 4) if rt_f1s else None,

        "gen_meta": gen_meta,
        "per_qa": per_qa,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved: {output_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--model_tag", type=str, required=True)

    # Support both flags
    ap.add_argument("--chunks", type=str, default=str(DEFAULT_CHUNKS_PATH))
    ap.add_argument("--chunks_path", dest="chunks", type=str)

    ap.add_argument("--limit_qas", type=int, default=None)
    ap.add_argument("--roundtrip_model", type=str, default=None)
    ap.add_argument("--roundtrip_ok_threshold", type=float, default=0.60)
    ap.add_argument("--success_f1_threshold", type=float, default=0.20)
    ap.add_argument("--bow_dim", type=int, default=2048)

    args = ap.parse_args()

    evaluate_jsonl(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_tag=args.model_tag,
        chunks_path=Path(args.chunks),
        limit_qas=args.limit_qas,
        roundtrip_model=args.roundtrip_model,
        roundtrip_ok_threshold=args.roundtrip_ok_threshold,
        success_f1_threshold=args.success_f1_threshold,
        bow_dim=args.bow_dim,
    )


if __name__ == "__main__":
    main()

