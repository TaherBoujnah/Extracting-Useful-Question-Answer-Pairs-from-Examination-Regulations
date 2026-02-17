from __future__ import annotations
import argparse, json, re, hashlib
from pathlib import Path
from typing import Dict, List, Tuple

def read_jsonl(p: Path) -> List[dict]:
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def write_jsonl(p: Path, rows: List[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

_ws = re.compile(r"\s+")
def norm(s: str) -> str:
    s = s.lower().strip()
    s = _ws.sub(" ", s)
    return s

def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def is_bad_qa(q: str, a: str) -> bool:
    qn, an = q.strip(), a.strip()
    if len(qn) < 12: return True
    if len(an) < 25: return True
    if qn.endswith("?") and len(qn) <= 14:  # e.g. "Bachelorarbeit?"
        return True
    if an.lower().startswith(("ich weiß nicht", "keine ahnung", "sorry")):
        return True
    # avoid ultra-generic answers
    if an.lower() in {"ja", "nein", "kommt drauf an"}:
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--max_per_seed", type=int, default=0, help="0 = keep all")
    args = ap.parse_args()

    inp = Path(args.infile)
    out = Path(args.outfile)

    rows = read_jsonl(inp)

    # 1) drop obviously bad
    cleaned = []
    for r in rows:
        q = str(r.get("question","")).strip()
        a = str(r.get("answer","")).strip()
        if not q or not a: 
            continue
        if is_bad_qa(q,a):
            continue
        cleaned.append(r)

    # 2) dedup by normalized question (keep first)
    seen = set()
    deduped = []
    for r in cleaned:
        q = norm(str(r.get("question","")))
        key = sha(q)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # 3) optionally cap per seed (if your rows contain "seed")
    if args.max_per_seed and args.max_per_seed > 0:
        by_seed: Dict[str, List[dict]] = {}
        for r in deduped:
            seed = str(r.get("seed", "unknown"))
            by_seed.setdefault(seed, []).append(r)
        limited = []
        for seed, lst in by_seed.items():
            limited.extend(lst[: args.max_per_seed])
        deduped = limited

    write_jsonl(out, deduped)
    print(f"✅ in={len(rows)}  after_filter={len(cleaned)}  after_dedup={len(deduped)}  -> {out}")

if __name__ == "__main__":
    main()
