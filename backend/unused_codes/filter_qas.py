# backend/filter_qas.py
import json
import re
import argparse
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def run_filter(input_path: Path, output_path: Path, sim_threshold=0.9):
    data = json.loads(input_path.read_text(encoding="utf-8"))

    # support wrapper format: {"model":..., "qas":[...]}
    if isinstance(data, dict) and "qas" in data:
        wrapper = data
        qas = data["qas"]
    else:
        wrapper = None
        qas = data

    filtered = []
    seen_q = set()
    seen_a = set()

    for qa in qas:
        q = qa.get("question", "")
        a = qa.get("answer", "")
        qn = normalize(q)
        an = normalize(a)

        if qn in seen_q or an in seen_a:
            continue

        dup = False
        for prev in filtered:
            if similarity(qn, normalize(prev.get("question", ""))) >= sim_threshold:
                dup = True
                break
        if dup:
            continue

        if len(a) < 25 or len(a) > 500:
            continue

        filtered.append(qa)
        seen_q.add(qn)
        seen_a.add(an)

    if wrapper is not None:
        wrapper["qas"] = filtered
        output = wrapper
    else:
        output = filtered

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"🧹 Filtered QAs: {len(qas)} → {len(filtered)}")
    print(f"✅ Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sim", type=float, default=0.9)
    args = parser.parse_args()

    run_filter(Path(args.input), Path(args.output), sim_threshold=args.sim)
