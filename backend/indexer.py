# backend/indexer.py
import json
import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index(generated_path: Path, out_path: Path, embed_model: str = DEFAULT_EMBED_MODEL):
    data = json.loads(generated_path.read_text(encoding="utf-8"))
    qas = data["qas"] if isinstance(data, dict) and "qas" in data else data

    # Expect fields: track, question, answer
    questions = [qa["question"] for qa in qas]
    model = SentenceTransformer(embed_model)
    emb = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)

    payload = {
        "embed_model": embed_model,
        "source_file": str(generated_path),
        "qas": qas,
        "embeddings": emb.astype(np.float32).tolist()
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved index → {out_path} | qas={len(qas)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="generated_*.json")
    ap.add_argument("--output", required=True, help="faq_index_*.json")
    ap.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    args = ap.parse_args()

    build_index(Path(args.input), Path(args.output), embed_model=args.embed_model)
