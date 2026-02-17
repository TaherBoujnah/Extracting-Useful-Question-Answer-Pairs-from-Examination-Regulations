# backend/api.py
import json
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# change this to whichever index you want active
INDEX_PATH = DATA_DIR / "faq_index_active.json"

EMAILS = {
    "bsc_informatik": "BSc-Informatik@hhu.de",
    "msc_informatik": "master@cs.uni-duesseldorf.de",
    "msc_ai_ds": "master-ai@hhu.de",
}

SIM_THRESHOLD = 0.55  # tune after testing

app = FastAPI(title="HHU FAQ Chat (Safe Retrieval)")

# loaded at startup
INDEX = None


class ChatRequest(BaseModel):
    track: str
    message: str


@app.on_event("startup")
def load_index():
    global INDEX
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INDEX_PATH}. Build an index then copy/rename it to faq_index_active.json"
        )
    INDEX = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    INDEX["embeddings"] = np.array(INDEX["embeddings"], dtype=np.float32)


@app.get("/health")
def health():
    return {"ok": True, "index_file": str(INDEX_PATH)}


def cosine_best(query_vec: np.ndarray, mat: np.ndarray):
    sims = mat @ query_vec  # embeddings are normalized
    best_idx = int(np.argmax(sims))
    return best_idx, float(sims[best_idx])


@app.post("/chat")
def chat(req: ChatRequest):
    from sentence_transformers import SentenceTransformer

    track = req.track
    message = req.message.strip()

    if track not in EMAILS:
        return {"type": "error", "message": "Unknown track."}

    qas = INDEX["qas"]
    emb = INDEX["embeddings"]

    # filter to track
    track_idxs = [i for i, qa in enumerate(qas) if qa.get("track") == track]
    if not track_idxs:
        return {
            "type": "redirect",
            "reason": "no_faq_for_track",
            "emails": EMAILS
        }

    # encode query
    model = SentenceTransformer(INDEX["embed_model"])
    qvec = model.encode([message], normalize_embeddings=True)[0].astype(np.float32)

    mat = emb[track_idxs]
    best_local, score = cosine_best(qvec, mat)
    best_idx = track_idxs[best_local]
    best = qas[best_idx]

    if score < SIM_THRESHOLD:
        return {
            "type": "redirect",
            "reason": "no_similar_question_found",
            "score": score,
            "emails": EMAILS
        }

    return {
        "type": "answer",
        "score": score,
        "matched_question": best["question"],
        "answer": best["answer"]
    }
