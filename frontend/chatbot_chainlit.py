"""
Single-file Chainlit FAQ chatbot (local retrieval over generated QA JSONL).

PowerShell:
  .\.venv\Scripts\Activate
  pip install -U chainlit sentence-transformers numpy orjson
  $env:FAQ_FILE="data\generated\faqs_hybrid__qwen2.5_7b__production.jsonl"
  chainlit run frontend/app.py
"""

import os
import re
from pathlib import Path

import chainlit as cl
import numpy as np

try:
    import orjson as jsonlib
except Exception:
    import json as jsonlib

from sentence_transformers import SentenceTransformer


# ----------------------------
# Config
# ----------------------------
FAQ_FILE = Path(os.getenv("FAQ_FILE", "data/generated/faqs_hybrid__qwen2.5_7b__production.jsonl"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

TOP_K = int(os.getenv("TOP_K", "3"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.55"))

EMB_CACHE = FAQ_FILE.with_suffix(".embeddings.npy")

# Fallback contacts shown when no match is found
CONTACTS_FALLBACK = {
    "Bachelor": os.getenv("CONTACT_BACHELOR_EMAIL", "bachelor@hhu.de"),
    "Master": os.getenv("CONTACT_MASTER_EMAIL", "master@hhu.de"),
    "Master AI & Data Science": os.getenv("CONTACT_MAIDS_EMAIL", "ai-datascience@hhu.de"),
}


# ----------------------------
# Globals
# ----------------------------
_MODEL = None
_QA = None
_EMB = None


# ----------------------------
# Helpers
# ----------------------------
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"FAQ_FILE not found: {path}")

    rows: list[dict] = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if jsonlib.__name__ == "orjson":
                    obj = jsonlib.loads(line)
                else:
                    obj = jsonlib.loads(line.decode("utf-8"))
            except Exception:
                continue

            q = obj.get("question") or obj.get("q") or obj.get("prompt")
            a = obj.get("answer") or obj.get("a") or obj.get("completion")
            if not q or not a:
                continue

            rows.append(
                {
                    "question": str(q).strip(),
                    "answer": str(a).strip(),
                }
            )

    if not rows:
        raise ValueError(f"No usable Q/A rows found in {path}")
    return rows


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def _ensure_loaded() -> None:
    global _MODEL, _QA, _EMB

    if _QA is None:
        _QA = _read_jsonl(FAQ_FILE)

    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBED_MODEL)

    if _EMB is None:
        if EMB_CACHE.exists():
            emb = np.load(EMB_CACHE)
            if emb.ndim == 2 and emb.shape[0] == len(_QA):
                _EMB = emb.astype(np.float32)
                return

        questions = [row["question"] for row in _QA]
        emb = _MODEL.encode(questions, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        emb = np.asarray(emb, dtype=np.float32)
        emb = _l2_normalize(emb)
        np.save(EMB_CACHE, emb)
        _EMB = emb


def _embed_query(text: str) -> np.ndarray:
    vec = _MODEL.encode([text], normalize_embeddings=True)[0]
    vec = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(vec) + 1e-12
    return vec / n


def _topk_cosine(query_vec: np.ndarray, emb: np.ndarray, k: int) -> list[tuple[int, float]]:
    sims = emb @ query_vec
    if k >= len(sims):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def _is_question(text: str) -> bool:
    """
    Simple German-friendly question detector.
    True if:
      - contains '?' OR
      - starts with common question words OR
      - contains patterns like "wie viel / wie viele / darf ich / muss ich / kann ich"
    """
    t = " ".join((text or "").strip().split())
    if not t:
        return False
    low = t.lower()

    if "?" in t:
        return True

    starts = (
        "wie", "was", "wann", "wo", "wer", "wem", "wessen", "warum", "wieso", "weshalb",
        "welche", "welcher", "welches",
        "darf", "muss", "kann", "soll", "brauche", "benötige", "gilt", "gibt es"
    )
    if low.startswith(starts):
        return True

    # Common embedded patterns
    patterns = [
        r"\bwie viel(e|)\b",
        r"\b(darf|muss|kann|soll)\s+ich\b",
        r"\b(brauche|benötige)\s+ich\b",
        r"\bgilt\b",
        r"\bwo\b.*\bfinde\b",
    ]
    return any(re.search(p, low) for p in patterns)


def _contacts_block() -> str:
    return (
        "📩 **Kontakt (falls du keine passende Antwort findest):**\n"
        f"- **Bachelor:** {CONTACTS_FALLBACK['Bachelor']}\n"
        f"- **Master:** {CONTACTS_FALLBACK['Master']}\n"
        f"- **Master AI & Data Science:** {CONTACTS_FALLBACK['Master AI & Data Science']}\n"
    )


# ----------------------------
# Chainlit
# ----------------------------
@cl.on_chat_start
async def on_start():
    # Don't show any "ready" message.
    try:
        _ensure_loaded()
    except Exception as e:
        await cl.Message(
            content=(
                "❌ Ich konnte die FAQ-Daten nicht laden.\n\n"
                f"**Fehler:** {e}\n\n"
                "Bitte prüfe die FAQ-Datei (JSONL) und die Umgebungsvariable `FAQ_FILE`."
            )
        ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    text = (msg.content or "").strip()

    # 1) Non-question detection
    if not _is_question(text):
        await cl.Message(
            content="Bitte stelle eine **konkrete Frage** (z. B. „Wie viele Leistungspunkte brauche ich für den Bachelorabschluss?“)."
        ).send()
        return

    # 2) Ensure data is loaded
    try:
        _ensure_loaded()
    except Exception as e:
        await cl.Message(content=f"❌ Systemfehler beim Laden der Daten: {e}").send()
        return

    qvec = _embed_query(text)
    hits = _topk_cosine(qvec, _EMB, TOP_K)

    best_i, best_score = hits[0]
    best = _QA[best_i]

    # 3) Not found / low confidence
    if best_score < SIM_THRESHOLD:
        await cl.Message(
            content=(
                "Ich habe dazu leider **keine passende Antwort** in den Prüfungsregeln gefunden.\n\n"
                + _contacts_block()
            )
        ).send()
        return

    # 4) Found: return ONLY the answer (no score, no matched question, no alternatives)
    await cl.Message(content=best["answer"]).send()

