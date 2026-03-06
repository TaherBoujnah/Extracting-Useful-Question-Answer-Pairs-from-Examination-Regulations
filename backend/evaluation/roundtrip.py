# backend/evaluation/roundtrip.py
from __future__ import annotations

import json
import urllib.request

from backend.evaluation.judgefree_metrics import content_tokens, jaccard

OLLAMA_BASE = "http://localhost:11434"


def ollama_generate(model: str, prompt: str, timeout_seconds: int = 180, temperature: float = 0.0, num_predict: int = 140) -> str:
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return (out.get("response") or "").strip()


def roundtrip_score(question: str, answer: str, evidence: str, *, model: str) -> tuple[float, str]:
    """
    Return (similarity, regenerated_question)
    """
    prompt = (
        "Du bekommst einen Textausschnitt und eine Antwort. "
        "Formuliere eine einzelne, realistische Studentenfrage, "
        "die genau zu dieser Antwort passt.\n\n"
        f"TEXT:\n{evidence}\n\n"
        f"ANTWORT:\n{answer}\n\n"
        "Gib NUR die Frage aus."
    )
    regen = ollama_generate(model, prompt, temperature=0.0, num_predict=80)

    q1 = content_tokens(question)
    q2 = content_tokens(regen)
    sim = jaccard(q1, q2)
    return sim, regen


def roundtrip_ok(similarity: float, threshold: float = 0.35) -> bool:
    return similarity >= threshold
