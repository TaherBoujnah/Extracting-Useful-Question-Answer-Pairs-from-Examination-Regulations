# backend/evaluation/entailment.py
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Optional

from backend.evaluation.judgefree_metrics import content_tokens, extract_numbers, jaccard


OLLAMA_BASE = "http://localhost:11434"


def _ollama_generate(model: str, prompt: str, timeout_seconds: int = 120) -> str:
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_predict": 120}}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return (out.get("response") or "").strip()


def entailment_proxy(premise: str, hypothesis: str) -> float:
    """
    Judge-free-ish proxy entailment in [0,1] using:
    - lexical overlap of content tokens
    - numeric consistency: numbers in answer should appear in premise
    - penalize if answer introduces many new tokens
    """
    p = content_tokens(premise)
    h = content_tokens(hypothesis)

    if not h:
        return 0.0
    if not p:
        return 0.0

    base = jaccard(p, h)  # overlap proxy

    # numeric consistency penalty
    p_nums = set(extract_numbers(premise))
    h_nums = set(extract_numbers(hypothesis))
    if h_nums and not (h_nums <= p_nums):
        # if answer introduces new numbers not in premise => strong penalty
        base *= 0.5

    # novelty penalty
    p_set = set(p)
    novel = [t for t in h if t not in p_set]
    novelty_ratio = len(novel) / max(1, len(h))
    if novelty_ratio > 0.6:
        base *= 0.7
    if novelty_ratio > 0.8:
        base *= 0.5

    return max(0.0, min(1.0, base))


def entailment_llm(premise: str, hypothesis: str, *, model: str) -> float:
    """
    Optional LLM-based entailment (still 'no separate judge model' if you reuse an existing local model).
    Returns 1.0 for entails, 0.0 otherwise.
    """
    prompt = (
        "You are verifying if an answer is fully supported by the provided text.\n"
        "Return ONLY one token: ENTAILS or NOT_ENTAILS.\n\n"
        f"TEXT:\n{premise}\n\n"
        f"ANSWER:\n{hypothesis}\n"
    )
    out = _ollama_generate(model, prompt).strip().upper()
    if "ENTAILS" in out and "NOT" not in out:
        return 1.0
    return 0.0
