# backend/evaluation/judgefree_metrics.py
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+", re.UNICODE)

GERMAN_STOPWORDS = {
    "der", "die", "das", "ein", "eine", "einer", "eines", "einem",
    "und", "oder", "aber", "wenn", "dann", "weil", "dass", "da", "denn",
    "zu", "zum", "zur", "im", "in", "auf", "an", "am", "aus", "bei", "mit",
    "von", "für", "über", "unter", "nach", "vor", "während", "ohne", "gegen",
    "ist", "sind", "war", "waren", "sein", "hat", "haben", "wird", "werden",
    "kann", "können", "muss", "müssen", "darf", "dürfen", "soll", "sollen",
    "nicht", "nur", "auch", "noch", "schon", "als", "wie", "wo", "was", "wer",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "man",
}

def tokenize(text: str, *, lowercase: bool = True) -> list[str]:
    if not text:
        return []
    s = text.strip()
    if lowercase:
        s = s.lower()
    return _WORD_RE.findall(s)

def content_tokens(text: str) -> list[str]:
    toks = tokenize(text)
    return [t for t in toks if t not in GERMAN_STOPWORDS and len(t) >= 2]

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def token_f1(pred: str, ref: str) -> float:
    """
    Token-level F1 on CONTENT tokens (stopword-filtered).
    This is a judge-free proxy: "is the answer supported by retrieved evidence?"
    """
    p = content_tokens(pred)
    r = content_tokens(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0

    cp, cr = Counter(p), Counter(r)
    overlap = sum((cp & cr).values())
    prec = overlap / max(1, sum(cp.values()))
    rec = overlap / max(1, sum(cr.values()))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def coverage(answer: str, evidence: str) -> float:
    """
    Coverage = fraction of content tokens from answer that appear in evidence.
    """
    a = content_tokens(answer)
    if not a:
        return 0.0
    e = set(content_tokens(evidence))
    hit = sum(1 for t in a if t in e)
    return hit / max(1, len(a))

def hallucination_rate(answer: str, evidence: str) -> float:
    """
    Hallucination proxy = 1 - coverage.
    """
    return 1.0 - coverage(answer, evidence)

def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", text or "")

