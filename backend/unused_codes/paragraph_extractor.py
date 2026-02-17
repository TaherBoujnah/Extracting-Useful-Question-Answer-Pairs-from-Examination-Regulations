# backend/paragraph_extractor.py
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_TEXT = DATA_DIR / "exam_regulations.txt"
OUTPUT_JSON = DATA_DIR / "paragraphs.json"

# ---- document-level anchors ----
BSC_DOC_ANCHOR = r"Neubekanntmachung\s+der\s+Prüfungsordnung.*Bachelor of Science"
MSC_DOC_ANCHOR = r"Neubekanntmachung\s+der\s+Prüfungsordnung.*Master of Science"

# ---- appendix anchors (program-specific) ----
APP_BSC_INF = r"Fachspezifischer\s+Anhang\s+zur\s+Prüfungsordnung.*Bachelor.*Informatik"
APP_MSC_INF = r"Fachspezifischer\s+Anhang.*Master.*Informatik"
APP_MSC_AIDS = r"Fachspezifischer\s+Anhang.*Artificial\s+Intelligence\s+and\s+Data\s+Science"

# Any appendix starts like this (used to find the end boundary)
ANY_APPENDIX = r"Fachspezifischer\s+Anhang"

TRACKS = [
    ("bsc_informatik", APP_BSC_INF),
    ("msc_informatik", APP_MSC_INF),
    ("msc_ai_ds", APP_MSC_AIDS),
]


# ------------------ helpers ------------------
def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_meta_paragraphs(paras: list[str]) -> list[str]:
    """
    Remove TOC/header/editorial paragraphs that are not regulation content.
    """
    out = []
    for p in paras:
        low = p.lower()
        # obvious TOC block
        if "inhalt" in low and "seite" in low:
            continue
        # editorial header
        if "herausgeber" in low and "redaktion" in low:
            continue
        # phone/address blocks
        if "telefon" in low and "universitätsstraße" in low:
            continue
        # pure “lists with page numbers”
        if p.count(",") >= 4 and re.search(r"\b\d{1,3}\s*$", p.strip()):
            continue
        out.append(p)
    return out


def split_paragraphs(text: str) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in paras if len(p.strip()) >= 120]
    paras = drop_meta_paragraphs(paras)

    # drop table-heavy paragraphs
    cleaned = []
    for p in paras:
        digit_ratio = sum(ch.isdigit() for ch in p) / max(1, len(p))
        if digit_ratio > 0.18:
            continue
        if p.strip().startswith("[TABELLE]"):
            continue
        cleaned.append(p)

    return cleaned


def find_doc_block(text: str, doc_anchor: str, next_anchor: str | None) -> str:
    """
    Extract the full doc block for Bachelor or Master.
    """
    m = re.search(doc_anchor, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    start = m.start()
    tail = text[start:]

    if not next_anchor:
        return tail.strip()

    m2 = re.search(next_anchor, tail[500:], flags=re.IGNORECASE | re.DOTALL)
    end = 500 + m2.start() if m2 else len(tail)
    return tail[:end].strip()


def find_best_appendix_start(doc_block: str, appendix_anchor: str) -> int | None:
    """
    Find best appendix occurrence (avoid TOC mentions).
    We prefer occurrences that contain 'Zu §' nearby (= real appendix text).
    """
    candidates = []
    for m in re.finditer(appendix_anchor, doc_block, flags=re.IGNORECASE | re.DOTALL):
        s = m.start()
        window = doc_block[s:s+900]
        score = 0
        if "zu §" in window.lower() or "zu §" in window.lower().replace(" ", ""):
            score += 4
        if window.count("\n") >= 3:
            score += 2
        # penalize “listing line”
        first_line = window.split("\n", 1)[0]
        if first_line.count(",") >= 3:
            score -= 5
        if re.search(r"\b\d{1,3}\s*$", first_line.strip()):
            score -= 4
        candidates.append((score, s))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][1]


def extract_appendix(doc_block: str, appendix_anchor: str) -> str:
    """
    Extract ONLY this appendix:
    start = appendix_anchor
    end = next ANY_APPENDIX after start
    """
    start = find_best_appendix_start(doc_block, appendix_anchor)
    if start is None:
        return ""

    tail = doc_block[start:]
    m_end = re.search(ANY_APPENDIX, tail[400:], flags=re.IGNORECASE | re.DOTALL)
    end = 400 + m_end.start() if m_end else len(tail)
    return tail[:end].strip()


def dedupe(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for it in items:
        key = (it["track"], re.sub(r"\s+", " ", it["paragraph"]).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ------------------ main ------------------
def run_paragraph_extractor():
    if not INPUT_TEXT.exists():
        raise FileNotFoundError(f"Missing {INPUT_TEXT}")

    text = normalize(INPUT_TEXT.read_text(encoding="utf-8"))

    # Get Bachelor and Master document blocks
    bsc_block = find_doc_block(text, BSC_DOC_ANCHOR, MSC_DOC_ANCHOR)
    msc_block = find_doc_block(text, MSC_DOC_ANCHOR, None)

    if not bsc_block:
        print("⚠️ Bachelor block not found. Adjust BSC_DOC_ANCHOR.")
    if not msc_block:
        print("⚠️ Master block not found. Adjust MSC_DOC_ANCHOR.")

    out = []

    # Optional: include general rules (before first appendix) per doc
    if bsc_block:
        general_bsc = re.split(ANY_APPENDIX, bsc_block, maxsplit=1, flags=re.IGNORECASE)[0]
        for p in split_paragraphs(general_bsc):
            out.append({"track": "bsc_informatik", "paragraph": p})

    if msc_block:
        general_msc = re.split(ANY_APPENDIX, msc_block, maxsplit=1, flags=re.IGNORECASE)[0]
        for p in split_paragraphs(general_msc):
            out.append({"track": "msc_informatik", "paragraph": p})
            out.append({"track": "msc_ai_ds", "paragraph": p})

    # Program-specific appendices (ONLY)
    if bsc_block:
        app = extract_appendix(bsc_block, APP_BSC_INF)
        for p in split_paragraphs(app):
            out.append({"track": "bsc_informatik", "paragraph": p})

    if msc_block:
        app = extract_appendix(msc_block, APP_MSC_INF)
        for p in split_paragraphs(app):
            out.append({"track": "msc_informatik", "paragraph": p})

        app = extract_appendix(msc_block, APP_MSC_AIDS)
        for p in split_paragraphs(app):
            out.append({"track": "msc_ai_ds", "paragraph": p})

    out = dedupe(out)

    OUTPUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved {len(out)} paragraphs to {OUTPUT_JSON}")


if __name__ == "__main__":
    run_paragraph_extractor()


