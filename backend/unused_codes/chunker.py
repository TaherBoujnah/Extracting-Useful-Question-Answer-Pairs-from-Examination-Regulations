# backend/chunker.py
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_RAW_JSONL = DATA_DIR / "raw_pages.jsonl"
OUTPUT_TEXT = DATA_DIR / "exam_regulations.txt"


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove standalone page numbers
    text = re.sub(r"\n\d+\s*\n", "\n", text)

    # Fix hyphenated line breaks
    text = re.sub(r"-\n", "", text)

    # Normalize whitespace
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def run_chunker():
    if not INPUT_RAW_JSONL.exists():
        raise FileNotFoundError(f"Missing {INPUT_RAW_JSONL}. Run pdf_parser.py first.")

    parts = []
    with INPUT_RAW_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page_text = rec.get("text", "")
            tables = rec.get("tables", [])

            parts.append(page_text)
            for t in tables:
                parts.append("\n[TABELLE]\n" + t + "\n")

    raw = "\n\n".join(parts)
    cleaned = clean_text(raw)

    OUTPUT_TEXT.write_text(cleaned, encoding="utf-8")
    print(f"✅ Saved cleaned text → {OUTPUT_TEXT}")


if __name__ == "__main__":
    run_chunker()
