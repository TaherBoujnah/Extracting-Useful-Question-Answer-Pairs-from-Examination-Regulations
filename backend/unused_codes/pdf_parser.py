# backend/pdf_parser.py
import json
from pathlib import Path
import pdfplumber

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

PDF_FILES = list(DATA_DIR.glob("*.pdf"))
if not PDF_FILES:
    raise FileNotFoundError(f"No PDF found in: {DATA_DIR}")
PDF_PATH = PDF_FILES[0]

# Store raw extraction as JSONL: one record per page
OUTPUT_RAW_JSONL = DATA_DIR / "raw_pages.jsonl"


def table_to_text(table):
    if not table or len(table) < 2:
        return ""
    header = table[0]
    rows = table[1:]
    lines = []
    for row in rows:
        if not any(row):
            continue
        parts = []
        for i, cell in enumerate(row):
            if cell and i < len(header):
                parts.append(f"{header[i]}: {cell}")
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def run_parser():
    OUTPUT_RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)

    print(f"📄 Extracting from: {PDF_PATH.name}")
    with pdfplumber.open(PDF_PATH) as pdf, OUTPUT_RAW_JSONL.open("w", encoding="utf-8") as f:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            table_texts = []
            for t in page.extract_tables():
                tt = table_to_text(t)
                if tt:
                    table_texts.append(tt)

            record = {
                "page": page_number,
                "text": text,
                "tables": table_texts
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if page_number % 10 == 0:
                print(f"✅ extracted page {page_number}/{len(pdf.pages)}")

    print(f"✅ Saved raw extraction → {OUTPUT_RAW_JSONL}")


if __name__ == "__main__":
    run_parser()
