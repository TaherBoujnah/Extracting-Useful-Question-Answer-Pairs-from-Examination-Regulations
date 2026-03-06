# backend/span_extractor.py
import json
import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_TEXT = DATA_DIR / "exam_regulations.txt"
OUTPUT_SPANS = DATA_DIR / "spans.json"

KEYWORDS = [
    "muss", "müssen", "darf", "dürfen", "kann", "können", "soll", "sollen",
    "frist", "spätestens", "mindestens", "höchstens",
    "antrag", "zulassung", "voraussetzung", "prüfung", "prüfungs", "wiederholung",
    "nachteilsausgleich", "härtefall", "ausnahme", "bewertung",
    "bachelorarbeit", "masterarbeit", "abschlussarbeit", "kreditpunk", "leistungspunkt", "ects"
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SECTION_SPLIT = re.compile(r"(?=§\s*\d+)")


def split_sections(text: str) -> list[str]:
    # section-level split happens only here
    secs = SECTION_SPLIT.split(text)
    return [s.strip() for s in secs if len(s.strip()) > 80]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 0]


def looks_useful(sentence: str) -> bool:
    s = sentence.lower()
    if len(sentence) < 60:
        return False
    return any(k in s for k in KEYWORDS)


def run_span_extractor(limit_sections=None, mode="sentence"):
    if not INPUT_TEXT.exists():
        raise FileNotFoundError(f"Missing {INPUT_TEXT}. Run chunker.py first.")

    text = INPUT_TEXT.read_text(encoding="utf-8")
    sections = split_sections(text)

    if limit_sections is not None:
        sections = sections[:limit_sections]

    spans = []

    for i, sec in enumerate(sections, start=1):
        m = re.match(r"(§\s*\d+)", sec)
        section_id = m.group(1) if m else "Unknown"

        sentences = split_sentences(sec)

        if mode == "sentence":
            for sent in sentences:
                if looks_useful(sent):
                    spans.append({"section": section_id, "span": sent})
        else:
            # window mode merges consecutive useful sentences
            current = []
            for sent in sentences:
                if looks_useful(sent):
                    current.append(sent)
                else:
                    if current:
                        spans.append({"section": section_id, "span": " ".join(current)})
                        current = []
            if current:
                spans.append({"section": section_id, "span": " ".join(current)})

        if i % 25 == 0:
            print(f"✅ processed {i} sections | spans: {len(spans)}")

    OUTPUT_SPANS.write_text(json.dumps(spans, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved {len(spans)} spans → {OUTPUT_SPANS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_sections", type=int, default=None)
    parser.add_argument("--mode", choices=["sentence", "window"], default="sentence")
    args = parser.parse_args()

    run_span_extractor(limit_sections=args.limit_sections, mode=args.mode)

