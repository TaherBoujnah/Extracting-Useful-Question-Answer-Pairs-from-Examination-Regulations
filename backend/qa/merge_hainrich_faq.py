import argparse
import json
import html
import re
from pathlib import Path
from typing import List, Tuple


def load_jsonl_qas(path: Path) -> List[Tuple[str, str]]:
    qas = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q and a:
                qas.append((q, a))
    return qas


def make_panel(i: int, q: str, a: str) -> str:
    q = html.escape(q)
    a = html.escape(a)
    return f"""
  <div class="panel">
    <button id="gen_q{i}">
      <span class="text-box">{q}</span>
    </button>
    <div class="panel-body">
      <div class="ce-bodytext">{a}</div>
    </div>
  </div>
"""


def extract_existing_questions(html_text: str) -> set:
    # Grab question texts from existing FAQ panels
    existing = set()
    for m in re.finditer(r'<span\s+class="text-box"\s*>(.*?)</span>', html_text, flags=re.DOTALL | re.IGNORECASE):
        existing.add(html.unescape(m.group(1)).strip())
    return existing


def find_first_panel_group_bounds(text: str) -> Tuple[int, int]:
    """
    Returns (start_index_of_panel_group_opening_tag, end_index_after_matching_closing_div)
    for the FIRST <div class="panel-group"...> ... </div>.
    We locate the opening <div ...> then count nested <div> ... </div>.
    """
    # Find the opening tag of the first panel-group div
    m = re.search(r'<div[^>]*class\s*=\s*["\'][^"\']*\bpanel-group\b[^"\']*["\'][^>]*>', text, flags=re.IGNORECASE)
    if not m:
        raise SystemExit("Could not find <div class='panel-group'> in base FAQ.html")

    start = m.start()
    pos = m.end()

    # Now walk forward counting <div ...> and </div>
    open_divs = 1
    tag_re = re.compile(r'</div\s*>|<div\b', flags=re.IGNORECASE)

    for t in tag_re.finditer(text, pos):
        token = t.group(0).lower()
        if token.startswith("<div"):
            open_divs += 1
        else:
            open_divs -= 1

        if open_divs == 0:
            end = t.end()
            return start, end

    raise SystemExit("Could not find the matching closing </div> for the first panel-group.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_faq_html", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_html", required=True)
    ap.add_argument("--dedup_by_question_text", action="store_true")
    args = ap.parse_args()

    base_path = Path(args.base_faq_html)
    text = base_path.read_text(encoding="utf-8", errors="replace")

    # Find the first panel-group block precisely
    pg_start, pg_end = find_first_panel_group_bounds(text)
    panel_group_block = text[pg_start:pg_end]

    existing_questions = extract_existing_questions(panel_group_block) if args.dedup_by_question_text else set()

    qas = load_jsonl_qas(Path(args.input_jsonl))
    panels = []
    added = 0
    for i, (q, a) in enumerate(qas):
        if args.dedup_by_question_text and q in existing_questions:
            continue
        panels.append(make_panel(i, q, a))
        added += 1

    insertion = "\n<!-- GENERATED FAQ START -->\n" + "".join(panels) + "\n<!-- GENERATED FAQ END -->\n"

    # Insert right before the closing </div> of the panel-group block
    # pg_end points right AFTER that closing </div>, so insert at (pg_end - len("</div>")) is risky.
    # Instead, we insert just before the last "</div>" within the panel_group_block.
    last_close = panel_group_block.lower().rfind("</div>")
    if last_close == -1:
        raise SystemExit("panel-group block had no </div> close tag (unexpected).")

    new_panel_group_block = panel_group_block[:last_close] + insertion + panel_group_block[last_close:]
    merged = text[:pg_start] + new_panel_group_block + text[pg_end:]

    out_path = Path(args.out_html)
    out_path.write_text(merged, encoding="utf-8")
    print(f"✅ Merged {added} generated QAs into: {out_path}")


if __name__ == "__main__":
    main()