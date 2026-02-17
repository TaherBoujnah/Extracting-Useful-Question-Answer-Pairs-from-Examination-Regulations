import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

INPUT_MD = "data/ALL_Informatik_Exam_Regulations.md"   # your combined file
OUT_JSONL = "data/chunks.jsonl"

# If a chunk gets too large (characters), we split it into sub-chunks with overlap
MAX_CHARS = 9000
OVERLAP_SUBSECTIONS = 1  # overlap 1 subsection when splitting


# ------------------ Helpers ------------------

def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def find_pages(text: str) -> List[int]:
    # collects [Page X] markers inside the chunk
    return sorted({int(x) for x in re.findall(r"\[Page\s+(\d+)\]", text)})

def normalize_ws(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def safe_id(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:80]

def detect_depends_on(text: str) -> List[str]:
    """
    Find references to other § sections.
    Catches patterns like: § 11, §11, § 16 Abs. 2, etc.
    """
    nums = re.findall(r"§\s*(\d{1,2})", text)
    # keep unique, in order
    seen = set()
    out = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(f"§{n}")
    return out

def split_top_sections(full_md: str) -> List[Tuple[str, str]]:
    """
    Splits the combined file into the 5 big parts you created:
    # General Bachelor Rules
    # Bachelor Informatik
    # General Master Rules
    # Master Informatik
    # Master AI Data Science
    """
    # Your combined file was created with "# <name>" markers.
    # We'll split by H1 headings.
    parts = re.split(r"\n#\s+", "\n" + full_md)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # first line is title
        lines = p.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        out.append((title, body))
    return out

def infer_scope_from_top_title(top_title: str) -> Dict[str, str]:
    t = top_title.lower()
    # degree_level
    if "bachelor" in t:
        degree = "Bachelor"
    elif "master" in t:
        degree = "Master"
    else:
        degree = "Unknown"

    # program
    if "ai" in t or "artificial" in t or "data science" in t:
        program = "AI & Data Science"
    elif "informatik" in t:
        # could be general master/bachelor rules too, but those titles are "General ..."
        program = "Informatik"
    elif "general" in t:
        program = "General"
    else:
        program = "Unknown"

    return {"degree_level": degree, "program": program, "parent_section": top_title}

def split_by_paragraphs(text: str) -> List[str]:
    # paragraphs separated by blank lines
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts

def split_into_subchunks_by_subsections(title: str, text: str) -> List[Tuple[str, str]]:
    """
    Split a long § chunk into sub-chunks by Abs./(1)/(2)/etc when present.
    Keeps overlap so linked paragraphs stay meaningful.
    """
    # Detect subsection boundaries like "(1)", "(2)", or "Abs. 1"
    # We'll split at lines that start with "(number)".
    lines = text.splitlines()
    blocks = []
    current = []
    for line in lines:
        if re.match(r"^\(\d+\)\s", line.strip()):
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())

    # If we didn't find subsection structure, fall back to paragraph splitting
    if len(blocks) <= 1:
        blocks = split_by_paragraphs(text)

    # Build subchunks with overlap
    if not blocks:
        return [(title, text)]

    subchunks = []
    i = 0
    while i < len(blocks):
        # accumulate blocks until size limit
        acc = []
        size = 0
        j = i
        while j < len(blocks) and size + len(blocks[j]) < MAX_CHARS:
            acc.append(blocks[j])
            size += len(blocks[j])
            j += 1

        # ensure progress
        if j == i:
            acc = [blocks[i]]
            j = i + 1

        chunk_text = "\n\n".join(acc).strip()
        sub_title = title if (i == 0 and j == len(blocks)) else f"{title} (part {len(subchunks)+1})"
        subchunks.append((sub_title, chunk_text))

        # next window with overlap
        i = max(j - OVERLAP_SUBSECTIONS, i + 1)

    return subchunks

def split_section_into_paragraph_chunks(top_title: str, body: str) -> List[Dict[str, Any]]:
    """
    Prefer splitting by § headings. If none, keep as one (and later split if too big).
    """
    scope = infer_scope_from_top_title(top_title)

    # Try to split by "§ <number>" headings
    # We handle cases where § appears at line start.
    pattern = r"(?m)^(§\s*\d{1,2}\b.*)$"
    matches = list(re.finditer(pattern, body))

    chunks = []
    if not matches:
        # No explicit § headings in this block, keep as one chunk (or later split by size)
        title = top_title
        raw_text = body
        base = {
            "title": title,
            "text": normalize_ws(raw_text),
            "pages": find_pages(raw_text),
            **scope
        }
        chunks.append(base)
        return chunks

    # Build chunks between § headings
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx+1].start() if idx+1 < len(matches) else len(body)
        section_title = m.group(1).strip()
        section_text = body[start:end].strip()
        base = {
            "title": section_title,
            "text": normalize_ws(section_text),
            "pages": find_pages(section_text),
            **scope
        }
        chunks.append(base)

    return chunks

# ------------------ Main build ------------------

def build_chunks(full_md: str) -> List[Dict[str, Any]]:
    top_sections = split_top_sections(full_md)

    # First pass: split by top section -> then by § -> then sub-split long ones with overlap
    raw_chunks: List[Dict[str, Any]] = []
    for top_title, body in top_sections:
        base_chunks = split_section_into_paragraph_chunks(top_title, body)
        for c in base_chunks:
            # Split oversized chunks (usually long §) by subsections/paragraphs with overlap
            if len(c["text"]) > MAX_CHARS:
                subs = split_into_subchunks_by_subsections(c["title"], c["text"])
                for st, stxt in subs:
                    raw_chunks.append({
                        **{k: c[k] for k in ["degree_level", "program", "parent_section"]},
                        "title": st,
                        "text": normalize_ws(stxt),
                        "pages": find_pages(stxt),
                    })
            else:
                raw_chunks.append(c)

    # Assign chunk_ids
    for i, c in enumerate(raw_chunks):
        base = f"{c['degree_level']}_{c['program']}_{safe_id(c['title'])}_{i+1}"
        c["chunk_id"] = base

    # Neighbors (within same parent_section)
    # group by parent_section while keeping original order
    by_parent: Dict[str, List[int]] = {}
    for idx, c in enumerate(raw_chunks):
        by_parent.setdefault(c["parent_section"], []).append(idx)

    for parent, indices in by_parent.items():
        for pos, idx in enumerate(indices):
            prev_id = raw_chunks[indices[pos-1]]["chunk_id"] if pos > 0 else None
            next_id = raw_chunks[indices[pos+1]]["chunk_id"] if pos + 1 < len(indices) else None
            raw_chunks[idx]["neighbors"] = {
                "prev": prev_id,
                "next": next_id
            }

    # depends_on: detect § references and link to chunk_ids where possible
    # Build a lookup: (degree_level, parent_section, "§14") -> chunk_id (best effort)
    # Also a global lookup by "§14" within same degree_level
    sec_lookup_local = {}
    sec_lookup_degree = {}

    for c in raw_chunks:
        # Identify if title starts with "§<num>"
        m = re.match(r"§\s*(\d{1,2})\b", c["title"])
        if m:
            sec = f"§{m.group(1)}"
            key_local = (c["degree_level"], c["parent_section"], sec)
            sec_lookup_local[key_local] = c["chunk_id"]
            key_deg = (c["degree_level"], sec)
            # if multiple, keep first (usually the main one)
            sec_lookup_degree.setdefault(key_deg, c["chunk_id"])

    for c in raw_chunks:
        deps = detect_depends_on(c["text"])
        resolved = []
        for sec in deps:
            # try local first
            key_local = (c["degree_level"], c["parent_section"], sec)
            if key_local in sec_lookup_local:
                resolved.append(sec_lookup_local[key_local])
                continue
            # fallback to degree-level
            key_deg = (c["degree_level"], sec)
            if key_deg in sec_lookup_degree:
                resolved.append(sec_lookup_degree[key_deg])
        # store both raw refs and resolved ids
        c["depends_on_sections"] = deps
        c["depends_on"] = resolved

    return raw_chunks

def write_jsonl(chunks: List[Dict[str, Any]], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

def main():
    md = read_text(INPUT_MD)
    chunks = build_chunks(md)
    write_jsonl(chunks, OUT_JSONL)
    print(f"Wrote {len(chunks)} chunks to {OUT_JSONL}")

if __name__ == "__main__":
    main()
