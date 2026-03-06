import json
from typing import Dict, Any
from pathlib import Path

def load_chunks_map(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    chunks = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks[obj["chunk_id"]] = obj
    return chunks
