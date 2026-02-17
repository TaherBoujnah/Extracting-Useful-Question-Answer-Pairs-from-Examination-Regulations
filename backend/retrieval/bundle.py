from typing import List, Dict, Any

def bundle_chunk_ids(
    seed_chunk_ids: List[str],
    chunks_map: Dict[str, Dict[str, Any]],
    add_neighbors: bool = True,
    add_deps: bool = True,
) -> List[Dict[str, Any]]:
    """
    Expands retrieved chunk ids by including:
    - prev/next neighbors
    - depends_on chunk ids
    Then returns the actual chunk objects in a stable order.
    """
    out = set(seed_chunk_ids)

    for cid in list(out):
        c = chunks_map.get(cid)
        if not c:
            continue

        if add_neighbors:
            n = c.get("neighbors", {})
            prev_id = n.get("prev")
            next_id = n.get("next")
            if prev_id:
                out.add(prev_id)
            if next_id:
                out.add(next_id)

        if add_deps:
            for dep in c.get("depends_on", []):
                if dep:
                    out.add(dep)

    def sort_key(cid: str):
        c = chunks_map[cid]
        pages = c.get("pages") or [9999]
        return (min(pages), c.get("parent_section", ""), c.get("title", ""))

    ordered_ids = sorted([cid for cid in out if cid in chunks_map], key=sort_key)
    return [chunks_map[cid] for cid in ordered_ids]
