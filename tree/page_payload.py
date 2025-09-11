from typing import Any, Dict, List, Optional


def _truncate_text(s: str, max_len: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def build_page_payload(
    root: Dict[str, Any],
    page_idx: int,
    *,
    max_text: int = 160,
    max_table_lines: int = 6,
    include_bbox: bool = True,
    page_image_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct a compact, model-friendly payload for a given page from a flat doctree.

    - Includes minimal fields to guide ordering/leveling decisions while avoiding full text/HTML.
    - Does NOT perform any network or file I/O.

    Args:
      root: The doctree root (type=document) with keys: doc_id, children, indices?
      page_idx: Target page index to summarize.
      max_text: Max characters for text/equation snippets.
      max_table_lines: Max lines from table_text to include.
      include_bbox: If True, include outline/bbox when available.
      page_image_ref: Optional reference/URI/path to the page image for VLM.

    Returns:
      dict payload with fields: { meta, page_image, elements }
    """
    doc_id = root.get("doc_id") or "document"
    children: List[Dict[str, Any]] = list(root.get("children", []))
    elems: List[Dict[str, Any]] = []

    for ch in children:
        if ch.get("page_idx") != page_idx:
            continue
        typ = ch.get("type")
        if not isinstance(typ, str):
            continue
        node = {
            "node_id": ch.get("node_id") or f"{doc_id}#{ch.get('node_idx', '')}",
            "type": typ,
            "page_idx": page_idx,
        }
        if include_bbox and isinstance(ch.get("outline"), list):
            node["outline"] = ch["outline"]

        if typ == "text":
            node["snippet"] = _truncate_text(ch.get("text", ""), max_text)
        elif typ == "equation":
            node["snippet"] = _truncate_text(ch.get("text", ""), max_text)
        elif typ == "table":
            tt = ch.get("table_text")
            if isinstance(tt, str) and tt.strip():
                lines = [ln.rstrip() for ln in tt.splitlines() if ln.strip()]
                node["snippet"] = "\n".join(lines[:max_table_lines])
            else:
                node["snippet"] = ""
        elif typ == "image":
            # Keep it minimal (description or OCR text if present)
            snippet = ch.get("description") or ch.get("text") or ""
            node["snippet"] = _truncate_text(snippet, max_text // 2)
        else:
            node["snippet"] = _truncate_text(ch.get("text", ""), max_text)

        elems.append(node)

    payload: Dict[str, Any] = {
        "meta": {"doc_id": doc_id, "page_idx": page_idx},
        "page_image": page_image_ref,
        "elements": elems,
    }
    return payload

