import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _page_nodes(flat_root: Dict[str, Any], page_idx: int) -> List[Dict[str, Any]]:
    return [ch for ch in flat_root.get("children", []) if ch.get("page_idx") == page_idx]


def _norm_text(s: str, max_len: Optional[int] = 200) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    if max_len is not None and len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def split_concatenated_toc_text(text: str) -> List[str]:
    """
    Split a very long concatenated TOC blob into candidate lines based on
    common heading-start patterns (numeric or roman numbered). If fewer than
    2 start positions are found, return the original text as a single line.
    This function does NOT judge TOC; it only segments for LLM consumption.
    """
    s = (text or "").strip()
    if not s:
        return []

    starts: List[int] = [0]
    patterns = [
        r"(?:(?<=\s)|^)(\d{1,3}(?:\.\d{1,3})*)\s+",       # 1  / 1.2 / 3.10
        r"(?:(?<=\s)|^)([IVXLCM]{1,6})\.\s+",               # I. / IV. / X.
        # Note: we intentionally do NOT enable A. / B. to avoid over-splitting
    ]
    for pat in patterns:
        for m in re.finditer(pat, s):
            idx = m.start(1)
            if idx not in starts:
                starts.append(idx)
    starts = sorted(set(starts))
    # Require at least 2 distinct starts (otherwise it's likely a single line)
    if len(starts) <= 1:
        return [s]
    # Build segments
    segments: List[str] = []
    for i, pos in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(s)
        seg = s[pos:end].strip()
        if seg:
            segments.append(seg)
    return segments or [s]


def build_toc_page_payload(
    flat_root: Dict[str, Any],
    page_idx: int,
    *,
    include_geometry: bool = True,
    split_concatenated: bool = True,
    long_text_threshold: int = 300,
    max_text_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Minimal payload for LLM-only judgement: provide only raw text lines
    (and optional geometry y/x). No derived features like has_dots or trailing_number.
    """
    doc_id = flat_root.get("doc_id") or "document"
    nodes = _page_nodes(flat_root, page_idx)
    lines: List[Dict[str, Any]] = []
    for n in nodes:
        if n.get("type") != "text":
            continue
        raw = (n.get("text", "") or "").strip()
        if not raw:
            continue
        outline = n.get("outline") if include_geometry else None
        y = outline[1] if isinstance(outline, (list, tuple)) and len(outline) >= 2 else None
        x = outline[0] if isinstance(outline, (list, tuple)) and len(outline) >= 1 else None
        # Optionally split very long concatenated blobs into candidate lines
        texts: List[str]
        if split_concatenated and len(raw) >= long_text_threshold:
            parts = split_concatenated_toc_text(raw)
            texts = parts if len(parts) >= 2 else [raw]
        else:
            texts = [raw]
        for t in texts:
            t_norm = _norm_text(t, max_text_len)
            if t_norm:
                lines.append({"text": t_norm, "y": y, "x": x})

    lines.sort(key=lambda d: (d["y"] if isinstance(d.get("y"), (int, float)) else 1e9))
    return {"meta": {"doc_id": doc_id, "page_idx": page_idx}, "lines": lines}


def render_toc_detect_prompt(payload: Dict[str, Any]) -> str:
    """
    Render a strict JSON-only prompt asking the model to answer if this page is a TOC.
    Expected output JSON: {"is_toc": true|false, "confidence": 0..1}
    """
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are given a page summary with text lines (no local heuristics).\n"
        "Decide if this page is a table of contents (TOC).\n"
        "Output a single JSON object only: {\"is_toc\": boolean, \"confidence\": number}.\n"
        "No commentary and no code fences.\n\n"
        + compact
    )


def detect_toc_page(llm_call: Callable[[List[Dict[str, Any]], Optional[List[Any]]], str], payload: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Call the provided llm_call(messages, images=None)->str with a system+user prompt to detect TOC.
    Returns (is_toc, confidence). On failure, returns (False, 0.0).
    """
    messages = [
        {"role": "system", "content": "You only output JSON."},
        {"role": "user", "content": render_toc_detect_prompt(payload)},
    ]
    try:
        raw = llm_call(messages, None)  # raw JSON string expected
        obj = json.loads(raw)
        return bool(obj.get("is_toc", False)), float(obj.get("confidence", 0.0))
    except Exception:
        return False, 0.0


def find_toc_pages(
    flat_root: Dict[str, Any],
    llm_call: Callable[[List[Dict[str, Any]], Optional[List[Any]]], str],
    *,
    start_page: int = 0,
    max_scan_pages: int = 20,
    payload_builder: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None,
) -> List[int]:
    """
    Scan pages from start_page forward to find consecutive TOC pages.
    Logic:
      - Continuity tracking: remember previous page detection.
      - Smart stop: if scanned beyond max_scan_pages and previous page is not TOC, stop.
      - TOC end: when current is not TOC but previous is TOC, stop.
    Returns the list of detected TOC page indices (consecutive block).
    """
    indices = sorted({ch.get("page_idx") for ch in flat_root.get("children", []) if isinstance(ch.get("page_idx"), int)})
    indices = [p for p in indices if p >= start_page]
    toc_pages: List[int] = []
    last_is_toc = False
    builder = payload_builder or (lambda root, pidx: build_toc_page_payload(root, pidx))
    for i, p in enumerate(indices):
        if i >= max_scan_pages and not last_is_toc:
            break
        payload = builder(flat_root, p)
        is_toc, _ = detect_toc_page(llm_call, payload)
        if is_toc:
            toc_pages.append(p)
            last_is_toc = True
        else:
            # if previous was TOC and current is not → TOC block ended
            if last_is_toc:
                break
            last_is_toc = False
    return toc_pages


def render_toc_parse_prompt(pages_payload: List[Dict[str, Any]]) -> str:
    """
    Render a strict JSON-only prompt to parse multiple TOC pages into a hierarchical tree.
    Expected output JSON:
      {
        "headings": [
          {"title": str, "level": int>=1, "page": int?, "children": [...]},
          ...
        ]
      }
    """
    compact = json.dumps(pages_payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are given several pages of a document's table of contents (TOC).\n"
        "Parse them into a hierarchical JSON tree with heading levels.\n"
        "Output JSON with a single key 'headings': a list of heading objects.\n"
        "Each heading object: {title: string, level: integer >= 1, page: integer? (original page number if present), children: [heading...]}.\n"
        "Do not include any extra keys. Only return JSON, no code fences.\n\n"
        + compact
    )


def build_toc_tree_with_llm(
    flat_root: Dict[str, Any],
    toc_pages: Sequence[int],
    llm_call: Callable[[List[Dict[str, Any]], Optional[List[Any]]], str],
) -> Dict[str, Any]:
    """
    Build a TOC tree using multiple detected TOC pages.
    Returns {type: "toc", doc_id, pages: [..indices..], headings: [...]}
    """
    # Use LLM-only payload for parsing as well
    payloads = [build_toc_page_payload(flat_root, p) for p in toc_pages]
    messages = [
        {"role": "system", "content": "You only output JSON."},
        {"role": "user", "content": render_toc_parse_prompt(payloads)},
    ]
    headings: List[Dict[str, Any]] = []
    try:
        raw = llm_call(messages, None)
        obj = json.loads(raw)
        hs = obj.get("headings")
        if isinstance(hs, list):
            headings = hs
    except Exception:
        headings = []

    toc_root: Dict[str, Any] = {
        "type": "toc",
        "doc_id": flat_root.get("doc_id") or "document",
        "pages": list(toc_pages),
        "headings": headings,
    }
    if flat_root.get("source_path"):
        toc_root["source_path"] = flat_root["source_path"]
    return toc_root
