import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import os
from .utils_2 import ensure_page_image


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
    Minimal payload for LLM-only judgement: provide only raw text lines.
    No geometry (x/y) and no derived features like has_dots or trailing_number.
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
                lines.append({"text": t_norm})

    return {"meta": {"doc_id": doc_id, "page_idx": page_idx}, "lines": lines}


def render_toc_detect_prompt(payload: Dict[str, Any], *, images_only: bool = False) -> str:
    """
    Render a strict JSON-only prompt asking the model to answer if this page is a TOC.
    Expected output JSON: {"is_toc": true|false, "confidence": 0..1}
    """
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if images_only:
        return (
            "You are given one or more full-page images (printed pages).\n"
            "Decide whether the page contains a Table of Contents (TOC) section based ONLY on the images. Ignore any provided text lines.\n"
            "Rules:\n"
            "- Set is_toc=true only if there is clear TOC evidence, such as:\n"
            "  - multiple entries like 'title … page-number',\n"
            "  - consistent leader dots leading to right-aligned page numbers,\n"
            "  - clearly aligned TOC columns, or printed page numbers in headers/footers matching entries.\n"
            "- A few ordinary section headings or numbered list items without page-number/leader evidence should NOT be treated as TOC.\n"
            "- Partial TOC counts only if at least two distinct TOC-like entries are visible.\n"
            "- Judge only from the images. Do not invent or assume extra signals.\n"
            "Output a single JSON object only: {\"is_toc\": boolean, \"confidence\": number}.\n"
            "No commentary and no code fences.\n\n"
            + compact
        )
    else:
        return (
            "You are given a page summary consisting of text lines. If page images are attached, they show the full printed page.\n"
            "Decide whether this page contains a Table of Contents (TOC) section.\n"
            "Rules:\n"
            "- Set is_toc=true only if there is clear TOC evidence, such as:\n"
            "  - multiple entries like 'title … page-number',\n"
            "  - consistent leader dots leading to right-aligned page numbers,\n"
            "  - clearly aligned columns typical of TOC, or printed page numbers in headers/footers matching entries.\n"
            "- A few ordinary section headings or numbered list items without page-number/leader evidence should NOT be treated as TOC.\n"
            "- Partial TOC counts only if at least two distinct TOC-like entries are visible.\n"
            "- Judge only from the provided lines and (if present) the images. Do not invent or assume extra signals.\n"
            "Output a single JSON object only: {\"is_toc\": boolean, \"confidence\": number}.\n"
            "No commentary and no code fences.\n\n"
            + compact
        )


def detect_toc_page(
    llm_call: Callable[[List[Dict[str, Any]], Optional[List[Any]]], str],
    payload: Dict[str, Any],
    images: Optional[List[Any]] = None,
    *,
    min_confidence: Optional[float] = None,
    images_only: bool = False,
) -> Tuple[bool, float]:
    """
    Call the provided llm_call(messages, images)->str with a system+user prompt to detect TOC.
    Returns (is_toc, confidence). If min_confidence is provided, the boolean is_toc is
    thresholded as (raw_is_toc and confidence >= min_confidence). On failure, returns (False, 0.0).
    """
    messages = [
        {"role": "system", "content": "You only output JSON."},
        {"role": "user", "content": render_toc_detect_prompt(payload, images_only=images_only)},
    ]
    try:
        raw = llm_call(messages, images)  # raw JSON string expected
        obj = json.loads(raw)
        is_toc_raw = bool(obj.get("is_toc", False))
        conf = float(obj.get("confidence", 0.0))
        if min_confidence is not None:
            return (is_toc_raw and conf >= float(min_confidence)), conf
        return is_toc_raw, conf
    except Exception:
        return False, 0.0


def find_toc_pages(
    flat_root: Dict[str, Any],
    llm_call: Callable[[List[Dict[str, Any]], Optional[List[Any]]], str],
    *,
    start_page: int = 0,
    max_scan_pages: int = 20,
    payload_builder: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None,
    min_confidence: float = 0.75,
    images_only: bool = False,
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
    builder = payload_builder or (lambda root, pidx: build_toc_page_payload(root, pidx))

    # Cache detections
    cache: Dict[int, Tuple[bool, float]] = {}

    def detect(p: int) -> Tuple[bool, float]:
        if p not in cache:
            payload = builder(flat_root, p)
            cache[p] = detect_toc_page(llm_call, payload, min_confidence=min_confidence, images_only=images_only)
        return cache[p]

    in_run = False
    scanned = 0

    source_dir = flat_root.get("source_path")
    for i, p in enumerate(indices):
        if scanned >= max_scan_pages and not in_run:
            break
        scanned += 1
        # Resolve page image if available
        imgs = None
        if isinstance(source_dir, str) and source_dir:
            img_path = ensure_page_image(source_dir, p)
            if img_path:
                imgs = [img_path]
        cur_text = False
        cur_img = False
        if not images_only:
            cur_text, _ = detect(p)
        if imgs is not None:
            payload = builder(flat_root, p)
            cur_img, _ = detect_toc_page(
                llm_call,
                payload,
                images=imgs,
                min_confidence=min_confidence,
                images_only=images_only,
            )
        # Final decision
        if (images_only and cur_img) or (not images_only and (cur_text or cur_img)):
            toc_pages.append(p)
            in_run = True
        else:
            if in_run:
                # Smart end: end on first non-TOC page once a run has started
                break
            # if not in_run, just continue scanning
            continue

    return toc_pages


# === Consolidation: collapse scattered TOC nodes into a single toc node ===

def _build_segmentation_payload(
    flat_root: Dict[str, Any],
    page_idx: int,
    *,
    include_geometry: bool = True,
    split_concatenated: bool = True,
    long_text_threshold: int = 300,
    max_text_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a payload for line-level selection on a single page. Each line is a candidate
    TOC line with an index 'i' and maps back to a source node via node_idx/node_id.
    """
    # Reuse build_toc_page_payload to get normalized lines
    base = build_toc_page_payload(
        flat_root,
        page_idx,
        include_geometry=include_geometry,
        split_concatenated=split_concatenated,
        long_text_threshold=long_text_threshold,
        max_text_len=max_text_len,
    )
    # Map page nodes by (y sorted) to retrieve node ids; we align by proximity in order.
    # For robustness, we attach the nearest node (in original order) that has same page_idx.
    page_nodes = [
        ch for ch in flat_root.get("children", []) if ch.get("page_idx") == page_idx and ch.get("type") == "text"
    ]
    # Build a simple sequence mapping: lines are in reading order; map to nodes in sequence
    # Note: when a single node was split into multiple lines, we will assign the same node repeatedly.
    lines = base.get("lines", [])
    out_lines: List[Dict[str, Any]] = []
    ni = 0
    for i, ln in enumerate(lines):
        # Advance ni while current node has empty/invalid text
        while ni < len(page_nodes) and not isinstance(page_nodes[ni].get("text"), str):
            ni += 1
        node = page_nodes[ni] if ni < len(page_nodes) else None
        if node is not None:
            node_id = node.get("node_id")
            node_idx = node.get("node_idx")
        else:
            node_id = None
            node_idx = None
        out_lines.append(
            {
                "i": i,
                "text": ln.get("text"),
                "node_id": node_id,
                "node_idx": node_idx,
            }
        )
        # Heuristic: if this line is long, assume it came from the same node; else advance
        # In practice, even advancing each time is acceptable; keep simple: advance on every line
        if ni < len(page_nodes):
            ni = min(ni + 1, len(page_nodes))

    return {"meta": base.get("meta", {}), "lines": out_lines}


def _reindex_flat(
    doc_id: str,
    children: List[Dict[str, Any]],
    source_path: Optional[str],
    preserve_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Reassign node_idx/node_id and rebuild indices while preserving existing root-level metadata.
    - preserve_meta: optional dict of the previous root; copies over its top-level keys except
      for structural keys that are recomputed here (children, indices, doc_id, type, source_path).
    """
    # Reassign node_idx/node_id and rebuild indices
    for i, ch in enumerate(children):
        ch["node_idx"] = i
        ch["node_id"] = f"{doc_id}#{i}"
    root: Dict[str, Any] = {"type": "document", "doc_id": doc_id, "children": children}
    if source_path:
        root["source_path"] = source_path
    by_page: Dict[int, List[int]] = {}
    by_type: Dict[str, List[int]] = {}
    id_to_idx: Dict[str, int] = {}
    for i, ch in enumerate(children):
        p = ch.get("page_idx")
        if isinstance(p, int):
            by_page.setdefault(p, []).append(i)
        t = ch.get("type")
        if isinstance(t, str):
            by_type.setdefault(t, []).append(i)
        id_to_idx[f"{doc_id}#{i}"] = i
    root["indices"] = {"by_page": by_page, "by_type": by_type, "id_to_idx": id_to_idx}
    # Preserve additional metadata from previous root if provided
    if isinstance(preserve_meta, dict):
        for k, v in preserve_meta.items():
            if k in ("children", "indices", "doc_id", "type", "source_path"):
                continue
            # Do not overwrite values we just set unless absent
            if k not in root:
                root[k] = v
    return root

# === Integrated parse+span: ask LLM to return headings plus (start_idx, end_idx) directly ===

def build_toc_pages_payload_with_nodes(
    flat_root: Dict[str, Any],
    pages: Sequence[int],
    *,
    include_geometry: bool = True,
    split_concatenated: bool = True,
    long_text_threshold: int = 300,
    max_text_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a compact multi-page payload for LLM to both parse headings and choose a global
    span to replace. Each page contains lines with mapping back to source nodes (node_idx/node_id).
    """
    doc_id = flat_root.get("doc_id") or "document"
    out_pages: List[Dict[str, Any]] = []
    for p in sorted(set(pages)):
        seg = _build_segmentation_payload(
            flat_root,
            p,
            include_geometry=include_geometry,
            split_concatenated=split_concatenated,
            long_text_threshold=long_text_threshold,
            max_text_len=max_text_len,
        )
        out_pages.append({"page_idx": p, "lines": seg.get("lines", [])})
    return {"meta": {"doc_id": doc_id}, "pages": out_pages}


def render_toc_parse_with_span_prompt(pages_payload: Dict[str, Any]) -> str:
    """
    Ask the LLM to return both a hierarchical TOC tree and a global span to replace.
    Required JSON keys in the output:
      - headings: [ {title: string, level: int>=1, page?: int, children: [...]}, ... ]
      - start_idx: integer (global node_idx inclusive start)
      - end_idx: integer (global node_idx inclusive end)
      - pages: [int,...] (the page indices that truly contain TOC)
    """
    compact = json.dumps(pages_payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are given candidate pages, each with lines (text) mapped to source nodes (node_idx/node_id).\n"
        "Task: (1) Parse a hierarchical Table of Contents (TOC) tree; (2) Choose a single contiguous span\n"
        "of nodes to replace with a toc node.\n"
        "If page images are provided, use them to verify printed page numbers in headers/footers and\n"
        "fill the optional 'page' field for headings when possible. Do not infer from body text.\n"
        "Output a single JSON object only with keys: headings, start_idx, end_idx, pages.\n"
        "- headings: list of TOC headings, each {title: string, level: integer >= 1, page?: integer, children: [...]}.\n"
        "- start_idx/end_idx: integers specifying the inclusive range of node_idx to replace.\n"
        "- pages: list of page indices that truly contain TOC content.\n"
        "No commentary and no code fences.\n\n"
        + compact
    )


def build_toc_tree_and_span_with_llm(
    flat_root: Dict[str, Any],
    candidate_pages: Sequence[int],
    llm_call: Callable[[List[Dict[str, Any]], Optional[List[Any]]], str],
    *,
    include_geometry: bool = True,
    split_concatenated: bool = True,
    long_text_threshold: int = 300,
    max_text_len: Optional[int] = None,
    include_images: bool = True,
) -> Tuple[List[Dict[str, Any]], int, int, List[int]]:
    """
    One-shot LLM call to produce both headings and a global span [start_idx..end_idx].
    Returns (headings, start_idx, end_idx, pages). On failure, returns ([], -1, -1, []).
    """
    payload = build_toc_pages_payload_with_nodes(
        flat_root,
        candidate_pages,
        include_geometry=include_geometry,
        split_concatenated=split_concatenated,
        long_text_threshold=long_text_threshold,
        max_text_len=max_text_len,
    )
    messages = [
        {"role": "system", "content": "You only output JSON."},
        {"role": "user", "content": render_toc_parse_with_span_prompt(payload)},
    ]
    # Optionally attach page images to assist with page number verification
    images = None
    if include_images:
        try:
            source_dir = flat_root.get("source_path")
            if isinstance(source_dir, str) and source_dir:
                imgs: List[str] = []
                for p in sorted(set(candidate_pages)):
                    img = ensure_page_image(source_dir, p)
                    if img:
                        imgs.append(img)
                images = imgs if imgs else None
        except Exception:
            images = None
    try:
        raw = llm_call(messages, images)
        obj = json.loads(raw)
        headings = obj.get("headings") if isinstance(obj.get("headings"), list) else []
        start_idx = int(obj.get("start_idx")) if isinstance(obj.get("start_idx"), int) else -1
        end_idx = int(obj.get("end_idx")) if isinstance(obj.get("end_idx"), int) else -1
        pages = obj.get("pages") if isinstance(obj.get("pages"), list) else []
        # Basic sanity
        if start_idx >= 0 and end_idx >= start_idx and headings is not None:
            return headings, start_idx, end_idx, pages
    except Exception:
        pass
    return [], -1, -1, []


def consolidate_toc_v2(
    flat_root: Dict[str, Any],
    candidate_pages: Sequence[int],
    llm_call: Callable[[List[Dict[str, Any]], Optional[List[Any]]], str],
    *,
    include_geometry: bool = True,
    split_concatenated: bool = True,
    long_text_threshold: int = 300,
    max_text_len: Optional[int] = None,
    include_images: bool = True,
) -> Dict[str, Any]:
    """
    Integrated consolidation: ask LLM to output headings AND [start_idx..end_idx] in one shot.
    Useful when per-page line selection is brittle.
    """
    if not candidate_pages:
        return flat_root
    doc_id = flat_root.get("doc_id") or "document"
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))
    headings, start_idx, end_idx, pages = build_toc_tree_and_span_with_llm(
        flat_root,
        candidate_pages,
        llm_call,
        include_geometry=include_geometry,
        split_concatenated=split_concatenated,
        long_text_threshold=long_text_threshold,
        max_text_len=max_text_len,
        include_images=include_images,
    )
    if not headings:
        return flat_root
    if start_idx < 0 or end_idx < start_idx:
        # Fallback: do nothing
        return flat_root
    toc_node: Dict[str, Any] = {
        "type": "toc",
        "pages": pages if pages else list(sorted(set(candidate_pages))),
        "headings": headings,
        "page_idx": children[start_idx].get("page_idx"),
        "node_level": -1,
    }
    new_children = children[:start_idx] + [toc_node] + children[end_idx + 1 :]
    return _reindex_flat(doc_id, new_children, flat_root.get("source_path"), preserve_meta=flat_root)
