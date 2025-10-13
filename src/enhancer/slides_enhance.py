import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from ..utils.llm_clients import gpt_llm_call
from ..utils.utils_2 import ensure_page_image


def _loose_json_parse(s: str) -> Optional[dict]:
    """Try strict JSON parse; fallback to extracting the first balanced {...}."""
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    frag = s[start:i+1]
                    return json.loads(frag)
        return None
    except Exception:
        return None


def _vlm_title_for_page(
    source_dir: str,
    page_idx: int,
    *,
    llm_model: str = "gpt-4o",
    llm_call = gpt_llm_call,
) -> Optional[Dict[str, Any]]:
    """
    Ask a VLM to detect the main slide title. If no explicit title exists,
    the model should generate a concise title. Returns a dict:
      {"title": str, "confidence": float, "generated": bool}
    or None on failure.
    """
    img = ensure_page_image(source_dir, page_idx)
    if not img:
        return None

    messages = [
        {"role": "system", "content": "You only output JSON."},
        {
            "role": "user",
            "content": (
                "You are given one presentation slide image.\n"
                "Task: return the slide's main title. If no clear title text exists,\n"
                "generate a concise, appropriate title that summarizes the slide.\n"
                "Rules:\n"
                "- Keep the title short (3–10 words) and in the slide's language.\n"
                "- Do not include page numbers, dates, watermarks, or template boilerplate.\n"
                "- If you generated the title (not explicitly detected), set generated=true.\n"
                "- confidence in [0,1]: higher when an exact title text is clearly present.\n"
                "Output strictly one JSON object: {\"title\": string, \"confidence\": number, \"generated\": boolean}.\n"
            ),
        },
    ]
    try:
        raw = llm_call(messages, images=[img], model=llm_model, json_mode=True)
        obj = _loose_json_parse(raw) or {}
        title = obj.get("title")
        if not isinstance(title, str) or not title.strip():
            return None
        conf = obj.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        gen = bool(obj.get("generated", False))
        return {"title": title.strip(), "confidence": conf_f, "generated": gen}
    except Exception:
        return None


def _heuristic_title_from_page_text(children: List[Dict[str, Any]], page_idx: int) -> Optional[str]:
    """
    Fallback: choose the first short-ish text line on the page that looks like a title.
    Simple heuristics only; avoids bullet markers and long paragraphs.
    """
    for n in children:
        if n.get("page_idx") != page_idx or n.get("type") != "text":
            continue
        s = (n.get("text") or "").strip()
        if not s:
            continue
        # skip obvious bullet/list markers
        if s[:2] in ("- ", "• ", "· ", "* ") or s[:3].lower() in ("1) ", "a) "):
            continue
        # prefer shorter candidates
        if 8 <= len(s) <= 80:
            return s
        if len(s) < 8:
            # keep looking for a better one
            continue
        # if it's longer than 80, probably a paragraph; skip
    return None


def _reindex_nested(doc_id: str, root: Dict[str, Any]) -> Dict[str, Any]:
    i = 0
    def visit(n: Dict[str, Any]):
        nonlocal i
        n["node_idx"] = i
        n["node_id"] = f"{doc_id}#{i}"
        i += 1
        kids = n.get("children")
        if isinstance(kids, list):
            for ch in kids:
                if isinstance(ch, dict):
                    visit(ch)
    kids = root.get("children") if isinstance(root.get("children"), list) else []
    for ch in kids:
        if isinstance(ch, dict):
            visit(ch)
    return root


def assign_levels_slides_on_root(
    flat_root: Dict[str, Any],
    *,
    llm_model: str = "gpt-4o",
    llm_call = gpt_llm_call,
) -> Dict[str, Any]:
    """
    Construct a simple slide hierarchy:
      - L1: synthetic slide container per page (text: "Slide {n}")
      - L2: slide title via VLM (detect or generate). If VLM fails, heuristic fallback or "Slide {n}".
      - L3: all MinerU items on this page (text/image/table/equation) as children content.

    Returns a new flat root with reindexed children.
    """
    if not isinstance(flat_root, dict):
        return flat_root

    source_dir = flat_root.get("source_path") if isinstance(flat_root.get("source_path"), str) else None
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))
    pages = sorted({int(ch.get("page_idx")) for ch in children if isinstance(ch.get("page_idx"), int)})

    new_children: List[Dict[str, Any]] = []
    for idx, p in enumerate(pages, start=1):
        # Title via VLM (with heuristic fallback)
        title_obj: Optional[Dict[str, Any]] = None
        if source_dir:
            title_obj = _vlm_title_for_page(source_dir, p, llm_model=llm_model, llm_call=llm_call)
        if not title_obj:
            ht = _heuristic_title_from_page_text(children, p)
            if ht:
                title_obj = {"title": ht, "confidence": 0.55, "generated": False}
            else:
                title_obj = {"title": f"Slide {idx}", "confidence": 0.4, "generated": True}
        # Collect page content items
        page_children: List[Dict[str, Any]] = []
        for n in children:
            if n.get("page_idx") != p:
                continue
            obj = dict(n)
            meta = dict(obj.get("heading_meta") or {})
            if "via" not in meta:
                meta["via"] = "slides_content"
            obj["heading_meta"] = meta
            page_children.append(obj)

        # Ensure a page image is available and store a relative link
        page_img = None
        if source_dir:
            try:
                img_path = ensure_page_image(source_dir, p)
                if img_path and os.path.isabs(img_path) and os.path.commonprefix([os.path.abspath(source_dir), os.path.abspath(img_path)]):
                    page_img = os.path.relpath(img_path, start=source_dir)
                else:
                    page_img = img_path
            except Exception:
                page_img = None

        slide_container = {
            "type": "slide",
            "page_idx": p,
            "slide_number": idx,
            "node_level": 1,  # page-level container
            "heading_meta": {"via": "slides_container", "confidence": 1.0},
            **({"page_image": page_img} if page_img else {}),
            "children": [
                {
                    "type": "heading",
                    "text": title_obj.get("title"),
                    "page_idx": p,
                    "node_level": 2,
                    "heading_meta": {
                        "via": "vlm" if source_dir else "heuristic",
                        "confidence": float(title_obj.get("confidence", 0.0)),
                        "generated": bool(title_obj.get("generated", False)),
                    },
                },
                *page_children,
            ],
        }
        new_children.append(slide_container)

    doc_id = flat_root.get("doc_id") or "document"
    source_path = flat_root.get("source_path") if isinstance(flat_root.get("source_path"), str) else None
    new_root: Dict[str, Any] = {
        "type": "document",
        "doc_id": doc_id,
        "source_path": source_path,
        "children": new_children,
        "mode": "slides",
    }
    # Preserve top-level extras from original flat_root
    for k, v in flat_root.items():
        if k in ("children", "doc_id", "source_path", "type", "indices"):
            continue
        if k not in new_root:
            new_root[k] = v
    return _reindex_nested(doc_id, new_root)
