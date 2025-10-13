from typing import Any, Dict, List, Optional
from .utils import ensure_page_image


def build_page_tree(flat_root: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a simple Page Tree: document -> pages -> nodes.

    Input: flat_root with keys: doc_id, source_path?, children (list of nodes with page_idx)
    Output schema:
      {
        type: "document",
        doc_id: str,
        source_path?: str,
        pages: [
          {
            type: "page",
            page_idx: int,
            children: [ original nodes for this page in reading order ]
          },
          ...
        ]
      }
    """
    doc_id = flat_root.get("doc_id") or "document"
    source_path = flat_root.get("source_path")
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))

    # Group nodes by page_idx preserving order
    pages_map: Dict[int, List[Dict[str, Any]]] = {}
    for ch in children:
        p = ch.get("page_idx")
        if isinstance(p, int):
            pages_map.setdefault(p, []).append(ch)

    pages_list: List[Dict[str, Any]] = []
    for p in sorted(pages_map.keys()):
        pages_list.append({
            "type": "page",
            "page_idx": p,
            "children": pages_map[p],
        })

    out: Dict[str, Any] = {"type": "document", "doc_id": doc_id, "pages": pages_list}
    if source_path:
        out["source_path"] = source_path
    return out


def build_chapter_tree(flat_root: Dict[str, Any], *, max_level: int = 6) -> Dict[str, Any]:
    """
    Build a Chapter Tree based on node_level headings in the flat doctree.

    Strategy:
    - Iterate flat children in order.
    - When encountering a text node with node_level in [1..max_level], treat as a section heading.
    - Maintain a stack of sections by level.
    - Non-heading nodes attach to the current top section.

    Output schema:
      {
        type: "document",
        doc_id: str,
        source_path?: str,
        sections: [ section, ... ]
      }

    Section schema:
      {
        type: "section",
        level: int,
        title: str,
        title_node_idx: int,
        page_idx: int,
        children: [ node or section ]
      }

    Notes:
    - Original nodes are embedded under sections as-is (no custom keys added to them).
    - If the document starts with body content before any heading, a synthetic level-0 root section is used to collect them.
    """
    doc_id = flat_root.get("doc_id") or "document"
    source_path = flat_root.get("source_path")
    nodes: List[Dict[str, Any]] = list(flat_root.get("children", []))

    def make_section(level: int, title: str, title_node_idx: int, page_idx: Optional[int]) -> Dict[str, Any]:
        sec: Dict[str, Any] = {
            "type": "section",
            "level": level,
            "title": title,
            "title_node_idx": title_node_idx,
            "page_idx": page_idx if isinstance(page_idx, int) else None,
            "children": [],
        }
        return sec

    root_sections: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []  # stack of current sections

    # Ensure there is a top-level container to catch preface content
    preface = make_section(0, "Preface", -1, None)
    root_sections.append(preface)
    stack.append(preface)

    for ch in nodes:
        lvl = ch.get("node_level")
        typ = ch.get("type")
        if typ == "text" and isinstance(lvl, int) and 1 <= lvl <= max_level:
            # Heading encountered: adjust stack
            while stack and stack[-1]["level"] >= lvl:
                stack.pop()
            title_text = ch.get("text", "").strip()
            sec = make_section(lvl, title_text, ch.get("node_idx", -1), ch.get("page_idx"))
            if stack:
                stack[-1]["children"].append(sec)
            else:
                root_sections.append(sec)
            stack.append(sec)
        else:
            # Body content: attach to current section
            stack[-1]["children"].append(ch)

    out: Dict[str, Any] = {"type": "document", "doc_id": doc_id, "sections": root_sections}
    if source_path:
        out["source_path"] = source_path
    return out


def build_slides_tree(flat_root: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a Slides Tree: document -> slides (one image node per page) -> page elements.

    For each page, create an image node as the slide container with fields:
      - type: "image"
      - page_idx: int
      - img_path: path resolved via ensure_page_image (may be None)
      - description: "" (placeholder for future enrichment)
      - children: original nodes on that page in reading order
    """
    doc_id = flat_root.get("doc_id") or "document"
    source_path = flat_root.get("source_path")
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))

    # Group nodes by page_idx preserving order
    pages_map: Dict[int, List[Dict[str, Any]]] = {}
    for ch in children:
        p = ch.get("page_idx")
        if isinstance(p, int):
            pages_map.setdefault(p, []).append(ch)

    slides: List[Dict[str, Any]] = []
    for p in sorted(pages_map.keys()):
        img_path = None
        if isinstance(source_path, str) and source_path:
            try:
                img_path = ensure_page_image(source_path, p)
            except Exception:
                img_path = None
        slide: Dict[str, Any] = {
            "type": "image",
            "page_idx": p,
            "img_path": img_path,
            "description": "",
            "children": pages_map[p],
        }
        slides.append(slide)

    out: Dict[str, Any] = {"type": "document", "doc_id": doc_id, "slides": slides}
    if source_path:
        out["source_path"] = source_path
    return out


def build_tree_by_mode(flat_root: Dict[str, Any], mode: str = "doc") -> Dict[str, Any]:
    """
    Dispatcher to build different tree views from a flat doctree.
    - mode == "slides": build a slides tree (one image node per page)
    - otherwise: default to page tree
    """
    if mode == "slides":
        return build_slides_tree(flat_root)
    return build_page_tree(flat_root)
