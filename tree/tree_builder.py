import os
from typing import Any, Dict, List, Optional, Tuple

from .utils_2 import load_json, dump_json


DEFAULT_OUT_BASENAME = "doctree.mm.json"
RENDER_LISTS_AS_TEXT_ENV = "MM_BUILDER_LIST_AS_TEXT"


def _infer_doc_id(in_file: str, root_obj: Any) -> str:
    if isinstance(root_obj, dict) and isinstance(root_obj.get("doc_id"), str):
        return root_obj["doc_id"]
    doc_dir = os.path.dirname(in_file)
    return os.path.basename(doc_dir) or "document"


def _first_caption(caps: Any) -> str:
    if isinstance(caps, list) and caps:
        try:
            c = caps[0]
            return c if isinstance(c, str) else str(c)
        except Exception:
            return ""
    return ""


def _norm_title(t: str) -> str:
    s = (t or "").strip().lower()
    return " ".join(s.split())


def _make_typed_id(prefix: str, node_idx: Any, fallback_idx: int) -> str:
    try:
        idx = int(node_idx)
    except Exception:
        idx = fallback_idx
    return f"{prefix}_{idx}"


def _text_title(item: dict) -> str:
    txt = item.get("text") or ""
    if not isinstance(txt, str):
        txt = str(txt)
    # Heading title: prefer first line trimmed
    line = txt.strip().split("\n", 1)[0].strip()
    return line


def _convert_item_to_leaf(item: dict, *, fallback_idx: int) -> Optional[Dict[str, Any]]:
    t = item.get("type")
    node_idx = item.get("node_idx", fallback_idx)
    page_idx = item.get("page_idx")
    read_order = item.get("read_order_idx", node_idx)
    bbox = item.get("outline") or item.get("bbox")

    if t == "text":
        lvl = item.get("node_level", -1)
        if isinstance(lvl, int) and lvl >= 1:
            # handled as section elsewhere
            return None
        return {
            "type": "text",
            "node_id": _make_typed_id("t", node_idx, fallback_idx),
            "page_idx": page_idx,
            "read_order_idx": read_order,
            "role": "paragraph",
            "text": item.get("text", ""),
            **({"bbox": bbox} if isinstance(bbox, (list, tuple, list)) else {}),
        }
    if t == "image":
        nid = _make_typed_id("img", node_idx, fallback_idx)
        node: Dict[str, Any] = {
            "type": "image",
            "node_id": nid,
            "page_idx": page_idx,
            "read_order_idx": read_order,
            "image_path": item.get("img_path"),
            **({"bbox": bbox} if isinstance(bbox, (list, tuple, list)) else {}),
        }
        caps = item.get("image_caption") if isinstance(item.get("image_caption"), list) else []
        if caps:
            node["children"] = [{
                "type": "text",
                "node_id": _make_typed_id("cap", node_idx, fallback_idx),
                "role": "caption",
                "text": caps[0] if isinstance(caps[0], str) else str(caps[0]),
                "ref": nid,
                "page_idx": page_idx,
                "read_order_idx": read_order,
            }]
        desc = item.get("description") or item.get("text")
        if isinstance(desc, str) and desc:
            node["description"] = desc
        return node
    if t == "table":
        nid = _make_typed_id("tab", node_idx, fallback_idx)
        node = {
            "type": "table",
            "node_id": nid,
            "page_idx": page_idx,
            "read_order_idx": read_order,
            "data": item.get("table_body") or item.get("table_text") or "",
            **({"bbox": bbox} if isinstance(bbox, (list, tuple, list)) else {}),
        }
        caps = item.get("table_caption") if isinstance(item.get("table_caption"), list) else []
        if caps:
            node.setdefault("children", []).append({
                "type": "text",
                "node_id": _make_typed_id("cap", node_idx, fallback_idx),
                "role": "caption",
                "text": caps[0] if isinstance(caps[0], str) else str(caps[0]),
                "ref": nid,
                "page_idx": page_idx,
                "read_order_idx": read_order,
            })
        return node
    if t == "equation":
        return {
            "type": "equation",
            "node_id": _make_typed_id("eq", node_idx, fallback_idx),
            "page_idx": page_idx,
            "read_order_idx": read_order,
            "text": item.get("text", ""),
            **({"bbox": bbox} if isinstance(bbox, (list, tuple, list)) else {}),
        }
    if t == "list":
        items = item.get("list_items") if isinstance(item.get("list_items"), list) else []
        sub_type = item.get("sub_type") if isinstance(item.get("sub_type"), str) else "text"
        as_text = os.environ.get(RENDER_LISTS_AS_TEXT_ENV, "0") == "1"
        if as_text:
            text = "\n".join(f"- {str(x)}" for x in items)
            return {
                "type": "text",
                "node_id": _make_typed_id("t", node_idx, fallback_idx),
                "page_idx": page_idx,
                "read_order_idx": read_order,
                "role": "paragraph",
                "text": text,
                **({"bbox": bbox} if isinstance(bbox, (list, tuple, list)) else {}),
            }
        else:
            return {
                "type": "list",
                "node_id": _make_typed_id("lst", node_idx, fallback_idx),
                "page_idx": page_idx,
                "read_order_idx": read_order,
                "items": [{"text": str(x)} for x in items],
                "sub_type": sub_type,
                **({"bbox": bbox} if isinstance(bbox, (list, tuple, list)) else {}),
            }
    # ignore structural-only types not useful for QA (header/footer/page_number/etc.)
    return None


def build_mm_doctree(
    in_file: str,
    *,
    out_path: Optional[str] = None,
) -> str:
    """
    Build a hierarchical multimodal DocTree from an enhanced content root or raw content_list.
    - Input may be an enhanced root {type: document, children: [...]} or a raw list (content_list).
    - Hierarchy is formed by text items with node_level>=1; other items attach to the nearest preceding heading.
    - Output schema matches the requested minimal Node design.
    """
    obj = load_json(in_file)
    # Enhanced root or bare content_list
    if isinstance(obj, dict) and obj.get("type") == "document" and isinstance(obj.get("children"), list):
        items = obj["children"]
        doc_id = _infer_doc_id(in_file, obj)
        mode = obj.get("mode")
        columns = obj.get("columns")
    elif isinstance(obj, list):
        items = obj
        doc_id = _infer_doc_id(in_file, {})
        mode = None
        columns = None
    elif isinstance(obj, dict) and isinstance(obj.get("content_list"), list):
        items = obj["content_list"]
        doc_id = _infer_doc_id(in_file, obj)
        mode = obj.get("mode")
        columns = obj.get("columns")
    else:
        raise ValueError("Unsupported input: expected enhanced root or content_list array")

    # Collect unique pages for 'views'
    pages = sorted({it.get("page_idx") for it in items if isinstance(it.get("page_idx"), int)})

    # Extract TOC (if present as a node in items)
    toc_entries: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict) and it.get("type") == "toc" and isinstance(it.get("headings"), list):
            for h in it["headings"]:
                if not isinstance(h, dict):
                    continue
                toc_entries.append({
                    "title": h.get("title") or h.get("text") or "",
                    "level": h.get("level") or h.get("node_level") or h.get("depth") or None,
                    "page": h.get("page"),
                })
            break

    # Root document
    root: Dict[str, Any] = {
        "type": "document",
        "doc_id": doc_id,
        "meta": {
            "pages": len(pages),
            **({"mode": mode} if mode is not None else {}),
            **({"columns": columns} if columns is not None else {}),
        },
        "toc": toc_entries,
        "views": [{"type": "page", "node_id": f"P{p}", "page_idx": p} for p in pages],
        "children": [],
        "index": {"page_to_nodes": {}, "type_index": {}, "heading_index": {}},
    }

    # Build section hierarchy using a stack
    stack: List[Tuple[int, Dict[str, Any]]] = []

    def attach(parent: Dict[str, Any], child: Dict[str, Any]) -> None:
        parent.setdefault("children", []).append(child)

    def add_index(node: Dict[str, Any]) -> None:
        nid = node.get("node_id")
        typ = node.get("type")
        p = node.get("page_idx")
        if isinstance(p, int) and isinstance(nid, str):
            root["index"]["page_to_nodes"].setdefault(str(p), []).append(nid)
        if isinstance(typ, str) and isinstance(nid, str):
            root["index"]["type_index"].setdefault(typ, []).append(nid)

    # First pass: create sections and attach leaves
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        t = it.get("type")
        lvl = it.get("node_level", -1)
        # Skip toc nodes (already extracted)
        if t == "toc":
            continue

        # Slides: treat each slide container as a level-1 section
        if t == "slide":
            # title from first heading child if available
            title = None
            for ch in it.get("children", []) if isinstance(it.get("children"), list) else []:
                if isinstance(ch, dict) and ch.get("type") in ("heading", "text"):
                    ch_lvl = ch.get("node_level", 0)
                    ch_text = ch.get("text")
                    if isinstance(ch_text, str) and ch_text.strip():
                        title = ch_text.strip()
                        break
            if not title:
                title = f"Slide {it.get('slide_number') or ''}".strip()
            sec = {
                "type": "section",
                "node_id": _make_typed_id("sec", it.get("node_idx", i), i),
                "level": 1,
                "title": title,
                "title_norm": _norm_title(title),
                "page_idx": it.get("page_idx"),
                "read_order_idx": it.get("read_order_idx", it.get("node_idx", i)),
            }
            # Update heading index
            root["index"]["heading_index"].setdefault("1", {})[sec["title_norm"]] = sec["node_id"]
            # Attach as top-level (reset stack for slides)
            stack = []
            attach(root, sec)
            add_index(sec)
            stack.append((1, sec))
            # Attach slide children as leaves (skip heading nodes)
            for ch in it.get("children", []) if isinstance(it.get("children"), list) else []:
                if not isinstance(ch, dict):
                    continue
                if ch.get("type") in ("heading", "slide"):
                    continue
                leaf = _convert_item_to_leaf(ch, fallback_idx=ch.get("node_idx", i))
                if leaf is None:
                    continue
                attach(sec, leaf)
                add_index(leaf)
                for sub in leaf.get("children", []) if isinstance(leaf.get("children"), list) else []:
                    if isinstance(sub, dict):
                        add_index(sub)
            continue

        # Headings produced as explicit nodes
        if t == "heading" and isinstance(lvl, int) and lvl >= 1:
            title = _text_title(it)
            sec = {
                "type": "section",
                "node_id": _make_typed_id("sec", it.get("node_idx", i), i),
                "level": int(lvl),
                "title": title,
                "title_norm": _norm_title(title),
                "page_idx": it.get("page_idx"),
                "read_order_idx": it.get("read_order_idx", it.get("node_idx", i)),
            }
            lvl_key = str(int(lvl))
            root["index"]["heading_index"].setdefault(lvl_key, {})[sec["title_norm"]] = sec["node_id"]
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            parent = stack[-1][1] if stack else root
            attach(parent, sec)
            add_index(sec)
            stack.append((int(lvl), sec))
            continue

        if t == "text" and isinstance(lvl, int) and lvl >= 1:
            # Section node
            title = _text_title(it)
            sec = {
                "type": "section",
                "node_id": _make_typed_id("sec", it.get("node_idx", i), i),
                "level": int(lvl),
                "title": title,
                "title_norm": _norm_title(title),
                "page_idx": it.get("page_idx"),
                "read_order_idx": it.get("read_order_idx", it.get("node_idx", i)),
            }
            # Update heading_index
            lvl_key = str(int(lvl))
            root["index"]["heading_index"].setdefault(lvl_key, {})[sec["title_norm"]] = sec["node_id"]
            # Push to hierarchy
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            parent = stack[-1][1] if stack else root
            attach(parent, sec)
            add_index(sec)
            stack.append((int(lvl), sec))
            continue

        # Leaf node
        leaf = _convert_item_to_leaf(it, fallback_idx=i)
        if leaf is None:
            continue
        parent = stack[-1][1] if stack else root
        attach(parent, leaf)
        add_index(leaf)
        # also index captions if any
        for ch in leaf.get("children", []) if isinstance(leaf.get("children"), list) else []:
            if isinstance(ch, dict):
                add_index(ch)

    # Best-effort: link toc targets to sections using normalized title and level
    if root.get("toc"):
        # Build map from (level, norm_title) -> node_id
        rev: Dict[Tuple[Optional[int], str], str] = {}
        for lvl_str, m in root["index"].get("heading_index", {}).items():
            try:
                lvl_int = int(lvl_str)
            except Exception:
                lvl_int = None
            for k, v in m.items():
                rev[(lvl_int, k)] = v
        for e in root["toc"]:
            if not isinstance(e, dict):
                continue
            title = e.get("title") or ""
            norm = _norm_title(title)
            lvl = e.get("level")
            e["target"] = rev.get((int(lvl) if isinstance(lvl, int) else None, norm))

    if out_path is None:
        out_path = os.path.join(os.path.dirname(in_file), DEFAULT_OUT_BASENAME)
    dump_json(root, out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    def _choose_input_path(doc_dir: str) -> Optional[str]:
        # Prefer enhancer output, then adapter v2 output. Keep it simple and explicit.
        cand = [
            os.path.join(doc_dir, "content_list.enhanced.json"),
            os.path.join(doc_dir, "content_list.adapted.v2.json"),
        ]
        for p in cand:
            if os.path.exists(p):
                return p
        return None

    def build_directory(in_dir: str, *, out_name: str = DEFAULT_OUT_BASENAME) -> tuple[int, int]:
        # Find all document folders that contain either enhanced or adapted v2 content lists
        doc_dirs = set()
        for root, _, files in os.walk(in_dir):
            names = set(files)
            if ("content_list.enhanced.json" in names) or ("content_list.adapted.v2.json" in names):
                doc_dirs.add(root)
        ok = 0
        err = 0
        for d in sorted(doc_dirs):
            src = _choose_input_path(d)
            if not src:
                continue
            try:
                out_path = os.path.join(d, out_name)
                build_mm_doctree(src, out_path=out_path)
                ok += 1
            except Exception:
                err += 1
        return ok, err

    parser = argparse.ArgumentParser(description="Build a hierarchical multimodal DocTree from enhanced content or adapted v2 content")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-file", dest="in_file", type=str, help="Path to content_list.enhanced.json or content_list.adapted.v2.json")
    group.add_argument("--in-dir", dest="in_dir", type=str, help="Root directory that contains document folders")
    parser.add_argument("--out", dest="out", type=str, default=None, help="Output file (with --in-file). Default: doctree.mm.json beside input")
    parser.add_argument("--suffix", dest="suffix", type=str, default=None, help="With --in-dir, write doctree{suffix} in each doc folder (default: .mm.json)")
    args = parser.parse_args()

    if args.in_file:
        out = build_mm_doctree(args.in_file, out_path=args.out)
        print(out)
    else:
        out_name = DEFAULT_OUT_BASENAME if not args.suffix else f"doctree{args.suffix}"
        ok, err = build_directory(args.in_dir, out_name=out_name)
        print(f"Build completed: {ok} success, {err} errors")
