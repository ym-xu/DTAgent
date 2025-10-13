import os
from typing import Any, Dict, List, Optional, Tuple

from ..utils.json_io import load_json, dump_json


DEFAULT_OUT_BASENAME = "doctree.mm.json"
RENDER_LISTS_AS_TEXT_ENV = "MM_BUILDER_LIST_AS_TEXT"


def _infer_doc_id(in_file: str, root_obj: Any) -> str:
    if isinstance(root_obj, dict) and isinstance(root_obj.get("doc_id"), str):
        return root_obj["doc_id"]
    doc_dir = os.path.dirname(in_file)
    return os.path.basename(doc_dir) or "document"


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
    line = txt.strip().split("\n", 1)[0].strip()
    return line


def _convert_item_to_leaf(item: dict, *, fallback_idx: int) -> Optional[Dict[str, Any]]:
    t = item.get("type")
    node_idx = item.get("node_idx", fallback_idx)
    page_idx = item.get("page_idx")
    read_order = item.get("read_order_idx", node_idx)
    bbox = item.get("outline") or item.get("bbox")
    logical_page = item.get("logical_page")

    if t == "text":
        lvl = item.get("node_level", -1)
        if isinstance(lvl, int) and lvl >= 1:
            return None
        return {
            "type": "text",
            "node_id": _make_typed_id("t", node_idx, fallback_idx),
            "page_idx": page_idx,
            "read_order_idx": read_order,
            **({"logical_page": logical_page} if logical_page is not None else {}),
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
            **({"logical_page": logical_page} if logical_page is not None else {}),
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
                **({"logical_page": logical_page} if logical_page is not None else {}),
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
            **({"logical_page": logical_page} if logical_page is not None else {}),
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
                **({"logical_page": logical_page} if logical_page is not None else {}),
            })
        return node
    if t == "equation":
        return {
            "type": "equation",
            "node_id": _make_typed_id("eq", node_idx, fallback_idx),
            "page_idx": page_idx,
            "read_order_idx": read_order,
            **({"logical_page": logical_page} if logical_page is not None else {}),
            "text": item.get("text", ""),
            **({"bbox": bbox} if isinstance(bbox, (list, tuple, list)) else {}),
        }
    if t == "list":
        items = item.get("list_items") if isinstance(item.get("list_items"), list) else []
        sub_type = item.get("sub_type", "text")
        if items and all(isinstance(x, str) for x in items):
            text = "\n".join(str(x) for x in items)
            return {
                "type": "text",
                "node_id": _make_typed_id("t", node_idx, fallback_idx),
                "page_idx": page_idx,
                "read_order_idx": read_order,
                **({"logical_page": logical_page} if logical_page is not None else {}),
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
                **({"logical_page": logical_page} if logical_page is not None else {}),
                "items": [{"text": str(x)} for x in items],
                "sub_type": sub_type,
                **({"bbox": bbox} if isinstance(bbox, (list, tuple, list)) else {}),
            }
    return None


def build_mm_doctree(
    in_file: str,
    *,
    out_path: Optional[str] = None,
) -> str:
    obj = load_json(in_file)
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

    pages = sorted({it.get("page_idx") for it in items if isinstance(it.get("page_idx"), int)})

    root: Dict[str, Any] = {
        "type": "document",
        "doc_id": doc_id,
        "meta": {
            "pages": len(pages),
            **({"mode": mode} if mode is not None else {}),
            **({"columns": columns} if columns is not None else {}),
        },
        "toc": [],
        "views": [{"type": "page", "node_id": f"P{p}", "page_idx": p} for p in pages],
        "children": [],
        "index": {"page_to_nodes": {}, "type_index": {}, "heading_index": {}},
    }

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

    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        t = it.get("type")
        lvl = it.get("node_level", -1)
        if t == "toc":
            continue
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
                **({"logical_page": it.get("logical_page")} if it.get("logical_page") is not None else {}),
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
            title = _text_title(it)
            sec = {
                "type": "section",
                "node_id": _make_typed_id("sec", it.get("node_idx", i), i),
                "level": int(lvl),
                "title": title,
                "title_norm": _norm_title(title),
                "page_idx": it.get("page_idx"),
                "read_order_idx": it.get("read_order_idx", it.get("node_idx", i)),
                **({"logical_page": it.get("logical_page")} if it.get("logical_page") is not None else {}),
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

        leaf = _convert_item_to_leaf(it, fallback_idx=i)
        if leaf is None:
            continue
        parent = stack[-1][1] if stack else root
        attach(parent, leaf)
        add_index(leaf)
        for ch in leaf.get("children", []) if isinstance(leaf.get("children"), list) else []:
            if isinstance(ch, dict):
                add_index(ch)

    if out_path is None:
        out_path = os.path.join(os.path.dirname(in_file), DEFAULT_OUT_BASENAME)
    dump_json(root, out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a hierarchical multimodal DocTree from enhanced content or adapted v2 content")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-file", dest="in_file", type=str, help="Path to content list JSON")
    group.add_argument("--in-dir", dest="in_dir", type=str, help="Root directory that contains document folders")
    parser.add_argument("--out", dest="out", type=str, default=None, help="Output file (with --in-file). Default: doctree.mm.json beside input")
    args = parser.parse_args()

    if args.in_file:
        out = build_mm_doctree(args.in_file, out_path=args.out)
        print(out)
    else:
        # Directory mode: scan for document folders and build doctrees
        in_dir = args.in_dir
        succ = 0
        err = 0
        def _pick_input_file(d: str) -> Optional[str]:
            # Prefer enhanced, then adapted.v2, then adapted, then raw content_list.json
            cands = [
                os.path.join(d, "content_list.enhanced.json"),
                os.path.join(d, "content_list.adapted.v2.json"),
                os.path.join(d, "content_list.adapted.json"),
                os.path.join(d, "content_list.json"),
            ]
            for p in cands:
                if os.path.exists(p):
                    return p
            return None
        doc_dirs: set[str] = set()
        for root_dir, dirs, files in os.walk(in_dir, followlinks=True):
            if any(name in files for name in ("content_list.enhanced.json","content_list.adapted.v2.json","content_list.adapted.json","content_list.json")):
                doc_dirs.add(root_dir)
        for d in sorted(doc_dirs):
            src = _pick_input_file(d)
            if not src:
                continue
            try:
                out_path = os.path.join(d, DEFAULT_OUT_BASENAME)
                build_mm_doctree(src, out_path=out_path)
                print(out_path)
                succ += 1
            except Exception:
                err += 1
                continue
        print(f"Built doctrees: {succ} success, {err} errors")
