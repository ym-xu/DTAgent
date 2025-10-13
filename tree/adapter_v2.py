import glob
import os
from typing import Any, Dict, List, Optional, Tuple

from .utils_2 import (
    load_json,
    dump_json,
    html_table_to_markdown,
    sanitize_item_fields,
)


DEFAULT_SUFFIX = ".adapted.v2.json"


def _make_out_path_standard(src_path: str, suffix: str) -> str:
    """Write beside the source as 'content_list{suffix}' (no doc_id prefix)."""
    doc_dir = os.path.dirname(src_path)
    out_name = f"content_list{suffix}"
    return os.path.join(doc_dir, out_name)


def _is_new_mineru_format(items: List[dict]) -> bool:
    if not items:
        return False
    has_bbox = any(isinstance(it, dict) and isinstance(it.get("bbox"), (list, tuple)) for it in items)
    has_new_types = any(str(it.get("type") or "") in {"list", "header", "aside_text", "page_number", "page_footnote"} for it in items if isinstance(it, dict))
    return has_bbox or has_new_types


def _ensure_outline_from_bbox(it: dict) -> None:
    bbox = it.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        it.setdefault("outline", [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])


def _is_empty_text_node(it: Dict[str, Any]) -> bool:
    if it.get("type") != "text":
        return False
    txt = str(it.get("text") or "").strip()
    return not bool(txt)

def _build_blocks_index_with_discarded(layout_obj: Any) -> Dict[int, List[Dict[str, Any]]]:
    """
    Build page_idx -> blocks list from layout['pdf_info'], concatenating
    para_blocks + discarded_blocks in that order.

    Rationale: content_list per-page sequence aligns to para_blocks; items missing
    there are placed at the end within discarded_blocks. Keeping this order preserves
    1:1 alignment by position.
    """
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    if not isinstance(layout_obj, dict):
        return by_page
    pages = layout_obj.get("pdf_info")
    if not isinstance(pages, list):
        return by_page
    for pg in pages:
        try:
            pidx = pg.get("page_idx")
            if not isinstance(pidx, int):
                continue
            para = [b for b in (pg.get("para_blocks") or []) if isinstance(b, dict)]
            disc = [b for b in (pg.get("discarded_blocks") or []) if isinstance(b, dict)]
            by_page[int(pidx)] = para + disc
        except Exception:
            continue
    # 1-based -> 0-based if needed
    if by_page and (0 not in by_page) and (1 in by_page):
        try:
            by_page = {int(k) - 1: v for k, v in by_page.items()}
        except Exception:
            pass
    return by_page

def _try_get_block_bbox(block: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bbox = block.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            return (x0, y0, x1, y1)
        except Exception:
            return None
    # Fallbacks: first line or span bbox
    lines = block.get("lines")
    if isinstance(lines, list) and lines:
        ln = lines[0]
        bb = ln.get("bbox") if isinstance(ln, dict) else None
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            try:
                return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
            except Exception:
                pass
        spans = ln.get("spans") if isinstance(ln, dict) else None
        if isinstance(spans, list) and spans:
            sp = spans[0]
            bb2 = sp.get("bbox") if isinstance(sp, dict) else None
            if isinstance(bb2, (list, tuple)) and len(bb2) >= 4:
                try:
                    return (float(bb2[0]), float(bb2[1]), float(bb2[2]), float(bb2[3]))
                except Exception:
                    pass
    return None


def adapt_content_list_v2(
    content_list: List[dict],
    *,
    prefer_new_format: Optional[bool] = None,
    layout_obj: Optional[Any] = None,
) -> List[dict]:
    """
    Format-agnostic adapter for MinerU content_list.

    Goals (v2):
    - Always add node_idx/read_order_idx and depth=0
    - Rename text_level -> node_level when present; default node_level=-1
    - Do not depend on layout.json; respect incoming bbox
    - For image/table: set outline=bbox if present; ensure caption/footnote/description fields exist
    - For table: optionally add table_text (markdown) from table_body
    - For list: first-class support; keep sub_type/list_items/bbox
    - Keep other types as-is (header/aside_text/page_number/page_footnote/code/equation)
    - Light sanitize per-item
    """

    is_new = _is_new_mineru_format(content_list) if prefer_new_format is None else bool(prefer_new_format)

    # Optional alignment with layout to override incorrect bbox from content_list
    by_page_blocks: Dict[int, List[Dict[str, Any]]] = _build_blocks_index_with_discarded(layout_obj) if layout_obj is not None else {}

    adapted: List[dict] = []
    page_pos: Dict[int, int] = {}
    for idx, src in enumerate(content_list):
        it = dict(src) if isinstance(src, dict) else {}

        # indices
        it["node_idx"] = idx
        it.setdefault("read_order_idx", idx)
        it.setdefault("depth", 0)

        # node level normalization
        if "text_level" in it and "node_level" not in it:
            it["node_level"] = it.pop("text_level")
        it.setdefault("node_level", -1)

        typ = str(it.get("type") or "").lower()

        # Override bbox using layout alignment if available
        pg = it.get("page_idx") if isinstance(it.get("page_idx"), int) else None
        if isinstance(pg, int) and by_page_blocks:
            pos = page_pos.get(pg, 0)
            blocks = by_page_blocks.get(pg) or []
            if 0 <= pos < len(blocks):
                bb = _try_get_block_bbox(blocks[pos])
                if bb is not None:
                    it["bbox"] = [bb[0], bb[1], bb[2], bb[3]]

        # Respect incoming bbox; attach outline from bbox for media/table for downstream compatibility
        if typ in ("image", "table"):
            _ensure_outline_from_bbox(it)

        if typ == "image":
            it.setdefault("text", "")
            it.setdefault("description", "")
            it.setdefault("image_caption", [])
            it.setdefault("image_footnote", [])
            it.setdefault("image_subcaption", "")

        elif typ == "table":
            tb = it.get("table_body")
            if isinstance(tb, str):
                try:
                    md = html_table_to_markdown(tb)
                    if md:
                        it.setdefault("table_text", md)
                except Exception:
                    pass
            it.setdefault("description", "")
            it.setdefault("table_caption", [])
            it.setdefault("table_footnote", [])
            it.setdefault("table_subcaption", "")

        elif typ == "list":
            # Keep as-is; sanitize will ensure list_items is a list of strings
            it.setdefault("sub_type", "text")
            if not isinstance(it.get("list_items"), list):
                it["list_items"] = []

        # Drop empty text nodes (defensive), mainly relevant to old outputs
        if _is_empty_text_node(it):
            continue

        # Sanitize fields based on type
        try:
            sanitize_item_fields(it)
        except Exception:
            pass

        adapted.append(it)
        if isinstance(pg, int):
            page_pos[pg] = page_pos.get(pg, 0) + 1

    # Reindex to be safe
    for i, it in enumerate(adapted):
        it["node_idx"] = i
        it["read_order_idx"] = i

    return adapted


def adapt_single_file_v2(
    dom_file: str,
    *,
    write_in_place: bool = True,
    output_path: Optional[str] = None,
    suffix: str = DEFAULT_SUFFIX,
    prefer_new_format: Optional[bool] = None,
) -> str:
    raw = load_json(dom_file)
    # Keep it simple: accept either a bare list or a dict with 'content_list'
    if isinstance(raw, list):
        content_list = raw
    elif isinstance(raw, dict) and isinstance(raw.get("content_list"), list):
        content_list = raw["content_list"]
    else:
        raise ValueError("Unsupported content file format: expect list or {'content_list': [...]} root")

    # Load sibling layout.json if present
    layout_obj = None
    cand_layout = os.path.join(os.path.dirname(dom_file), "layout.json")
    if os.path.exists(cand_layout):
        try:
            layout_obj = load_json(cand_layout)
        except Exception:
            layout_obj = None

    adapted = adapt_content_list_v2(
        content_list,
        prefer_new_format=prefer_new_format,
        layout_obj=layout_obj,
    )
    if write_in_place:
        out_path = dom_file
    else:
        out_path = output_path or _make_out_path_standard(dom_file, suffix)
    dump_json(adapted, out_path)
    return out_path


def process_directory_v2(
    dom_dir: str,
    *,
    in_place: bool = False,
    suffix: str = DEFAULT_SUFFIX,
    prefer_new_format: Optional[bool] = None,
) -> int:
    # Simple, explicit: in each doc folder there's exactly one '*content_list.json'
    dom_files = glob.glob(os.path.join(dom_dir, "**", "*content_list.json"), recursive=True)

    ok = 0
    for dom in dom_files:
        try:
            adapt_single_file_v2(dom, write_in_place=in_place, suffix=suffix, prefer_new_format=prefer_new_format)
            ok += 1
        except Exception:
            continue
    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adapt MinerU content_list (v2) without layout alignment; expects *content_list.json")
    parser.add_argument("--in-dir", dest="in_dir", type=str, help="Root directory containing MinerU outputs")
    parser.add_argument("--in-file", dest="in_file", type=str, default=None, help="Specific *content_list.json file")
    parser.add_argument("--out", dest="out", type=str, default=None, help="Output file when using --in-file; overrides --suffix")
    parser.add_argument("--in-place", dest="in_place", action="store_true", help="Write back to the input content list file")
    parser.add_argument("--suffix", dest="suffix", type=str, default=DEFAULT_SUFFIX, help=f"When not --in-place, write alongside originals using this suffix (default: {DEFAULT_SUFFIX})")
    parser.add_argument("--prefer-new-format", dest="prefer_new_format", action="store_true", help="Force treat input as new MinerU format (bbox-rich)")

    args = parser.parse_args()

    if args.in_file:
        out = adapt_single_file_v2(
            args.in_file,
            write_in_place=args.in_place,
            output_path=args.out,
            suffix=args.suffix,
            prefer_new_format=True if args.prefer_new_format else None,
        )
        print(out)
    else:
        if not args.in_dir:
            parser.error("Provide --in-dir or --in-file")
        ok = process_directory_v2(
            args.in_dir,
            in_place=args.in_place,
            suffix=args.suffix,
            prefer_new_format=True if args.prefer_new_format else None,
        )
        print(f"Adapted {ok} files")
