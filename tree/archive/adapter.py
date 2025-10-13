import glob
import os
from typing import Any, Dict, List, Optional, Tuple

from .utils import (
    load_json,
    dump_json,
    html_table_to_markdown,
    sanitize_item_fields,
    ensure_all_page_images_for_pages,
)


DEFAULT_SUFFIX = ".adapted.json"


def _make_out_path(src_path: str, suffix: str) -> str:
    if src_path.endswith(".json"):
        return src_path[:-5] + suffix
    return src_path + suffix


# =========================
# Layout helpers (strict alignment)
# =========================

def _build_para_blocks_index(layout_obj: Any) -> Dict[int, List[Dict[str, Any]]]:
    """Build page_idx -> para_blocks list from layout['pdf_info'] with minimal filtering.

    Rules:
    - Keep blocks in original order to preserve 1:1 alignment with content_list
    - Do NOT drop blocks with lines_deleted or empty lines; downstream will fallback to block bbox
    - Convert 1-based page_idx to 0-based when necessary
    """
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    src = layout_obj
    if not isinstance(src, dict):
        return by_page
    pages = src.get("pdf_info")
    if not isinstance(pages, list):
        return by_page

    for pg in pages:
        try:
            pidx = pg.get("page_idx")
            raw_blocks = pg.get("para_blocks")
            if not (isinstance(pidx, int) and isinstance(raw_blocks, list)):
                continue
            # Keep blocks as-is (only ensure dict type), preserve order
            filt: List[Dict[str, Any]] = [blk for blk in raw_blocks if isinstance(blk, dict)]
            by_page[int(pidx)] = filt
        except Exception:
            continue

    # 1-based -> 0-based shift if needed
    if by_page and (0 not in by_page) and (1 in by_page):
        try:
            by_page = {int(k) - 1: v for k, v in by_page.items()}
        except Exception:
            pass
    return by_page


def _extract_text_spans_from_block(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract textual spans from a para_block (keep text and inline_equation)."""
    out: List[Dict[str, Any]] = []
    lines = block.get("lines") if isinstance(block, dict) else None
    if not isinstance(lines, list):
        return out
    for ln in lines:
        spans = ln.get("spans") if isinstance(ln, dict) else None
        if not isinstance(spans, list):
            continue
        for sp in spans:
            if not isinstance(sp, dict):
                continue
            typ = sp.get("type")
            if typ not in ("text", "inline_equation"):
                continue
            bbox = sp.get("bbox")
            txt = sp.get("content") if isinstance(sp.get("content"), str) else sp.get("text")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and isinstance(txt, str) and txt.strip():
                out.append({
                    "text": txt,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                })
    return out


def _validate_page_alignment(content_list: List[dict], para_blocks_by_page: Dict[int, List[dict]]) -> None:
    """Ensure 1:1 alignment per page; raise if any page count mismatches."""
    counts: Dict[int, int] = {}
    for it in content_list:
        p = it.get("page_idx")
        if isinstance(p, int):
            counts[p] = counts.get(p, 0) + 1
    for p, cnt in counts.items():
        blocks = para_blocks_by_page.get(p)
        if not isinstance(blocks, list):
            raise ValueError(f"layout missing page {p} for alignment")
        if cnt != len(blocks):
            raise ValueError(f"alignment count mismatch on page {p}: content_list={cnt} vs para_blocks={len(blocks)}")


# =========================
# List detection (marker only)
# =========================

_BULLETS = set("•◦·●○▪▫-–—*")
_RE_ORDERED = [
    # 1. 1) (1)
    ("num-dot", lambda s: True if __import__('re').match(r"^\s*\d+\.\s+", s) else False, lambda s: __import__('re').match(r"^\s*(\d+)\.\s+", s).group(1)),
    ("num-paren", lambda s: True if __import__('re').match(r"^\s*\d+\)\s+", s) else False, lambda s: __import__('re').match(r"^\s*(\d+)\)\s+", s).group(1)),
    ("num-wrap", lambda s: True if __import__('re').match(r"^\s*\(\d+\)\s+", s) else False, lambda s: __import__('re').match(r"^\s*\((\d+)\)\s+", s).group(1)),
    # a. a) (a)
    ("alpha-dot", lambda s: True if __import__('re').match(r"^\s*[A-Za-z]\.\s+", s) else False, lambda s: __import__('re').match(r"^\s*([A-Za-z])\.\s+", s).group(1)),
    ("alpha-paren", lambda s: True if __import__('re').match(r"^\s*[A-Za-z]\)\s+", s) else False, lambda s: __import__('re').match(r"^\s*([A-Za-z])\)\s+", s).group(1)),
    ("alpha-wrap", lambda s: True if __import__('re').match(r"^\s*\([A-Za-z]\)\s+", s) else False, lambda s: __import__('re').match(r"^\s*\(([A-Za-z])\)\s+", s).group(1)),
]


def _detect_list_item(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    if s and s[0] in _BULLETS:
        return {"role": "list_item", "ordered": False, "marker": s[0]}
    for _, pred, grab in _RE_ORDERED:
        try:
            if pred(s):
                return {"role": "list_item", "ordered": True, "marker": grab(s)}
        except Exception:
            continue
    return None


# =========================
# Core adapter
# =========================

def _concat_item_text(it: Dict[str, Any]) -> str:
    if isinstance(it.get("spans"), list) and it.get("spans"):
        try:
            return "".join([str(s.get("text") or "") for s in it.get("spans")])
        except Exception:
            pass
    return str(it.get("text") or "")


def _normalize_comp(s: str) -> str:
    try:
        import unicodedata, re
        s = unicodedata.normalize("NFKC", s or "")
        s = s.lower()
        s = re.sub(r"\s+", "", s)
        return s
    except Exception:
        return s or ""


def _strip_title_prefix(prev_title: str, para_text: str) -> Optional[str]:
    if not isinstance(prev_title, str) or not isinstance(para_text, str):
        return None
    pt = prev_title.strip()
    if not pt:
        return None
    # Fast path: raw prefix
    if para_text.startswith(pt):
        return para_text[len(pt):].lstrip()
    # Robust path: compare with whitespace-insensitive prefix
    prev_norm = _normalize_comp(pt)
    para_norm = _normalize_comp(para_text)
    if not prev_norm or not para_norm:
        return None
    if para_norm.startswith(prev_norm):
        # Find cut index in original para_text by counting non-space chars up to len(prev_norm)
        target = len(prev_norm)
        seen = 0
        cut = 0
        for i, ch in enumerate(para_text):
            if ch.strip():
                seen += 1
            if seen >= target:
                cut = i + 1
                break
        return para_text[cut:].lstrip()
    return None


def _is_empty_text_node(it: Dict[str, Any]) -> bool:
    if it.get("type") != "text":
        return False
    txt = str(it.get("text") or "").strip()
    if txt:
        return False
    spans = it.get("spans")
    if not isinstance(spans, list) or not spans:
        return True
    for s in spans:
        if isinstance(s, dict) and str(s.get("text") or "").strip():
            return False
    return True


def _split_inline_bulleted_items(text: str) -> Optional[Tuple[str, List[str]]]:
    import re
    if not isinstance(text, str) or "\n" in text:
        return None
    s = text
    ms = list(re.finditer(r"(?:(?<=\s)|^)([•◦·●○▪▫\-–—])\s*", s))
    if len(ms) < 2:
        return None
    items: List[str] = []
    marker = ms[0].group(1)
    for i, m in enumerate(ms):
        start = m.end()
        end = ms[i + 1].start() if i + 1 < len(ms) else len(s)
        chunk = s[start:end].strip()
        if chunk:
            items.append(chunk)
    return (marker, items) if len(items) >= 2 else None


def _split_inline_ordered_items(text: str) -> Optional[List[str]]:
    import re
    if not isinstance(text, str) or "\n" in text:
        return None
    # Match sequences like: 1. aaa 2. bbb 3. ccc  OR  (1) aaa (2) bbb ... OR a) aaa b) bbb ...
    pattern = r"(?:(?<=\s)|^)((?:\d+|[A-Za-z])(?:\.|\)|\))\s+)"
    ms = list(re.finditer(pattern, text))
    if len(ms) < 2:
        return None
    items: List[str] = []
    for i, m in enumerate(ms):
        start = m.end()
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            items.append(chunk)
    return items if len(items) >= 2 else None


def _postprocess_page(items: List[Dict[str, Any]], *, enable_list_split: bool = False) -> List[Dict[str, Any]]:
    # 1) strip title prefix
    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if i > 0:
            prev = out[-1] if out else None
            if prev and prev.get("type") == "text" and (int(prev.get("node_level", -1)) > 0 or str(prev.get("block_type") or "").lower() == "title"):
                if it.get("type") == "text":
                    prev_title = _concat_item_text(prev)
                    new_text = _strip_title_prefix(prev_title, str(it.get("text") or ""))
                    if isinstance(new_text, str):
                        it = dict(it)
                        it["text"] = new_text
        out.append(it)

    # 2) remove empty text nodes
    out = [it for it in out if not _is_empty_text_node(it)]

    # 3) optional: split inline multi-item lists
    if enable_list_split:
        split_out: List[Dict[str, Any]] = []
        for it in out:
            if it.get("type") == "text":
                txt = str(it.get("text") or "")
                multi = _split_inline_bulleted_items(txt)
                ordered = _split_inline_ordered_items(txt) if not multi else None
                if multi:
                    marker, parts = multi
                    for p in parts:
                        new_it = dict(it)
                        new_it["text"] = p
                        new_it["list_meta"] = {"role": "list_item", "ordered": False, "marker": marker}
                        split_out.append(new_it)
                    continue
                if ordered:
                    for p in ordered:
                        new_it = dict(it)
                        new_it["text"] = p
                        new_it["list_meta"] = {"role": "list_item", "ordered": True, "marker": ""}
                        split_out.append(new_it)
                    continue
            split_out.append(it)
        out = split_out

    return out

def adapt_content_list(
    content_list: List[dict],
    layout_obj: Optional[Any] = None,
    ocr_image_func: Optional[callable] = None,
    describe_image_func: Optional[callable] = None,
) -> List[dict]:
    """Adapt MinerU content_list using strict 1:1 alignment with layout.pdf_info.

    - node_idx/read_order_idx, node_level rename (default -1), depth default 0
    - text/title → spans (text + inline_equation) + bbox_union + alignment_confidence=1.0
    - image/table → outline from aligned para_block.bbox
    - list detection → list_meta marker only
    - No fuzzy matching; if per-page counts mismatch, raise ValueError
    """
    para_blocks_by_page: Dict[int, List[Dict[str, Any]]] = {}
    if layout_obj is not None:
        para_blocks_by_page = _build_para_blocks_index(layout_obj)
        if para_blocks_by_page:
            _validate_page_alignment(content_list, para_blocks_by_page)

    adapted: List[dict] = []
    page_pos: Dict[int, int] = {}

    for idx, src in enumerate(content_list):
        it = dict(src)
        it["node_idx"] = idx
        it.setdefault("read_order_idx", idx)
        it.setdefault("depth", 0)
        if "text_level" in it and "node_level" not in it:
            it["node_level"] = it.pop("text_level")
        it.setdefault("node_level", -1)

        pg = it.get("page_idx") if isinstance(it.get("page_idx"), int) else None
        blk = None
        if pg is not None and para_blocks_by_page:
            pos = page_pos.get(pg, 0)
            blocks = para_blocks_by_page.get(pg) or []
            if 0 <= pos < len(blocks):
                blk = blocks[pos]

        typ = it.get("type")
        if typ == "text":
            if blk is not None:
                spans = _extract_text_spans_from_block(blk)
                if not spans:
                    bbox = blk.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        spans = [{"text": it.get("text", ""), "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]}]
                spans.sort(key=lambda s: (float(s["bbox"][1]), float(s["bbox"][0])))
                it["spans"] = spans
                xs0 = [float(s["bbox"][0]) for s in spans]
                ys0 = [float(s["bbox"][1]) for s in spans]
                xs1 = [float(s["bbox"][2]) for s in spans]
                ys1 = [float(s["bbox"][3]) for s in spans]
                it["bbox_union"] = [min(xs0), min(ys0), max(xs1), max(ys1)]
                it["alignment_confidence"] = 1.0
                # record block type for later title-strip
                it["block_type"] = str(blk.get("type") or "")

            meta = _detect_list_item(it.get("text"))
            if meta:
                it["list_meta"] = meta

        elif typ in ("image", "table"):
            if blk is not None:
                bbox = blk.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    it["outline"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            if typ == "image":
                it.setdefault("text", "")
                it.setdefault("description", "")
                it.setdefault("image_caption", [])
                it.setdefault("image_footnote", [])
                it.setdefault("image_subcaption", "")
            else:  # table
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

        # sanitize and append
        try:
            sanitize_item_fields(it)
        except Exception:
            pass
        adapted.append(it)

        if isinstance(pg, int):
            page_pos[pg] = page_pos.get(pg, 0) + 1

    # Page-level postprocessing: strip title prefix; drop empty text; optional list splitting
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    others: List[Dict[str, Any]] = []
    for it in adapted:
        p = it.get("page_idx")
        if isinstance(p, int):
            by_page.setdefault(p, []).append(it)
        else:
            others.append(it)

    final_list: List[Dict[str, Any]] = []
    for p in sorted(by_page.keys()):
        fixed = _postprocess_page(by_page[p], enable_list_split=False)
        final_list.extend(fixed)
    # Append items without page_idx, preserving original relative order
    final_list.extend(others)

    # Reindex node_idx/read_order_idx
    for i, it in enumerate(final_list):
        it["node_idx"] = i
        it["read_order_idx"] = i

    return final_list


# =========================
# CLI helpers
# =========================

def adapt_single_file(
    dom_file: str,
    layout_file: Optional[str] = None,
    write_in_place: bool = True,
    output_path: Optional[str] = None,
    suffix: str = DEFAULT_SUFFIX,
) -> str:
    content_list = load_json(dom_file)
    layout_obj = None
    if layout_file:
        layout_obj = load_json(layout_file)
    else:
        cand = os.path.join(os.path.dirname(dom_file), "layout.json")
        layout_obj = load_json(cand) if os.path.exists(cand) else None

    adapted = adapt_content_list(content_list, layout_obj=layout_obj)

    # Optionally ensure page images for downstream
    try:
        pages = sorted({int(it.get("page_idx")) for it in adapted if isinstance(it.get("page_idx"), int)})
    except Exception:
        pages = []
    if pages:
        try:
            ensure_all_page_images_for_pages(os.path.dirname(dom_file), pages, show_progress=True)
        except Exception:
            pass

    out_path = dom_file if write_in_place else (output_path or _make_out_path(dom_file, suffix))
    dump_json(adapted, out_path)
    return out_path


def enrich_content_with_layout(
    dom_files: List[str],
    layout_files: List[str],
    *,
    in_place: bool = False,
    suffix: str = DEFAULT_SUFFIX,
) -> Tuple[int, int]:
    layout_dict: Dict[str, str] = {os.path.dirname(p): p for p in layout_files}
    success = 0
    errors = 0
    for dom in dom_files:
        try:
            layout_obj = None
            if os.path.dirname(dom) in layout_dict:
                layout_obj = load_json(layout_dict[os.path.dirname(dom)])
            content_list = load_json(dom)
            adapted = adapt_content_list(content_list, layout_obj=layout_obj)
            out = dom if in_place else _make_out_path(dom, suffix)
            dump_json(adapted, out)
            success += 1
        except Exception:
            errors += 1
            continue
    return success, errors


def process_directory(dom_dir: str, *, in_place: bool = False, suffix: str = DEFAULT_SUFFIX) -> Tuple[int, int]:
    dom_files = glob.glob(os.path.join(dom_dir, "**", "*content_list.json"), recursive=True)
    layout_files = glob.glob(os.path.join(dom_dir, "**", "*layout.json"), recursive=True)
    return enrich_content_with_layout(dom_files, layout_files, in_place=in_place, suffix=suffix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adapt MinerU content_list with strict index alignment to layout.pdf_info")
    parser.add_argument("--in-dir", dest="in_dir", type=str, help="Root directory containing MinerU outputs")
    parser.add_argument("--in-file", dest="in_file", type=str, default=None, help="Specific content_list.json file")
    parser.add_argument("--layout-file", dest="layout_file", type=str, default=None, help="Specific layout.json file")
    parser.add_argument("--out", dest="out", type=str, default=None, help="Output file when using --in-file; overrides --suffix")
    parser.add_argument("--in-place", dest="in_place", action="store_true", help="Write back to input content_list.json")
    parser.add_argument("--suffix", dest="suffix", type=str, default=DEFAULT_SUFFIX, help=f"When not --in-place, write alongside originals using this suffix (default: {DEFAULT_SUFFIX})")

    args = parser.parse_args()

    if args.in_file:
        out = adapt_single_file(
            args.in_file,
            layout_file=args.layout_file,
            write_in_place=args.in_place,
            output_path=args.out,
            suffix=args.suffix,
        )
        print(out)
    else:
        if not args.in_dir:
            parser.error("Provide --in-dir or --in-file")
        succ, err = process_directory(args.in_dir, in_place=args.in_place, suffix=args.suffix)
        print(f"Enrichment completed: {succ} success, {err} errors")
