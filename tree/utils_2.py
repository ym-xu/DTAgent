import json
import logging
import os
import re
import logging
from html.parser import HTMLParser
from typing import Any, List, Optional, Tuple, Dict
import sys
import random
import glob

from .llm_clients import gpt_llm_call

logger = logging.getLogger(__name__)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def html_table_to_markdown(html_str: str) -> Optional[str]:
    """
    Convert a single HTML <table> to Markdown table text when possible.
    Returns None if no rows detected. On failure, returns a plain-text fallback.
    """
    try:
        parser = _SimpleTableParser()
        parser.feed(html_str)
        md = _rows_to_markdown(parser.rows, parser.row_is_header)
        if md:
            return md
    except Exception as e:
        logger.info("HTML→Markdown table parse failed: %s", e)

    # Fallback: strip tags and collapse whitespace as plain text
    try:
        text = re.sub(r"<[^>]+>", " ", html_str)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
    except Exception:
        return None

def sanitize_item_fields(item: dict) -> None:
    """In-place sanitize textual fields based on node type."""
    typ = item.get("type")
    if typ == "text" or typ == "equation":
        if isinstance(item.get("text"), str):
            item["text"] = sanitize_text(item.get("text")) or ""
    elif typ == "image":
        if isinstance(item.get("text"), str):
            item["text"] = sanitize_text(item.get("text")) or ""
        if isinstance(item.get("description"), str):
            item["description"] = sanitize_text(item.get("description")) or ""
        # Ensure caption/footnote lists exist and sanitize entries
        cap = item.get("image_caption")
        if not isinstance(cap, list):
            item["image_caption"] = []
        else:
            item["image_caption"] = _sanitize_string_list(cap)
        fn = item.get("image_footnote")
        if not isinstance(fn, list):
            item["image_footnote"] = []
        else:
            item["image_footnote"] = _sanitize_string_list(fn)
    elif typ == "list":
        # Normalize list items and subtype
        if not isinstance(item.get("list_items"), list):
            item["list_items"] = []
        else:
            item["list_items"] = _sanitize_string_list(item["list_items"])
        if isinstance(item.get("sub_type"), str):
            item["sub_type"] = (item.get("sub_type") or "").strip() or "text"
        else:
            item["sub_type"] = "text"
    else:
        # For other types we keep as-is; do not touch table_body HTML etc.
        # But for table, normalize caption/footnote lists if present
        if typ == "table":
            cap = item.get("table_caption")
            if not isinstance(cap, list):
                item["table_caption"] = []
            else:
                item["table_caption"] = _sanitize_string_list(cap)
            fn = item.get("table_footnote")
            if not isinstance(fn, list):
                item["table_footnote"] = []
            else:
                item["table_footnote"] = _sanitize_string_list(fn)


def page_images_dir(source_dir: str) -> str:
    return os.path.join(source_dir or ".", "images")

def extract_page(source_dir: str, page_idx: int) -> Optional[str]:
    img_dir = page_images_dir(source_dir)
    return os.path.join(img_dir, f"page_{page_idx}.{'png'}")
     
def detect_mode_and_twoup(source_dir: str, pages: List[int], model: str = "gpt-4o-mini", api_key: Optional[str] = None,) -> Tuple[str, int, float]:
    cand_page_num = [pages[0],pages[-1]]
    cand_page_num.extend(random.sample(pages[1:-1], 2))
    print(cand_page_num)
    cand_pages :List[str] = []
    for i in cand_page_num: cand_pages.append(extract_page(source_dir, i))

    prompt = (
        "You are given 1-4 pages of a PDF as images.\n"
        "Classify: (A) overall document type; (B) whether each physical page contains two logical pages side-by-side (a two-up scan).\n"
        "- mode: 'slides' if this is a slide deck/presentation; otherwise 'doc' for a typical document/report/book/manual.\n"
        "- two_up: true if a page image appears to include two distinct page halves (left and right pages on one sheet), typical in scanned books or spreads; false otherwise.\n"
        "  Do NOT confuse two_up with two-column text layout on a single page; two_up means two separate pages visible side-by-side.\n"
        "Output only one JSON object with keys: mode, two_up, confidence.\n"
        "Examples of slides: large titles, bullet lists, minimal continuous paragraphs, one dominant visual per page.\n"
        "Examples of doc: dense paragraphs, figures/tables inline, running headers/footers, TOC pages.\n"
    )
    try:
        raw = gpt_llm_call(messages=[{"role": "user", "content": prompt}], images=cand_pages, model=model, api_key=api_key, json_mode=True)
        print(f"[{model}] raw output:", raw)
        obj = json.loads(raw)
        mode = str(obj.get("mode", "doc")).strip().lower()
        mode_str = "slides" if mode == "slides" else "doc"
        two_up = obj.get("two_up", False)
        two_up_bool = bool(two_up)
        conf = obj.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0

        cols = 2 if two_up_bool else 1
        return mode_str, cols, conf_f
    except Exception as e:
        print(f"[detect_mode_and_twoup] failed: {e}")
        return "doc", 1, 0.0

def _shorten_text(text: str, max_len: int = 80) -> str:
    s = (text or "").strip().replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 1)].rstrip() + "…"

def _media_title(n: dict) -> str:
    """Derive a short label for an image/table node using caption/description/path."""
    t = n.get("type")
    if t == "image":
        caps = n.get("image_caption") if isinstance(n.get("image_caption"), list) else []
        cap = caps[0] if caps else ""
        if cap and isinstance(cap, str):
            return cap
        desc = n.get("description") if isinstance(n.get("description"), str) else ""
        if desc:
            return desc
        p = n.get("img_path") if isinstance(n.get("img_path"), str) else "image"
        import os
        return os.path.basename(p) or "image"
    if t == "table":
        caps = n.get("table_caption") if isinstance(n.get("table_caption"), list) else []
        cap = caps[0] if caps else ""
        if cap and isinstance(cap, str):
            return cap
        return "table"
    return str(n.get("text", ""))

def format_heading_outline(
    flat_root: dict,
    *,
    by_logical_page: bool = False,
    include_meta: bool = True,
    max_len: int = 80,
    include_media: bool = True,
) -> list:
    """
    Build a list of one-line strings representing the heading hierarchy for quick inspection.
    - Indentation reflects node_level (2 spaces per level - 1).
    - Shows node_idx, page/logical_page, level, optional via/frozen flags.
    """
    lines: list = []
    items = list(flat_root.get("children", [])) if isinstance(flat_root, dict) else []
    for n in items:
        t = n.get("type")
        if t not in ("text", "image", "table"):
            continue
        if t in ("image", "table") and not include_media:
            continue
        try:
            lvl = int(n.get("node_level", -1))
        except Exception:
            continue
        if lvl < 1:
            continue
        indent = "  " * max(0, lvl - 1)
        idx = n.get("node_idx")
        lp = n.get("logical_page")
        p = n.get("page_idx")
        # choose label
        label = _media_title(n) if t in ("image", "table") else str(n.get("text", ""))
        text = _shorten_text(label, max_len=max_len)
        meta = n.get("heading_meta") if include_meta and isinstance(n.get("heading_meta"), dict) else None
        flags: list[str] = []
        if meta:
            if meta.get("via"):
                flags.append(str(meta.get("via")))
            if meta.get("frozen"):
                flags.append("frozen")
            if meta.get("inserted"):
                flags.append("inserted")
            if meta.get("corrected_cross_level"):
                flags.append("corrected")
        # media marker
        if t == "image":
            flags.append("image")
        elif t == "table":
            flags.append("table")
        where = f"lp={lp}" if by_logical_page else f"p={p}"
        flag_str = (" [" + ",".join(flags) + "]") if flags else ""
        lines.append(f"{indent}- L{lvl} idx={idx} {where}: {text}{flag_str}")
    return lines

def print_heading_outline(
    flat_root: dict,
    *,
    by_logical_page: bool = True,
    include_meta: bool = True,
    max_len: int = 80,
    include_media: bool = True,
) -> None:
    """Print heading outline to stdout for quick manual inspection."""
    try:
        lines = format_heading_outline(
            flat_root,
            by_logical_page=by_logical_page,
            include_meta=include_meta,
            max_len=max_len,
            include_media=include_media,
        )
        for ln in lines:
            print(ln)
    except Exception as e:
        print(f"[print_heading_outline] failed: {e}")


# local pages
def page_indices_from_children(children: List[Dict[str, Any]]) -> List[int]:
    pages = sorted({ch.get("page_idx") for ch in children if isinstance(ch.get("page_idx"), int)})
    return [int(p) for p in pages]

def detect_two_up_range_via_gpt(image_path: str, *, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """
    Ask GPT-4o to return a page number range [start,end] for a two-up scanned page image.
    Output JSON must be: {"range": [start,end] | null}. We accept only end == start+1.
    """
    try:
        from .llm_clients import gpt_llm_call
    except Exception:
        return None

    prompt = (
        'You are given one image of a physical PDF page that may contain two logical pages side-by-side (two-up scan).\n'
        'Return the printed page number range as Arabic numerals if you are confident they are the header/footer page numbers,\n'
        'NOT numbers from the body text.\n'
        'Rules:\n'
        '- Output strictly one JSON object: {"range": [start,end]} or {"range": null}.\n'
        '- Use integers only, and require end == start + 1 when returning a range. If unsure, use null (not the string).\n'
        '- Do not return Roman numerals, text like "cover", or placeholders like "null"/"none".\n'
    )

    messages = [
        {"role": "system", "content": "You only output JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = gpt_llm_call(messages, images=[image_path], model=model, api_key=api_key, json_mode=True)
        try:
            print(f"[gpt-4o two_up_range] {image_path}: {raw}")
        except Exception:
            pass
        obj = json.loads(raw)
        rng = obj.get("range", None)
        if isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, int) for x in rng):
            start, end = rng
            if end == start + 1:
                return (start, end)
        return None
    except Exception as e:
        try:
            print(f"[gpt-4o two_up_range] failed for {image_path}: {e}")
        except Exception:
            pass
        return None

def render_pdf_page_to_image(pdf_path: str, page_idx: int, out_dir: Optional[str] = None, *, dpi: int = 150) -> Optional[str]:
    """
    Render one PDF page to <out_dir>/images/page_{idx}.png (0-based idx by user spec). Returns saved path or None.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        logger.info("PyMuPDF not available; cannot render page %s from %s", page_idx, pdf_path)
        return None
    try:
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= len(doc):
            return None
        page = doc.load_page(page_idx)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_base = out_dir or page_images_dir(os.path.dirname(pdf_path))
        os.makedirs(out_base, exist_ok=True)
        out_path = os.path.join(out_base, f"page_{page_idx}.png")
        pix.save(out_path)
        return out_path
    except Exception as e:
        logger.info("Failed to render PDF page %s: %s", page_idx, e)
        return None

def find_pdf_in_dir(source_dir: str) -> Optional[str]:
    pdfs = glob.glob(os.path.join(source_dir or ".", "*.pdf"))
    return pdfs[0] if pdfs else None

def find_existing_page_image(source_dir: str, page_idx: int) -> Optional[str]:
    """
    Look for an existing page image in <source_dir>/images using common patterns.
    Primary pattern per user spec: page_{idx}.(png|jpg|jpeg). Also try 1-based.
    """
    img_dir = page_images_dir(source_dir)
    if not os.path.isdir(img_dir):
        return None
    patterns = []
    # 0-based
    for ext in ("png", "jpg", "jpeg"):
        patterns.append(os.path.join(img_dir, f"page_{page_idx}.{ext}"))
    # 1-based fallback
    one_based = page_idx + 1
    for ext in ("png", "jpg", "jpeg"):
        patterns.append(os.path.join(img_dir, f"page_{one_based}.{ext}"))
    for p in patterns:
        if os.path.exists(p):
            return p
    # also try subfolder patterns like images/pages/page_{n}.*
    for n in (page_idx, page_idx + 1):
        for ext in ("png", "jpg", "jpeg"):
            for pat in (
                os.path.join(img_dir, "pages", f"page_{n}.{ext}"),
                os.path.join(img_dir, "pages", f"{n:03d}.{ext}"),
            ):
                if os.path.exists(pat):
                    return pat
    return None

def ensure_page_image(source_dir: str, page_idx: int, *, pdf_path: Optional[str] = None, dpi: int = 150) -> Optional[str]:
    """
    Ensure there is a page image for page_idx under <source_dir>/images named page_{idx}.ext.
    - If exists, return its path
    - Else render from sibling PDF (first *.pdf or provided pdf_path) and save as page_{idx}.png
    Returns the path or None on failure.
    """
    # Existing
    existing = find_existing_page_image(source_dir, page_idx)
    if existing:
        return existing
    # Render
    pdf = pdf_path or find_pdf_in_dir(source_dir)
    if not pdf:
        return None
    return render_pdf_page_to_image(pdf, page_idx, out_dir=page_images_dir(source_dir), dpi=dpi)


def ensure_all_page_images_for_pages(
    source_dir: str,
    pages: List[int],
    *,
    dpi: int = 150,
    show_progress: bool = False,
) -> List[Optional[str]]:
    paths: List[Optional[str]] = []
    total = len(pages)
    bar_width = 30
    def _print_progress(i: int) -> None:
        if not show_progress:
            return
        done = int(bar_width * (i / max(total, 1)))
        bar = "#" * done + "-" * (bar_width - done)
        sys.stdout.write(f"\rRendering pages: [{bar}] {i}/{total}")
        sys.stdout.flush()

    if show_progress and total:
        sys.stdout.write(f"Rendering {total} pages to images...\n")
        sys.stdout.flush()

    for i, p in enumerate(pages, start=1):
        try:
            paths.append(ensure_page_image(source_dir, p, dpi=dpi))
        except Exception:
            paths.append(None)
        _print_progress(i)

    if show_progress and total:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return paths

def detect_arabic_page_number_via_gpt(image_path: str, *, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> Optional[int]:
    """
    Ask GPT-4o to return an Arabic page number for a single physical page image.
    Output JSON must be: {"page_number": int|null}
    - Only accept an integer page number; if unsure, return null.
    """
    try:
        from .llm_clients import gpt_llm_call
    except Exception:
        return None

    prompt = (
        'You are given one image of a physical PDF page.\n'
        'Return the printed page number only if you are confident it is the page header/footer page number,\n'
        'NOT a number from the body text.\n'
        'Rules:\n'
        '- Output strictly one JSON object: {"page_number": int|null}.\n'
        '- Use an integer (Arabic numerals) if confident; otherwise use null (not the string).\n'
        '- Do not return Roman numerals, text like "cover", or placeholders like "null"/"none".\n'
    )

    messages = [
        {"role": "system", "content": "You only output JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = gpt_llm_call(messages, images=[image_path], model=model, api_key=api_key, json_mode=True)
        try:
            print(f"[gpt-4o page_number] {image_path}: {raw}")
        except Exception:
            pass
        obj = json.loads(raw)
        val = obj.get("page_number", None)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            try:
                return int(val)
            except Exception:
                return None
        return None
    except Exception as e:
        try:
            print(f"[gpt-4o page_number] failed for {image_path}: {e}")
        except Exception:
            pass
        return None

def _longest_consecutive_run(pairs: List[Tuple[int, int]]) -> Tuple[Optional[int], List[int]]:
    """
    Given pairs (page_idx, number) with both ints, find the diff d = number - page_idx that forms
    the longest run of consecutive page_idx with constant d. Returns (d, run_pages).
    """
    if not pairs:
        return None, []
    by_diff: Dict[int, List[int]] = {}
    for p, n in pairs:
        by_diff.setdefault(n - p, []).append(p)
    best_diff: Optional[int] = None
    best_run: List[int] = []
    for d, plist in by_diff.items():
        s = sorted(set(plist))
        run: List[int] = []
        cur: List[int] = []
        prev: Optional[int] = None
        for x in s:
            if prev is None or x == prev + 1:
                cur.append(x)
            else:
                if len(cur) > len(run):
                    run = cur
                cur = [x]
            prev = x
        if len(cur) > len(run):
            run = cur
        if len(run) > len(best_run):
            best_run = run
            best_diff = d
    return best_diff, best_run


def assign_logical_pages_via_offset(
    flat_root: Dict[str, Any],
    *,
    scan_limit: int = 15,
    min_run: int = 2,
    early_stop_run: int = 6,
) -> Dict[str, Any]:
    """
    Detect Arabic page numbers on the first `scan_limit` pages, find a consecutive run to determine
    a constant offset (page_number - page_idx), then assign logical_page for all nodes accordingly.
    - Only sets node["logical_page"] as a string, when computed value >= 1; otherwise leaves it absent.
    """
    if not isinstance(flat_root, dict):
        return flat_root
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))
    pages = page_indices_from_children(children)
    if not pages:
        return flat_root
    source_dir = flat_root.get("source_path")
    if not isinstance(source_dir, str) or not source_dir:
        return flat_root

    # ensure images for front pages
    front = [p for p in pages if p - pages[0] < scan_limit][:scan_limit]
    try:
        ensure_all_page_images_for_pages(source_dir, front, show_progress=False)
    except Exception:
        pass

    columns = flat_root.get("columns", 1)
    if columns == 2:
        # Two-up: detect ranges and fit diff2 = start - 2*page_idx
        detections: List[Tuple[int, int]] = []  # (page_idx, start)
        # Early-stop tracking
        streak_d2: Optional[int] = None
        streak_len = 0
        streak_pages: List[int] = []
        for p in front:
            img = find_existing_page_image(source_dir, p)
            if not img:
                continue
            rng = detect_two_up_range_via_gpt(img)
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                start, end = rng
                if isinstance(start, int) and isinstance(end, int) and end == start + 1:
                    detections.append((p, start))
                    d2 = start - 2 * p
                    if streak_d2 is None or d2 != streak_d2 or (streak_pages and p != streak_pages[-1] + 1):
                        streak_d2 = d2
                        streak_len = 1
                        streak_pages = [p]
                    else:
                        streak_len += 1
                        streak_pages.append(p)
                    if early_stop_run and streak_len >= early_stop_run:
                        best_d2 = streak_d2
                        best_run2 = list(streak_pages)
                        break
        if 'best_d2' not in locals():
            by_diff2: Dict[int, List[int]] = {}
            for p, start in detections:
                by_diff2.setdefault(start - 2 * p, []).append(p)
            best_d2 = None  # type: Optional[int]
            best_run2: List[int] = []
            for d2, plist in by_diff2.items():
                s = sorted(set(plist))
                cur: List[int] = []
                run: List[int] = []
                prev: Optional[int] = None
                for x in s:
                    if prev is None or x == prev + 1:
                        cur.append(x)
                    else:
                        if len(cur) > len(run):
                            run = cur
                        cur = [x]
                    prev = x
                if len(cur) > len(run):
                    run = cur
                if len(run) > len(best_run2):
                    best_run2 = run
                    best_d2 = d2
        if best_d2 is None or len(best_run2) < max(1, min_run):
            return flat_root
        page_set = set(pages)
        assigned_pages: set[int] = set()
        for ch in children:
            p = ch.get("page_idx")
            if not isinstance(p, int) or p not in page_set:
                continue
            start = 2 * p + best_d2
            if start >= 1:
                ch["logical_page"] = f"{start}-{start+1}"
                assigned_pages.add(p)
            else:
                if "logical_page" in ch:
                    del ch["logical_page"]
        # Label pages before the first assigned page as cover
        if assigned_pages:
            earliest = min(assigned_pages)
            for ch in children:
                p = ch.get("page_idx")
                if isinstance(p, int) and p in page_set and p < earliest:
                    ch["logical_page"] = "cover"
    else:
        # Single-page mapping: diff = num - page_idx
        detections: List[Tuple[int, int]] = []
        # Early-stop tracking
        streak_d: Optional[int] = None
        streak_len = 0
        streak_pages: List[int] = []
        for p in front:
            img = find_existing_page_image(source_dir, p)
            if not img:
                continue
            num = detect_arabic_page_number_via_gpt(img)
            if isinstance(num, int):
                detections.append((p, num))
                d = num - p
                if streak_d is None or d != streak_d or (streak_pages and p != streak_pages[-1] + 1):
                    streak_d = d
                    streak_len = 1
                    streak_pages = [p]
                else:
                    streak_len += 1
                    streak_pages.append(p)
                if early_stop_run and streak_len >= early_stop_run:
                    run_pages = list(streak_pages)
                    # Assign using this d immediately
                    pass_d = d
                    break
        if 'pass_d' in locals():
            d = pass_d
        else:
            d, run_pages = _longest_consecutive_run(detections)
        if d is None or len(run_pages) < max(1, min_run):
            return flat_root
        page_set = set(pages)
        assigned_pages: set[int] = set()
        for ch in children:
            p = ch.get("page_idx")
            if not isinstance(p, int) or p not in page_set:
                continue
            val = p + d
            if val >= 1:
                ch["logical_page"] = str(val)
                assigned_pages.add(p)
            else:
                if "logical_page" in ch:
                    del ch["logical_page"]
        # Label pages before the first assigned page as cover
        if assigned_pages:
            earliest = min(assigned_pages)
            for ch in children:
                p = ch.get("page_idx")
                if isinstance(p, int) and p in page_set and p < earliest:
                    ch["logical_page"] = "cover"
    return flat_root
