import glob
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from .utils import load_json, dump_json


DEFAULT_SUFFIX = ".adapted.json"


def _make_out_path(src_path: str, suffix: str) -> str:
    """Construct an output path by appending a suffix.

    If src_path endswith .json, replace the extension with the provided suffix
    (e.g., suffix=".mm.json" -> content_list.mm.json). Otherwise, append suffix.
    """
    if src_path.endswith(".json"):
        return src_path[:-5] + suffix
    return src_path + suffix


def _build_layout_index(layout_obj: Any) -> Tuple[Dict[str, Any], List[Tuple[str, Any]]]:
    """
    Build an index from image filename -> bbox/outline from layout.json.
    Returns:
      - by_basename: dict mapping basename.jpg -> outline/bbox
      - by_path: list of (image_path, outline/bbox) for partial matching fallback
    The extraction logic follows the user-provided recursive search:
    - find dict entries with key 'image_path' (str) and sibling key 'bbox'
    - source root is layout.get('pdf_info', layout)
    """
    src = layout_obj.get("pdf_info", layout_obj) if isinstance(layout_obj, dict) else layout_obj
    layout_info: Dict[str, Any] = {}

    def recursive_search(obj: Any):
        if isinstance(obj, dict):
            # If this dict has an image_path and bbox, record it
            image_path = obj.get("image_path")
            if isinstance(image_path, str) and "bbox" in obj:
                layout_info[image_path] = obj["bbox"]
            # Recurse into values
            for v in obj.values():
                recursive_search(v)
        elif isinstance(obj, list):
            for it in obj:
                recursive_search(it)

    recursive_search(src)

    by_basename: Dict[str, Any] = {}
    by_path_pairs: List[Tuple[str, Any]] = []
    for path, bbox in layout_info.items():
        base = os.path.basename(path)
        by_basename[base] = bbox
        by_path_pairs.append((path, bbox))

    return by_basename, by_path_pairs


def adapt_content_list(
    content_list: List[dict],
    layout_obj: Optional[Any] = None,
    ocr_image_func: Optional[Callable[[str], Optional[str]]] = None,
    describe_image_func: Optional[Callable[[str], Optional[str]]] = None,
) -> List[dict]:
    """
    Adapt a MinerU content_list to an enriched list suitable for MM DocTree build.

    - Add node_idx (0..n-1)
    - Rename text_level -> node_level if present (do not alter existing node_level)
    - For non-text nodes (image/table/equation/code), ensure node_level = -1 if not set
    - Attach outline from layout.json for image/table if available
    - Optionally add OCR text to images and LLM description (if callables provided)
    """
    by_basename: Dict[str, Any] = {}
    by_path_pairs: List[Tuple[str, Any]] = []
    if layout_obj is not None:
        try:
            by_basename, by_path_pairs = _build_layout_index(layout_obj)
        except Exception:
            # Fail silently; enrichment will just be skipped
            by_basename, by_path_pairs = {}, []

    def attach_outline(item: dict) -> None:
        img_path = item.get("img_path")
        if not isinstance(img_path, str):
            return
        base = os.path.basename(img_path)
        if base in by_basename:
            item["outline"] = by_basename[base]
            return
        # partial matching fallback
        for layout_path, bbox in by_path_pairs:
            if base and base in layout_path:
                item["outline"] = bbox
                return

    adapted: List[dict] = []
    for idx, item in enumerate(content_list):
        it = dict(item)  # shallow copy
        it["node_idx"] = idx

        # Rename text_level -> node_level if present
        if "text_level" in it and "node_level" not in it:
            it["node_level"] = it.pop("text_level")

        # If node_level still missing (e.g., non-title text), default to -1
        if "node_level" not in it:
            it["node_level"] = -1

        typ = it.get("type")

        # Attach outline for images/tables if layout is provided
        if typ in {"image", "table"}:
            attach_outline(it)

            # Optional OCR and description for images
            if typ == "image":
                if ocr_image_func is not None and not it.get("text"):
                    try:
                        text = ocr_image_func(it["img_path"])  # may return None
                        if text:
                            it["text"] = text
                    except Exception:
                        pass
                if describe_image_func is not None and not it.get("description"):
                    try:
                        desc = describe_image_func(it["img_path"])  # may return None
                        if desc:
                            it["description"] = desc
                    except Exception:
                        pass

                # Ensure keys exist for downstream consumers
                it.setdefault("text", "")
                it.setdefault("description", "")

        adapted.append(it)

    return adapted


def adapt_single_file(
    dom_file: str,
    layout_file: Optional[str] = None,
    ocr_image_func: Optional[Callable[[str], Optional[str]]] = None,
    describe_image_func: Optional[Callable[[str], Optional[str]]] = None,
    write_in_place: bool = True,
    output_path: Optional[str] = None,
    suffix: str = DEFAULT_SUFFIX,
) -> str:
    """
    Adapt one content_list.json with optional layout.json enrichment.
    Returns the output path written.
    """
    content_list = load_json(dom_file)
    layout_obj = load_json(layout_file) if layout_file else None

    adapted = adapt_content_list(
        content_list,
        layout_obj=layout_obj,
        ocr_image_func=ocr_image_func,
        describe_image_func=describe_image_func,
    )

    if write_in_place:
        out_path = dom_file
    else:
        out_path = output_path or _make_out_path(dom_file, suffix)

    dump_json(adapted, out_path)
    return out_path


def enrich_content_with_layout(
    dom_files: List[str],
    layout_files: List[str],
    *,
    in_place: bool = False,
    suffix: str = DEFAULT_SUFFIX,
) -> Tuple[int, int]:
    """
    Enrich a batch of content_list files with layout info (outline/bbox for images/tables),
    following the user-provided approach. Updates files in place and returns (success, errors).
    Also adds node_idx and sets node_level for non-text types to -1 if absent.
    """
    layout_dict: Dict[str, str] = {}
    for layout_file in layout_files:
        layout_dir = os.path.dirname(layout_file)
        layout_dict[layout_dir] = layout_file

    success_count = 0
    error_count = 0

    for dom_file in dom_files:
        try:
            dom_dir_path = os.path.dirname(dom_file)
            layout_obj = None
            if dom_dir_path in layout_dict:
                layout_path = layout_dict[dom_dir_path]
                layout_obj = load_json(layout_path)

            # Load content_list
            content_list: List[dict] = load_json(dom_file)
            # Adapt and write back in place
            adapted = adapt_content_list(content_list, layout_obj=layout_obj)
            if in_place:
                out_path = dom_file
            else:
                out_path = _make_out_path(dom_file, suffix)
            dump_json(adapted, out_path)
            success_count += 1
        except Exception:
            error_count += 1
            continue

    return success_count, error_count


def process_directory(dom_dir: str, *, in_place: bool = False, suffix: str = DEFAULT_SUFFIX) -> Tuple[int, int]:
    """Convenience helper: scan a root directory for content_list/layout files and enrich.

    - When in_place=False (default), writes alongside originals as *.adapted.json
    - When in_place=True, overwrites the original content_list.json files
    """
    dom_files = glob.glob(os.path.join(dom_dir, "**", "*content_list.json"), recursive=True)
    layout_files = glob.glob(os.path.join(dom_dir, "**", "*layout.json"), recursive=True)
    return enrich_content_with_layout(dom_files, layout_files, in_place=in_place, suffix=suffix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adapt MinerU content_list with layout enrichment")
    parser.add_argument("--in-dir", dest="in_dir", type=str, help="Root directory that contains mineru outputs")
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
