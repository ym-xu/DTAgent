import glob
import os
from typing import Any, Dict, List, Optional, Tuple

from .utils import load_json, dump_json


DEFAULT_OUT_BASENAME = "doctree.json"


def _make_out_name(suffix: Optional[str]) -> str:
    if not suffix:
        return DEFAULT_OUT_BASENAME
    # If suffix starts with a dot, append to base name: doctree{suffix}
    return f"doctree{suffix}"


def _choose_content_list_path(dir_path: str, adapted_suffix: str) -> Optional[str]:
    """
    Prefer content_list{adapted_suffix} over content_list.json when both present.
    Returns the chosen path or None if neither exists.
    """
    base = os.path.join(dir_path, "content_list.json")
    adapted = os.path.join(dir_path, f"content_list{adapted_suffix}")
    if os.path.exists(adapted):
        return adapted
    if os.path.exists(base):
        return base
    return None


def build_flat_doctree(
    content_list: List[dict],
    *,
    source_dir: str,
    doc_id: str,
    include_indices: bool = True,
) -> Dict[str, Any]:
    """
    Build a flat DocTree dict with a document root and linear children from an adapted content_list.
    - doc_id: derived from directory name by convention
    - include_indices: when True, add by_page, by_type, id_to_idx for quick access
    """
    children: List[Dict[str, Any]] = []

    # Pre-size structures for indices
    by_page: Dict[int, List[int]] = {}
    by_type: Dict[str, List[int]] = {}
    id_to_idx: Dict[str, int] = {}

    for idx, item in enumerate(content_list):
        typ = item.get("type")
        node_idx = int(item.get("node_idx", idx))
        page_idx = item.get("page_idx")
        node_level = item.get("node_level", -1)

        node_id = f"{doc_id}#{node_idx}"

        child: Dict[str, Any] = {
            "type": typ,
            "node_id": node_id,
            "node_idx": node_idx,
            "page_idx": page_idx,
            "node_level": node_level,
        }

        if typ == "text":
            child["text"] = item.get("text", "")
        elif typ == "image":
            if "img_path" in item:
                child["img_path"] = item["img_path"]
            if "outline" in item:
                child["outline"] = item["outline"]
            # OCR text and description (may be empty strings if not populated)
            if "text" in item:
                child["text"] = item.get("text", "")
            if "description" in item:
                child["description"] = item.get("description", "")
        elif typ == "table":
            # Keep HTML + markdown/plain preview when available
            if "img_path" in item:
                child["img_path"] = item["img_path"]
            if "outline" in item:
                child["outline"] = item["outline"]
            if "table_body" in item:
                child["table_body"] = item["table_body"]
            if "table_text" in item:
                child["table_text"] = item["table_text"]
        elif typ == "equation":
            if "text" in item:
                child["text"] = item["text"]
            if "text_format" in item:
                child["text_format"] = item["text_format"]
        else:
            # Fallback: store known common fields if present without expanding schema
            for k in ("text", "img_path", "outline"):
                if k in item:
                    child[k] = item[k]

        children.append(child)

        if include_indices:
            # by_page
            if isinstance(page_idx, int):  # Keep int keys internally; JSON will stringify on dump
                by_page.setdefault(page_idx, []).append(node_idx)
            # by_type
            if isinstance(typ, str):
                by_type.setdefault(typ, []).append(node_idx)
            # id_to_idx
            id_to_idx[node_id] = node_idx

    root: Dict[str, Any] = {
        "type": "document",
        "doc_id": doc_id,
        "source_path": source_dir,
        "children": children,
    }

    if include_indices:
        root["indices"] = {
            "by_page": by_page,
            "by_type": by_type,
            "id_to_idx": id_to_idx,
        }

    return root


def build_single(
    in_file: str,
    *,
    out_path: Optional[str] = None,
    include_indices: bool = True,
) -> str:
    """
    Build doctree for one content_list file. Returns the written output path.
    """
    content_list = load_json(in_file)
    doc_dir = os.path.dirname(in_file)
    doc_id = os.path.basename(doc_dir) or "document"
    tree = build_flat_doctree(
        content_list,
        source_dir=doc_dir,
        doc_id=doc_id,
        include_indices=include_indices,
    )

    if not out_path:
        out_path = os.path.join(doc_dir, DEFAULT_OUT_BASENAME)
    dump_json(tree, out_path)
    return out_path


def build_directory(
    in_dir: str,
    *,
    adapted_suffix: str = ".adapted.json",
    suffix: Optional[str] = None,
    include_indices: bool = True,
) -> Tuple[int, int]:
    """
    Recursively scan a root dir for MinerU outputs and build doctree files.
    - Chooses content_list{adapted_suffix} when available, else content_list.json
    - Writes doctree{suffix or ".json"} into each document directory
    Returns (success_count, error_count)
    """
    # Collect candidate directories by finding either base or adapted files
    base_files = glob.glob(os.path.join(in_dir, "**", "content_list.json"), recursive=True)
    adapted_files = glob.glob(
        os.path.join(in_dir, "**", f"content_list{adapted_suffix}"), recursive=True
    )
    cand_dirs = {os.path.dirname(p) for p in base_files + adapted_files}

    out_name = _make_out_name(suffix)
    success = 0
    errors = 0

    for d in sorted(cand_dirs):
        src = _choose_content_list_path(d, adapted_suffix)
        if not src:
            continue
        try:
            content_list = load_json(src)
            doc_id = os.path.basename(d) or "document"
            tree = build_flat_doctree(
                content_list,
                source_dir=d,
                doc_id=doc_id,
                include_indices=include_indices,
            )
            out_path = os.path.join(d, out_name)
            dump_json(tree, out_path)
            success += 1
        except Exception:
            errors += 1
            continue

    return success, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a flat DocTree from MinerU content_list")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-dir", dest="in_dir", type=str, help="Root directory containing MinerU outputs")
    group.add_argument("--in-file", dest="in_file", type=str, help="Specific content_list(.adapted).json file")

    parser.add_argument("--out", dest="out", type=str, default=None, help="Output file when using --in-file")
    parser.add_argument("--suffix", dest="suffix", type=str, default=None, help="When using --in-dir, write doctree{suffix} (e.g., .flat.json)")
    parser.add_argument("--adapted-suffix", dest="adapted_suffix", type=str, default=".adapted.json", help="Preferred suffix for adapted content_list in directory mode")
    parser.add_argument("--no-indices", dest="no_indices", action="store_true", help="Do not include indices in doctree output")

    args = parser.parse_args()
    include_indices = not args.no_indices

    if args.in_file:
        out = build_single(args.in_file, out_path=args.out, include_indices=include_indices)
        print(out)
    else:
        succ, err = build_directory(
            args.in_dir,
            adapted_suffix=args.adapted_suffix,
            suffix=args.suffix,
            include_indices=include_indices,
        )
        print(f"Build completed: {succ} success, {err} errors")

