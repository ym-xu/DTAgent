import glob
import os
from typing import Any, Dict, List, Optional, Tuple
from tree.utils import load_json, print_heading_outline

from .utils import (
    load_json,
    dump_json,
    ensure_all_page_images_for_pages,
    detect_mode_and_columns,
)
from .toc import find_toc_pages, consolidate_toc_v2
from .llm_clients import gpt_llm_call
from .headings import merge_adjacent_headings
from .utils import page_indices_from_children
from .utils import assign_logical_pages_via_offset
from .media_enhance import enhance_media_on_root
from .caption_cleanup import cleanup_captions_with_llm
from .heading_levels import assign_heading_levels_case1, assign_non_toc_levels_with_llm, assign_heading_levels_case2, assign_heading_levels_case3


ENHANCED_FILENAME = "content_list.enhanced.json"


def _choose_content_list_path(dir_path: str, adapted_suffix: str) -> Optional[str]:
    base = os.path.join(dir_path, "content_list.json")
    adapted = os.path.join(dir_path, f"content_list{adapted_suffix}")
    if os.path.exists(adapted):
        return adapted
    if os.path.exists(base):
        return base
    return None


def _wrap_root(items: List[dict], source_dir: str) -> Dict[str, Any]:
    doc_id = os.path.basename(source_dir) or "document"
    return {"type": "document", "doc_id": doc_id, "source_path": source_dir, "children": list(items)}


def _unwrap_children(root: Dict[str, Any]) -> List[dict]:
    return list(root.get("children", []))


def enhance_single(
    in_file: str,
    *,
    out_path: Optional[str] = None,
) -> str:
    """
    Enhance one adapted content_list file and write a single enhanced JSON:
      {version, meta:{mode, columns, mode_confidence, ...}, content_list:[...]}
    Slides are left unmodified (content_list passthrough) at this step.
    """
    items = load_json(in_file)
    if isinstance(items, dict) and "content_list" in items:
        items = items["content_list"]
    if not isinstance(items, list):
        raise ValueError("input is not a content_list array or enhanced object with content_list")
    src_dir = os.path.dirname(in_file)

    # 第一步：确认 doc/slides, 确认 two-up类型
    # Ensure front-page images for detection (adapter already renders all, but be safe)
    pages = sorted({int(it.get("page_idx")) for it in items if isinstance(it.get("page_idx"), int)})
    if pages:
        try:
            ensure_all_page_images_for_pages(src_dir, pages, show_progress=False)
        except Exception:
            pass
    print('start mode and two-up detection')
    mode, cols, conf = detect_mode_and_columns(src_dir, pages)

    root = _wrap_root(items, src_dir)
    # Only keep mode/columns (and confidence) in meta; all other enhancements write back to nodes
    meta: Dict[str, Any] = {"mode": mode, "mode_confidence": conf, "columns": cols}

    if mode == "doc":
        # 第二步，寻找目录并构建目录节点
        # ToC detection + consolidation (with images)
        try:
            print('start toc detection')
            cand = find_toc_pages(root, gpt_llm_call, min_confidence=0.75, images_only=True)
            print('cand: ', cand)
            if cand:
                root = consolidate_toc_v2(root, cand, gpt_llm_call)
        except Exception:
            pass
        # 第三步，检测逻辑页面
        # Assign logical pages via offset (two-up aware)
        try:
            print('start add logical pages')
            root = assign_logical_pages_via_offset(root)
        except Exception:
            pass

        # 查找遗漏的图像/表格标题
        # Media enhance for large media (image/table) using VLM (no crop, red bbox)
        try:
            print('start image table enhance')
            root, _media_meta = enhance_media_on_root(root, model="gpt-4o", area_px_threshold=50000, override=False)
            print(_media_meta)
        except Exception:
            pass

        # 合并截断的标题
        # Merge adjacent headings (heuristics + LLM confirmation)
        try:
            root = merge_adjacent_headings(root, use_llm=True, llm_model="gpt-4o")
        except Exception:
            pass

        # 图像标题消岐
        # Disambiguation: caption-as-title cleanup via LLM
        try:
            print('start caption/title disambiguation via LLM')
            root, _cap_stats = cleanup_captions_with_llm(
                root,
                llm_model="gpt-4o",
                window_before=4,
                window_after=4,
                writeback_media_fields=True,
                require_confidence=True,
                min_confidence=0.85,
            )
            print(_cap_stats)
        except Exception:
            pass

        # 分配标题
        # Heading levels assignment by TOC: choose Case 1 (with page numbers) or Case 2 (no page numbers)
        try:
            children_for_toc = list(root.get('children', []))
            toc_node = next((n for n in children_for_toc if isinstance(n, dict) and n.get('type') == 'toc' and isinstance(n.get('headings'), list)), None)
            has_pages = False
            if toc_node:
                def _has_page(h):
                    if isinstance(h, dict) and isinstance(h.get('page'), int):
                        return True
                    for ch in (h.get('children') or []) if isinstance(h, dict) else []:
                        if _has_page(ch):
                            return True
                    return False
                heads = toc_node.get('headings') or []
                has_pages = any(_has_page(h) for h in heads)
            if toc_node is None:
                print('start heading level assignment (Case 3)')
                root, _h3_stats = assign_heading_levels_case3(
                    root,
                    llm_model="gpt-4o",
                    max_level=4,
                    promote_media=False,
                )
                print(_h3_stats)
            elif has_pages:
                print('start heading level assignment (Case 1)')
                root, _h_stats = assign_heading_levels_case1(
                    root,
                    high_threshold=0.85,
                    mid_threshold=0.72,
                    max_level=4,
                )
                print(_h_stats)
            else:
                print('start heading level assignment (Case 2)')
                root, _h2_stats = assign_heading_levels_case2(
                    root,
                    window_size=20,
                    llm_model="gpt-4o",
                )
                print(_h2_stats)
        except Exception as e:
            try:
                print('heading assignment failed:', e)
            except Exception:
                pass

        # Heading levels (Case 1, anchors by TOC already applied): assign non-TOC headings via LLM under constraints
        print('start non-TOC heading levels via LLM (Case 1)')
        try:
            root, _non_toc_stats = assign_non_toc_levels_with_llm(
                root,
                llm_model="gpt-4o",
                max_level=4,
                chunk_size=40,
            )
            try:
                print(_non_toc_stats)
            except Exception:
                pass
        except Exception as e:
            try:
                print('non-TOC assignment failed:', e)
            except Exception:
                pass
        
        print_heading_outline(root, by_logical_page=True, include_meta=True, max_len=80)

    enhanced = {
        "version": "enhanced-1",
        "meta": meta,
        "content_list": _unwrap_children(root),
    }

    out = out_path or os.path.join(src_dir, ENHANCED_FILENAME)
    dump_json(enhanced, out)
    return out


def enhance_directory(
    in_dir: str,
    *,
    adapted_suffix: str = ".adapted.json",
) -> Tuple[int, int]:
    base_files = glob.glob(os.path.join(in_dir, "**", "content_list.json"), recursive=True)
    adapted_files = glob.glob(os.path.join(in_dir, "**", f"content_list{adapted_suffix}"), recursive=True)
    cand_dirs = {os.path.dirname(p) for p in base_files + adapted_files}

    succ = 0
    err = 0
    for d in sorted(cand_dirs):
        src = _choose_content_list_path(d, adapted_suffix)
        if not src:
            continue
        try:
            enhance_single(src, out_path=os.path.join(d, ENHANCED_FILENAME))
            succ += 1
        except Exception:
            err += 1
            continue
    return succ, err


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhance adapted content_list into a single enhanced JSON (meta + content_list)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-file", dest="in_file", type=str, help="Adapted content_list(.json/.adapted.json) file")
    group.add_argument("--in-dir", dest="in_dir", type=str, help="Root directory containing adapted content_lists")

    parser.add_argument("--out", dest="out", type=str, default=None, help="Output enhanced JSON when using --in-file (default: content_list.enhanced.json beside input)")
    parser.add_argument("--adapted-suffix", dest="adapted_suffix", type=str, default=".adapted.json", help="Preferred suffix for adapted content_list in directory mode")

    args = parser.parse_args()

    if args.in_file:
        out = enhance_single(args.in_file, out_path=args.out)
        print(out)
    else:
        succ, err = enhance_directory(args.in_dir, adapted_suffix=args.adapted_suffix)
        print(f"Enhance completed: {succ} success, {err} errors")
