import glob
import os
import argparse
from typing import Any, Dict, List, Optional, Tuple
from ..utils.utils_2 import load_json, detect_mode_and_twoup, print_heading_outline, assign_logical_pages_via_offset, dump_json
from .media_enhance import enhance_media_on_root
from .toc import find_toc_pages, consolidate_toc_v2
from ..utils.llm_clients import gpt_llm_call
from .headings import merge_adjacent_headings
from .caption_cleanup import cleanup_captions_with_llm
from ..utils.cache_utils import StepCache, make_cached_llm
from .heading_levels import assign_heading_levels_case1, assign_non_toc_levels_with_llm, assign_heading_levels_case2, assign_heading_levels_case3
from .slides_enhance import assign_levels_slides_on_root

def _wrap_root(items: List[dict], source_dir: str) -> Dict[str, Any]:
    doc_id = os.path.basename(source_dir) or "document"
    return {"type": "document", "doc_id": doc_id, "source_path": source_dir, "children": list(items)}

def _choose_content_list_path(dir_path: str, adapted_suffix: str) -> Optional[str]:
    """Only accept content_list.adapted.v2.json in the given doc folder.
    No fallbacks.
    """
    p = os.path.join(dir_path, "content_list.adapted.v2.json")
    return p if os.path.exists(p) else None

def _toc_has_page(h):
    if isinstance(h, dict) and isinstance(h.get('page'), int):
        return True
    for ch in (h.get('children') or []) if isinstance(h, dict) else []:
        if _toc_has_page(ch):
            return True
    return False

def _extract_toc(root: dict) -> Optional[dict]:
    for n in root.get("children", []):
        if isinstance(n, dict) and n.get("type") == "toc" and isinstance(n.get("headings"), list):
            return n
    return None

def enhance_single(
    in_file: str,
    *,
    out_path: Optional[str] = None,
    cache_dir_name: str = "_stats",
    force: bool = False,
    cache_only: bool = False,
) -> str:
    # load adapted content list
    items = load_json(in_file)
    if isinstance(items, dict) and "content_list" in items:
        items = items["content_list"]
    if not isinstance(items, list):
        raise ValueError("input is not a content_list array or enhanced object with content_list")
    src_dir = os.path.dirname(in_file)

    doc_id = os.path.basename(src_dir) or "document"
    cache_root = os.path.join(os.getcwd(), cache_dir_name, doc_id)
    cache = StepCache(cache_root)
    cached_llm = make_cached_llm(gpt_llm_call, cache.subdir("llm_cache"))

    cache.save("00_items", items, params={"in_file": os.path.basename(in_file)}, deps={}, code_refs=[load_json])

    # --- Step 1: mode + twoup ---
    def _compute_mode():
        slides = ['0e94b4197b10096b1f4c699701570fbf', '52b3137455e7ca4df65021a200aef724', 'amb-siteaudits-ds15-150204174043-conversion-gate01_95', 'asdaaburson-marstellerarabyouthsurvey2014-140407100615-phpapp01_95', 'avalaunchpresentationsthatkickasteriskv3copy-150318114804-conversion-gate01_95', 'b3m5kaeqm2w8n4bwcesw-140602121350-phpapp02_95', 'bariumswallowpresentation-090810084400-phpapp01_95', 'bigdatatrends-120723191058-phpapp02_95','c31e6580d0175ab3f9d99d1ff0bfa000','caltraincapacitymountainview1-150701205750-lva1-app6891_95', 'catvsdogdlpycon15se-150512122612-lva1-app6891_95', 'chapter8-geneticscompatibilitymode-141214140247-conversion-gate02_95','competitiveoutcomes-091006065143-phpapp01_95','csewt7zsecmmbzjufbyx-signature-24d91a254426c21c3079384270e1f138dc43a271cfe15d6d520d68205855b2a3-poli-150306115347-conversion-gate01_95', 'ddoseattle-150627210357-lva1-app6891_95', 'digitalmeasurementframework22feb2011v6novideo-110221233835-phpapp01_95', 'disciplined-agile-business-analysis-160218012713_95', 'dr-vorapptchapter1emissionsources-121120210508-phpapp02_95', 'earlybird-110722143746-phpapp02_95', 'earthlinkweb-150213112111-conversion-gate02_95','ecommerceopportunityindia-141124010546-conversion-gate01_95','efis-140411041451-phpapp01_95','finalmediafindingspdf-141228031149-conversion-gate02_95','finalpresentationdeck-whatwhyhowofcertificationsocial-160324220748_95','formwork-150318073913-conversion-gate01_95','germanwingsdigitalcrisisanalysis-150403064828-conversion-gate01_95', 'indonesiamobilemarketresearch-ag-150106055934-conversion-gate02_95', 'measuringsuccessonfacebooktwitterlinkedin-160317142140_95','nielsen2015musicbizpresentation-final-150526143534-lva1-app6891_95','q1-2023-bilibili-inc-investor-presentation','reportq32015-151009093138-lva1-app6891_95']
        two_up = ['2024.ug.eprospectus','Bergen-Brochure-en-2022-23','GPL-Graduate-Studies-Professional-Learning-Brochure-Jul-2021', 'NUS-FASS-Graduate-Guidebook-2021-small']
        if doc_id in slides:
            return "slides", 1, 1.0
        elif doc_id in two_up:
            return "doc", 2, 1.0
        else:
            return "doc", 1, 1.0
        pages = sorted({int(it.get("page_idx")) for it in items if isinstance(it.get("page_idx"), int)})
        # mode, cols, conf = detect_mode_and_twoup(src_dir, pages)
        # print(mode, cols, conf)
        # try:
        #     ensure_all_page_images_for_pages(src_dir, pages, show_progress=False)
        # except Exception:
        #     pass
        return detect_mode_and_twoup(src_dir, pages) # return mode, cols, conf
    mode_cols_conf, _, _ = cache.load_or_compute(
        "01_mode_twoup",
        params={},
        deps_step_names=["00_items"],
        compute_fn=_compute_mode,
        code_refs=[detect_mode_and_twoup],
        force=force,
        cache_only=cache_only,
    )
    if not (isinstance(mode_cols_conf, (list, tuple)) and len(mode_cols_conf) == 3):
        mode, cols, conf = ("doc", 1, 0.0)
    else:
        mode, cols, conf = mode_cols_conf
    
    

    root = _wrap_root(items, src_dir)
    # attach mode/columns meta on root for downstream use
    meta: Dict[str, Any] = {"mode": mode, "mode_confidence": conf, "columns": cols}
    root["mode"] = meta["mode"]
    root["mode_confidence"] = meta["mode_confidence"]
    root["columns"] = meta["columns"]

    if mode == "doc":
        # --- Step 2: TOC detection + consolidation ---
        def _compute_toc():
            cand = find_toc_pages(root, gpt_llm_call, max_scan_pages=15, min_confidence=0.8, images_only=False)
            print(cand)
            if cand:
                # root = consolidate_toc_v2(root, cand, gpt_llm_call)
                return consolidate_toc_v2(root, cand, cached_llm)
            return root
        root, _, _ = cache.load_or_compute(
            "02_toc",
            params={"min_confidence": 0.8, "max_scan_pages": 15, "images_only": False},
            deps_step_names=["01_mode_twoup"],
            compute_fn=_compute_toc,
            code_refs=[find_toc_pages, consolidate_toc_v2],
            force=force,
            cache_only=cache_only,
        )
        
        # --- Step 3: logical pages ---
        root, _, _ = cache.load_or_compute(
            "03_logical_pages",
            params={},
            deps_step_names=["02_toc"],
            compute_fn=lambda: assign_logical_pages_via_offset(root),
            code_refs=[assign_logical_pages_via_offset],
            force=force,
            cache_only=cache_only,
        )
        # try:
        #     print('start add logical pages')
        #     # root = assign_logical_pages_via_offset(root)
        # except Exception:
        #     pass

        # --- Step 4: media enhance ---
        def _compute_media():
            r, meta_media = enhance_media_on_root(
                root,
                model="gpt-4o",
                area_px_threshold=20000,
                override=False,
                llm_call=cached_llm,
                image_cache_dir=cache.subdir("vlm_images"),
                parallel=True,
                workers=max(4, (os.cpu_count() or 4) * 2),
                llm_max_concurrency=3,
            )
            return {"root": r, "media_meta": meta_media}
        media_out, _, _ = cache.load_or_compute(
            "04_media",
            params={"model": "gpt-4o", "area_px_threshold": 20000},
            deps_step_names=["03_logical_pages"],
            compute_fn=_compute_media,
            code_refs=[enhance_media_on_root],
            force=force,
            cache_only=cache_only,
        )
        root = media_out["root"]

    #     # step 4: 图表 标题 脚注
    #     try:
    #         print('start image table enhance')
    #         root, _media_meta = enhance_media_on_root(root, model="gpt-4o", area_px_threshold=50000, override=False)
    #         # print(_media_meta)
    #     except Exception:
    #         pass

    #     # step 5: 合并截断的标题 Merge adjacent headings (heuristics + LLM confirmation)
    #     try:
    #         root = merge_adjacent_headings(root, use_llm=True, llm_model="gpt-4o")
    #     except Exception:
    #         pass

        # --- Step 5: merge headings ---
        root, _, _ = cache.load_or_compute(
            "05_merge_headings",
            params={"use_llm": True, "llm_model": "gpt-4o"},
            deps_step_names=["04_media"],
            compute_fn=lambda: merge_adjacent_headings(root, use_llm=True, llm_model="gpt-4o"),
            code_refs=[merge_adjacent_headings],
            force=force,
            cache_only=cache_only,
        )

        # --- Step 6: caption cleanup ---
        def _compute_cleanup():
            r, stats = cleanup_captions_with_llm(
                root,
                llm_model="gpt-4o",
                window_before=4,
                window_after=4,
                writeback_media_fields=True,
                require_confidence=True,
                min_confidence=0.85,
                # Disable geometry gating completely; rely on text similarity only
                geo_guard=False,
                min_text_sim_to_downgrade=0.5,
            )
            return {"root": r, "stats": stats}
        cap_out, _, _ = cache.load_or_compute(
            "06_caption_cleanup",
            params={"min_confidence": 0.85},
            deps_step_names=["05_merge_headings"],
            compute_fn=_compute_cleanup,
            code_refs=[cleanup_captions_with_llm],
            force=force,
            cache_only=cache_only,
        )
        root = cap_out["root"]

    #     # step 6: 图像标题消岐 Disambiguation: caption-as-title cleanup via LLM
    #     try:
    #         print('start caption/title disambiguation via LLM')
    #         root, _cap_stats = cleanup_captions_with_llm(
    #             root,
    #             llm_model="gpt-4o",
    #             window_before=4,
    #             window_after=4,
    #             writeback_media_fields=True,
    #             require_confidence=True,
    #             min_confidence=0.85,
    #         )
    #         print(_cap_stats)
    #     except Exception:
    #         pass

    #     # 分配标题
        # toc_node = _extract_toc(root)
        # print(toc_node)
        # if not toc_node:
        #     return 'case 3'
        # elif any(_toc_has_page(h) for h in toc_node.get("headings", [])):
        #     return 'case 1'
        # else:
        #     return 'case 2'

        # print_heading_outline(root, by_logical_page=True, include_meta=True, max_len=80)
        # Step 7: 主 heading assignment (Case1/2/3)
        def _compute_heading_assign():
            toc_node = _extract_toc(root)
            if not toc_node:
                r, stats = assign_heading_levels_case3(root, llm_model="gpt-4o", max_level=4, promote_media=False)
                return r
            elif any(_toc_has_page(h) for h in toc_node.get("headings", [])):
                # Case1: TOC + page numbers
                root1, stats1 = assign_heading_levels_case1(root, high_threshold=0.85, mid_threshold=0.72, max_level=4)
                # 直接在 Case1 内补跑 Step8
                root2, stats2 = assign_non_toc_levels_with_llm(root1, llm_model="gpt-4o", max_level=4, chunk_size=40)
                return root2
            else:
                r, stats = assign_heading_levels_case2(root, window_size=20, llm_model="gpt-4o")
                return r

        root, _, _ = cache.load_or_compute(
            "07_heading_levels",
            params={},
            deps_step_names=["06_caption_cleanup"],
            compute_fn=_compute_heading_assign,
            code_refs=[assign_heading_levels_case1, assign_heading_levels_case2, assign_heading_levels_case3],
            force=force,
            cache_only=cache_only,
        )
        print("children types:", [ch.get("type") for ch in root.get("children", [])][:20])
        print(doc_id)
        print_heading_outline(root, by_logical_page=True, include_meta=True, max_len=80)

    elif mode == "slides":
        # Slides pipeline: per-page slide container with VLM title and page content as children
        def _compute_slides():
            return assign_levels_slides_on_root(root, llm_model="gpt-4o", llm_call=cached_llm)
        root, _, _ = cache.load_or_compute(
            "02_slides_levels",
            params={"llm_model": "gpt-4o"},
            deps_step_names=["01_mode_twoup"],
            compute_fn=_compute_slides,
            code_refs=[assign_levels_slides_on_root],
            force=force,
            cache_only=cache_only,
        )
        # Enhance media for slides with conservative filters to avoid background/master images
        def _compute_slides_media():
            r, stats = enhance_media_on_root(
                root,
                model="gpt-4o",
                area_px_threshold=40_000,
                override=False,
                llm_call=cached_llm,
                recursive=True,
                min_area_ratio=0.06,
                max_area_ratio=0.85,
                background_edge_margin=8,
                via_tag="slides_media_vlm",
                image_cache_dir=cache.subdir("vlm_images"),
                parallel=True,
                workers=max(4, (os.cpu_count() or 4) * 2),
                llm_max_concurrency=3,
            )
            return {"root": r, "stats": stats}
        media_out, _, _ = cache.load_or_compute(
            "03_slides_media",
            params={
                "model": "gpt-4o",
                "area_px_threshold": 40000,
                "min_area_ratio": 0.06,
                "max_area_ratio": 0.85,
            },
            deps_step_names=["02_slides_levels"],
            compute_fn=_compute_slides_media,
            code_refs=[enhance_media_on_root],
            force=force,
            cache_only=cache_only,
        )
        root = media_out["root"]
        # Assign logical_page sequentially (1..N) by physical page order for slides mode
        try:
            pages = sorted({int(n.get("page_idx")) for n in root.get("children", []) if isinstance(n, dict) and isinstance(n.get("page_idx"), int)})
            page_to_logical = {p: i + 1 for i, p in enumerate(pages)}
            def _apply_lp(node: Any):
                if isinstance(node, dict):
                    p = node.get("page_idx")
                    if isinstance(p, int) and p in page_to_logical:
                        node["logical_page"] = page_to_logical[p]
                    ch = node.get("children")
                    if isinstance(ch, list):
                        for c in ch:
                            _apply_lp(c)
                elif isinstance(node, list):
                    for c in node:
                        _apply_lp(c)
            _apply_lp(root)
        except Exception:
            pass
    # print_heading_outline(root)
    # write final enhanced result
    if out_path is None:
        out_path = os.path.join(src_dir, "content_list.enhanced.json")
    dump_json(root, out_path)
    return out_path

def enhance_directory(
    in_dir: str,
    *,
    adapted_suffix: str = ".adapted.json",
    force: bool = False,
    cache_only: bool = False,
) -> Tuple[int, int]:
    """Walk `in_dir` (following symlinks) to find document folders that contain
    any suitable `content_list*.json`, then enhance each one.
    """
    doc_dirs: set[str] = set()
    for root, dirs, files in os.walk(in_dir, followlinks=True):
        names = set(files)
        if "content_list.adapted.v2.json" in names:
            doc_dirs.add(root)

    succ = 0
    err = 0
    for d in sorted(doc_dirs):
        src = _choose_content_list_path(d, adapted_suffix)
        if not src:
            continue
        try:
            enhance_single(
                src,
                out_path=os.path.join(d, "content_list.enhanced.json"),
                force=force,
                cache_only=cache_only,
            )
            succ += 1
        except Exception:
            err += 1
            continue
    return succ, err

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance adapted content_list into a single enhanced JSON (meta + content_list)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-file", dest="in_file", type=str, help="Adapted content_list(.json/.adapted.json) file")
    group.add_argument("--in-dir", dest="in_dir", type=str, help="Root directory containing adapted content_lists")

    parser.add_argument("--out", dest="out", type=str, default=None, help="Output enhanced JSON when using --in-file (default: content_list.enhanced.json beside input)")
    parser.add_argument("--adapted-suffix", dest="adapted_suffix", type=str, default=".adapted.json", help="Preferred suffix for adapted content_list in directory mode (ignored; only adapted.v2 used)")
    parser.add_argument("--force", dest="force", action="store_true", help="Force recompute cached steps (e.g., 04_media)")
    parser.add_argument("--cache-only", dest="cache_only", action="store_true", help="Fail if cache miss; do not compute")

    args = parser.parse_args()

    if args.in_file:
        out = enhance_single(args.in_file, out_path=args.out, force=args.force, cache_only=args.cache_only)
        print(out)
    else:
        succ, err = enhance_directory(
            args.in_dir,
            adapted_suffix=args.adapted_suffix,
            force=args.force,
            cache_only=args.cache_only,
        )
        print(f"Enhance completed: {succ} success, {err} errors")
