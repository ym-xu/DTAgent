import os
import json
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures as _fut
import threading as _th

from PIL import Image, ImageDraw  # type: ignore
import hashlib, io

from ..utils.utils_2 import find_pdf_in_dir
from ..utils.llm_clients import gpt_llm_call, qwen_llm_call


def _bbox_area_pixels(b: Tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = b
    return max(0, x1 - x0) * max(0, y1 - y0)

def _render_and_annotate_from_pdf(
    source_dir: str,
    page_idx: int,
    bbox_pdf: Tuple[float, float, float, float],
    *,
    dpi: int = 150,
) -> Optional[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """
    Render the PDF page at a fixed DPI and draw a red rectangle at the correct location.
    BBox is in PDF coordinates (points). We transform it using the same matrix as rendering.
    Returns (annotated PIL image, pixel_bbox) or None on failure.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        return None

    pdf_path = find_pdf_in_dir(source_dir)
    if not pdf_path:
        return None

    try:
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= len(doc):
            return None
        page = doc.load_page(page_idx)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        # transform bbox from PDF space to pixel space
        rect_pdf = fitz.Rect(float(bbox_pdf[0]), float(bbox_pdf[1]), float(bbox_pdf[2]), float(bbox_pdf[3]))
        rect_px = rect_pdf * mat
        pix = page.get_pixmap(matrix=mat, alpha=False)
        w, h = pix.width, pix.height

        # clamp rectangle to image bounds
        x0 = max(0, min(int(rect_px.x0), w - 1))
        y0 = max(0, min(int(rect_px.y0), h - 1))
        x1 = max(0, min(int(rect_px.x1), w - 1))
        y1 = max(0, min(int(rect_px.y1), h - 1))
        if x1 <= x0:
            x1 = min(w - 1, x0 + 1)
        if y1 <= y0:
            y1 = min(h - 1, y0 + 1)

        # convert pixmap to PIL image and draw rectangle
        im = Image.frombytes("RGB", (w, h), pix.samples)
        draw = ImageDraw.Draw(im)
        for off in range(0, 6):
            draw.rectangle((x0 - off, y0 - off, x1 + off, y1 + off), outline=(255, 0, 0))
        # im.save(f'{page_idx}.jpg')
        return im, (x0, y0, x1, y1)
    except Exception:
        return None


def _call_vlm_json(model: str, messages: List[Dict[str, Any]], image_path: str, *, llm_call=None) -> Optional[Dict[str, Any]]:
    # print('start')
    try:
        # print(model.lower())
        if llm_call is not None:
            raw = llm_call(messages, images=[image_path], model=model, json_mode=True)
        elif model.lower().startswith("qwen"):
            raw = qwen_llm_call(messages, images=[image_path], model=model, json_mode=True)
        else:
            raw = gpt_llm_call(messages, images=[image_path], model=model, json_mode=True)
        # print('raw: ',raw)
        return json.loads(raw)
    except Exception:
        return None

_STRIP_PREFIXES = [
    "the image inside the red rectangle ",
    "the image in the red rectangle ",
    "the image inside the rectangle ",
    "the image in the rectangle ",
    "inside the red rectangle, ",
    "within the red rectangle, ",
    "within the red box, ",
    "in the red box, ",
    "the red rectangle shows ",
    "the red rectangle highlights ",
    "the highlighted area ",
    "the highlighted region ",
    "the highlighted section ",
    "in the highlighted area, ",
    "this image shows ",
    "this image illustrates ",
    "this figure shows ",
    "this figure illustrates ",
    "the image shows ",
    "the figure shows ",
]

def _clean_description_text(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return s
    txt = s.strip()
    low = txt.lower()
    for pref in _STRIP_PREFIXES:
        if low.startswith(pref):
            txt = txt[len(pref):].lstrip()
            low = txt.lower()
            break
    for phrase in [
        "inside the red rectangle",
        "within the red rectangle",
        "in the red rectangle",
        "in the red box",
        "highlighted area",
        "highlighted region",
        "red rectangle",
    ]:
        if phrase in txt.lower():
            txt = txt.replace(phrase, "").replace("  ", " ")
    return txt.strip()


# def _image_prompt_text(bbox: Tuple[int, int, int, int], page_idx: Optional[int]) -> str:
#     return (
#         "You are given a page image with a red rectangle indicating the target figure/image region.\n"
#         "Tasks:\n"
#         "1) Provide a concise description (2–3 sentences) of the visual content in that region.\n"
#         "2) If the image has a caption, subcaption, and/or footnotes on this page, identify them by looking near the rectangle (typically above/below or adjacent).\n"
#         "   Only extract text that clearly belongs to this image. If uncertain or none, leave them empty.\n\n"
#         "Style constraints (VERY IMPORTANT):\n"
#         "- Do NOT mention the red rectangle/box or bounding box in your wording.\n"
#         "- Describe the content directly, as if already cropped to that region.\n"
#         "- Avoid boilerplate like 'this image/figure shows/illustrates'.\n\n"
#         "Important disambiguation rules:\n"
#         "- Do NOT take section headings, subheadings, or running headers/footers as image captions.\n"
#         "- A caption usually directly describes the figure/table/chart, or starts with 'Figure', 'Table', etc.\n"
#         "- A footnote usually includes data sources, notes, or explanations tied to the figure.\n"
#         "- If nearby text is just part of the main body (not describing the figure), do not treat it as caption/footnote.\n\n"
#         "Heuristic prior (very important):\n"
#         "- If the image looks like a statistical chart, graph, scientific figure, or table, then it is very likely to have a true caption and/or footnotes. Be proactive in checking adjacent text.\n"
#         "- If the image looks like a product manual illustration, icon, device diagram, or decorative pictogram, then it usually does NOT have a caption. In such cases, do NOT treat nearby section headings or body labels as captions; leave caption empty.\n\n"
#         "Output strict JSON:\n"
#         "{\n"
#         "  \"image_caption\": [string],\n"
#         "  \"image_subcaption\": string|null,\n"
#         "  \"image_footnote\": [string],\n"
#         "  \"description\": string|null\n"
#         "}\n"
#         f"bbox: {bbox}; page_idx: {page_idx if page_idx is not None else 'unknown'}\n"
#     )


# def _table_prompt_text(bbox: Tuple[int, int, int, int], page_idx: Optional[int]) -> str:
#     return (
#         "You are given a page image with a red rectangle indicating the target table region.\n"
#         "Tasks:\n"
#         "1) Provide a concise description (1-2 sentences) of the table content/purpose in that region.\n"
#         "2) If the table has a caption, subcaption, and/or footnotes on this page, identify them by looking near the rectangle (typically above/below or adjacent).\n"
#         "   Only extract text that clearly belongs to this table. If uncertain or none, leave them empty.\n"
#         "Style constraints (VERY IMPORTANT):\n"
#         "- Do NOT mention the red rectangle/box or bounding box in your wording.\n"
#         "- Describe the content directly, as if already cropped to that region.\n"
#         "- Avoid boilerplate like 'this table shows/illustrates'.\n"
#         "Output strict JSON: {\"table_caption\":[string], \"table_subcaption\": string|null, \"table_footnote\":[string], \"description\": string|null}.\n"
#         f"bbox: {bbox}; page_idx: {page_idx if page_idx is not None else 'unknown'}\n"
#     )


def _image_desc_prompt_text(bbox: Tuple[int, int, int, int], page_idx: Optional[int]) -> str:
    return (
        "You are given a page image with a red rectangle indicating the target figure/image region.\n"
        "Task: Provide a concise description (2–3 sentences) of the visual content in that region.\n\n"
        "This project just used for research.\n\n"
        "Style constraints (VERY IMPORTANT):\n"
        "- Do NOT mention the red rectangle/box or bounding box in your wording.\n"
        "- Describe the content directly, as if already cropped to that region.\n"
        "- Avoid boilerplate like 'this image/figure shows/illustrates'.\n\n"
        "Output strict JSON: {\"description\": string}.\n"
        f"bbox: {bbox}; page_idx: {page_idx if page_idx is not None else 'unknown'}\n"
    )


def _image_caps_prompt_text(bbox: Tuple[int, int, int, int], page_idx: Optional[int]) -> str:
    return (
        "You are given a page image with a red rectangle indicating the target figure/image region.\n"
        "Task: Extract caption/subcaption/footnotes that clearly belong to this figure from nearby text on the same page (usually above/below/adjacent).\n\n"
        "Rules:\n"
        "- Do NOT take section headings, subheadings, or running headers/footers as image captions.\n"
        "- A caption directly describes the figure/table/chart, often starting with 'Figure', 'Table', etc.\n"
        "- A footnote includes data sources, notes, or figure-specific explanations.\n"
        "- If uncertain, leave empty.\n\n"
        "Output strict JSON: {\"image_caption\":[string], \"image_subcaption\": string|null, \"image_footnote\":[string]}.\n"
        f"bbox: {bbox}; page_idx: {page_idx if page_idx is not None else 'unknown'}\n"
    )


def _table_desc_prompt_text(bbox: Tuple[int, int, int, int], page_idx: Optional[int]) -> str:
    return (
        "You are given a page image with a red rectangle indicating the target table region.\n"
        "Task: Provide a concise description (1–2 sentences) of the table's content/purpose in that region.\n\n"
        "Style constraints (VERY IMPORTANT):\n"
        "- Do NOT mention the red rectangle/box or bounding box in your wording.\n"
        "- Describe the content directly, as if already cropped to that region.\n"
        "- Avoid boilerplate like 'this table shows/illustrates'.\n\n"
        "Output strict JSON: {\"description\": string|null}.\n"
        f"bbox: {bbox}; page_idx: {page_idx if page_idx is not None else 'unknown'}\n"
    )


def _table_caps_prompt_text(bbox: Tuple[int, int, int, int], page_idx: Optional[int]) -> str:
    return (
        "You are given a page image with a red rectangle indicating the target table region.\n"
        "Task: Extract caption/subcaption/footnotes that clearly belong to this table from nearby text on the same page (usually above/below/adjacent).\n\n"
        "Rules:\n"
        "- Do NOT take section headings, subheadings, or running headers/footers as table captions.\n"
        "- A caption directly describes the table; a footnote includes data sources or notes.\n"
        "- If uncertain, leave empty.\n\n"
        "Output strict JSON: {\"table_caption\":[string], \"table_subcaption\": string|null, \"table_footnote\":[string]}.\n"
        f"bbox: {bbox}; page_idx: {page_idx if page_idx is not None else 'unknown'}\n"
    )


def _normalize_list_str(values: Any) -> List[str]:
    out: List[str] = []
    if isinstance(values, list):
        for v in values:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
    return list(dict.fromkeys(out))


def _apply_image_fields(node: Dict[str, Any], obj: Dict[str, Any], override: bool) -> None:
    cap = _normalize_list_str(obj.get("image_caption"))
    sub = obj.get("image_subcaption") if isinstance(obj.get("image_subcaption"), str) else None
    foot = _normalize_list_str(obj.get("image_footnote"))
    desc = obj.get("description") if isinstance(obj.get("description"), str) else None
    desc = _clean_description_text(desc)

    if override or not node.get("image_caption"):
        if cap:
            node["image_caption"] = cap
    if override or not node.get("image_subcaption"):
        if isinstance(sub, str):
            node["image_subcaption"] = sub
    if override or not node.get("image_footnote"):
        if foot:
            node["image_footnote"] = foot
    if override or not node.get("description"):
        if isinstance(desc, str):
            node["description"] = desc


def _apply_table_fields(node: Dict[str, Any], obj: Dict[str, Any], override: bool) -> None:
    cap = _normalize_list_str(obj.get("table_caption"))
    sub = obj.get("table_subcaption") if isinstance(obj.get("table_subcaption"), str) else None
    foot = _normalize_list_str(obj.get("table_footnote"))
    desc = obj.get("description") if isinstance(obj.get("description"), str) else None
    desc = _clean_description_text(desc)

    if override or not node.get("table_caption"):
        if cap:
            node["table_caption"] = cap
    if override or not node.get("table_subcaption"):
        if isinstance(sub, str):
            node["table_subcaption"] = sub
    if override or not node.get("table_footnote"):
        if foot:
            node["table_footnote"] = foot
    if override or not node.get("description"):
        if isinstance(desc, str):
            node["description"] = desc


def _save_image_deterministic(im: Image.Image, cache_dir: Optional[str]) -> Optional[str]:
    try:
        if cache_dir is None:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data = buf.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        subdir = os.path.join(cache_dir, sha[:2], sha[2:4])
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, f"{sha}.png")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)
        return path
    except Exception:
        return None


def enhance_media_on_root(
    root: Dict[str, Any],
    *,
    model: str = "qwen-vl-max",
    area_px_threshold: int = 150_000,
    desc_min_area_px: Optional[int] = None,
    caps_min_area_px: Optional[int] = None,
    override: bool = False,
    llm_call=None,
    recursive: bool = False,
    min_area_ratio: Optional[float] = None,
    max_area_ratio: Optional[float] = None,
    background_edge_margin: int = 8,
    via_tag: Optional[str] = None,
    do_desc: bool = True,
    do_caps: bool = True,
    prompt_version_desc: str = "v1",
    prompt_version_caps: str = "v1",
    image_cache_dir: Optional[str] = None,
    # parallelization
    parallel: bool = False,
    workers: Optional[int] = None,
    llm_max_concurrency: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Enhance image/table nodes that have outline bbox with area >= area_px_threshold.
    Draw a red rectangle on the page image and ask VLM to extract captions/footnotes/subcaption and description.
    Returns updated root and a simplified stats dict (overall counts only).
    """
    children: List[Dict[str, Any]] = list(root.get("children", []))
    source_dir = root.get("source_path") if isinstance(root.get("source_path"), str) else None
    stats = {
        "enhanced": 0,
        "skipped": 0,
        "failed": 0,
        "model": model,
        "area_px_threshold": area_px_threshold,
    }

    def _should_skip_by_ratio(pix_bbox: Tuple[int, int, int, int], im: Image.Image) -> bool:
        try:
            x0, y0, x1, y1 = pix_bbox
            w, h = im.size
            area = max(1, w * h)
            area_px = _bbox_area_pixels(pix_bbox)
            ratio = area_px / float(area)
            # tiny icons
            if min_area_ratio is not None and ratio < float(min_area_ratio):
                return True
            # full-bleed/background
            if max_area_ratio is not None and ratio > float(max_area_ratio):
                # also require touching at least 3 edges to be safe
                margin = max(0, int(background_edge_margin))
                touches = 0
                if x0 <= margin:
                    touches += 1
                if y0 <= margin:
                    touches += 1
                if x1 >= w - 1 - margin:
                    touches += 1
                if y1 >= h - 1 - margin:
                    touches += 1
                if touches >= 3:
                    return True
            return False
        except Exception:
            return False

    # Optional concurrency guard for LLM calls
    llm_sem: Optional[_th.Semaphore] = None
    if llm_max_concurrency and llm_max_concurrency > 0:
        llm_sem = _th.Semaphore(int(llm_max_concurrency))

    def _llm_call_guard(messages, images, model, json_mode=True):
        if llm_call is None:
            # fall back to default selection in _call_vlm_json
            return gpt_llm_call(messages, images=images, model=model, json_mode=json_mode)
        if llm_sem is None:
            return llm_call(messages, images=images, model=model, json_mode=json_mode)
        with llm_sem:
            return llm_call(messages, images=images, model=model, json_mode=json_mode)

    lock = _th.Lock()

    def _inc_stat(key: str) -> None:
        with lock:
            stats[key] = int(stats.get(key, 0)) + 1

    def _process_node(node: Dict[str, Any]) -> None:
        typ = node.get("type")
        if recursive and isinstance(node.get("children"), list):
            for ch in node["children"]:
                if isinstance(ch, dict):
                    _process_node(ch)
        if typ not in ("image", "table"):
            return
        bbox = node.get("outline")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            stats["skipped"] += 1
            return
        page_idx = node.get("page_idx") if isinstance(node.get("page_idx"), int) else None
        if source_dir is None or page_idx is None:
            stats["skipped"] += 1
            return
        # Render page and annotate
        node_idx = node.get("node_idx") if isinstance(node.get("node_idx"), int) else None
        typ_tag = typ if isinstance(typ, str) else "media"
        ann = _render_and_annotate_from_pdf(
            source_dir,
            page_idx,
            (bbox[0], bbox[1], bbox[2], bbox[3]),
            dpi=150,
        )
        if not ann:
            _inc_stat("skipped")
            return
        annotated_im, pix_bbox = ann
        area_px = _bbox_area_pixels(pix_bbox)
        desc_thr = desc_min_area_px if isinstance(desc_min_area_px, int) else area_px_threshold
        caps_thr = caps_min_area_px if isinstance(caps_min_area_px, int) else area_px_threshold
        # Ratio/background filtering
        if _should_skip_by_ratio(pix_bbox, annotated_im):
            _inc_stat("skipped")
            return
        # Debug
        media_name = node.get("img_path") or "(no-img-path)"
        bw = max(0, int(pix_bbox[2]) - int(pix_bbox[0]))
        bh = max(0, int(pix_bbox[3]) - int(pix_bbox[1]))
        # try:
        #     print(f"[media] page={page_idx} type={typ_tag} node_idx={node_idx} name={media_name} size_px={bw}x{bh}")
        # except Exception:
        #     pass
        # Prepare a deterministic cached image path
        stable_path = _save_image_deterministic(annotated_im, image_cache_dir)
        temp_created = False
        if not stable_path:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                stable_path = tmp.name
            try:
                annotated_im.save(stable_path)
                temp_created = True
            except Exception:
                pass

        # Description branch
        try:
            existing_desc = node.get("description")
            has_desc = isinstance(existing_desc, str) and existing_desc.strip() != ""
            need_desc = do_desc and area_px >= desc_thr and (override or not has_desc)
            if need_desc:
                if typ == "image":
                    user_text = _image_desc_prompt_text(pix_bbox, page_idx)
                else:
                    user_text = _table_desc_prompt_text(pix_bbox, page_idx)
                messages = [
                    {"role": "system", "content": f"You only output JSON. Task=desc; prompt_v={prompt_version_desc}"},
                    {"role": "user", "content": user_text},
                ]
                # call LLM with optional concurrency guard
                obj = _call_vlm_json(model, messages, stable_path, llm_call=lambda m, images, model, json_mode=True: _llm_call_guard(m, images, model, json_mode=json_mode))
                # print(obj)
                if isinstance(obj, dict):
                    if typ == "image":
                        _apply_image_fields(node, {"description": obj.get("description")}, override)
                    else:
                        _apply_table_fields(node, {"description": obj.get("description")}, override)
                    meta = dict(node.get("heading_meta") or {})
                    meta["via"] = via_tag or meta.get("via") or "media_vlm_desc"
                    node["heading_meta"] = meta
                    _inc_stat("enhanced")
        except Exception:
            _inc_stat("failed")

        # Captions branch
        try:
            has_caps = (node.get("image_caption") if typ == "image" else node.get("table_caption"))
            need_caps = do_caps and area_px >= caps_thr and (override or not has_caps)
            if need_caps:
                if typ == "image":
                    user_text = _image_caps_prompt_text(pix_bbox, page_idx)
                else:
                    user_text = _table_caps_prompt_text(pix_bbox, page_idx)
                messages = [
                    {"role": "system", "content": f"You only output JSON. Task=caps; prompt_v={prompt_version_caps}"},
                    {"role": "user", "content": user_text},
                ]
                obj = _call_vlm_json(model, messages, stable_path, llm_call=lambda m, images, model, json_mode=True: _llm_call_guard(m, images, model, json_mode=json_mode))
                if isinstance(obj, dict):
                    if typ == "image":
                        _apply_image_fields(node, obj, override)
                    else:
                        _apply_table_fields(node, obj, override)
                    meta = dict(node.get("heading_meta") or {})
                    meta["via"] = via_tag or meta.get("via") or "media_vlm_caps"
                    node["heading_meta"] = meta
                    _inc_stat("enhanced")
        except Exception:
            _inc_stat("failed")

        # Cleanup temp image if used
        if temp_created and stable_path and os.path.exists(stable_path):
            try:
                os.unlink(stable_path)
            except Exception:
                pass

    def _collect_targets(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            typ = n.get("type")
            if typ in ("image", "table"):
                targets.append(n)
            if recursive and isinstance(n.get("children"), list):
                targets.extend(_collect_targets([c for c in n.get("children") if isinstance(c, dict)]))
        return targets

    if parallel:
        pool_size = int(workers) if workers else max(4, (os.cpu_count() or 4) * 2)
        targets = _collect_targets(children)
        with _fut.ThreadPoolExecutor(max_workers=pool_size) as ex:
            futs = [ex.submit(_process_node, n) for n in targets]
            for f in _fut.as_completed(futs):
                try:
                    _ = f.result()
                except Exception:
                    _inc_stat("failed")
    else:
        # Walk all nodes serially (original behavior)
        for node in children:
            if isinstance(node, dict):
                _process_node(node)

    root["children"] = children
    return root, stats
