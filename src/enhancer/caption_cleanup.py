import json
from typing import Any, Dict, List, Optional, Tuple
import re

def _bbox_from_node(n: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    b = n.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        try:
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except Exception:
            return None
    # try outline for media
    o = n.get("outline")
    if isinstance(o, (list, tuple)) and len(o) >= 4:
        try:
            return (float(o[0]), float(o[1]), float(o[2]), float(o[3]))
        except Exception:
            return None
    return None

def _overlap_ratio_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    inter = max(0.0, hi - lo)
    denom = max(1e-6, min(abs(a1 - a0), abs(b1 - b0)))
    return inter / denom

def _vertical_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    # positive gap if disjoint; 0 if overlap
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0

def _is_same_column(media: Dict[str, Any], text: Dict[str, Any], *, min_overlap: float = 0.5, max_center_delta_ratio: float = 0.45) -> bool:
    mb = _bbox_from_node(media)
    tb = _bbox_from_node(text)
    if mb is None or tb is None:
        return True
    mx0, my0, mx1, my1 = mb
    tx0, ty0, tx1, ty1 = tb
    # Horizontal overlap fraction relative to narrower box
    h_ov = _overlap_ratio_1d(mx0, mx1, tx0, tx1)
    if h_ov >= float(min_overlap):
        return True
    # If centers are close relative to widths, also treat as same column
    mc = (mx0 + mx1) / 2.0
    tc = (tx0 + tx1) / 2.0
    mw = max(1.0, abs(mx1 - mx0))
    tw = max(1.0, abs(tx1 - tx0))
    return abs(mc - tc) <= max(mw, tw) * float(max_center_delta_ratio)


def _is_absurd_geometry(media: Dict[str, Any], text: Dict[str, Any]) -> bool:
    """Return True only when geometry strongly suggests they are unrelated.
    Heuristics:
    - Almost no horizontal overlap AND centers far apart (distinct columns)
    - OR text far wider than media (e.g., > 3x) AND almost no overlap
    Vertical distance is not considered if columns look aligned.
    """
    mb = _bbox_from_node(media)
    tb = _bbox_from_node(text)
    if mb is None or tb is None:
        return False
    mx0, my0, mx1, my1 = mb
    tx0, ty0, tx1, ty1 = tb
    h_ov = _overlap_ratio_1d(mx0, mx1, tx0, tx1)
    if h_ov >= 0.2:
        return False
    mc = (mx0 + mx1) / 2.0
    tc = (tx0 + tx1) / 2.0
    mw = max(1.0, abs(mx1 - mx0))
    tw = max(1.0, abs(tx1 - tx0))
    if abs(mc - tc) > (mw + tw) * 0.6:
        return True
    if h_ov < 0.05 and tw > mw * 3.0:
        return True
    return False

def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    # remove punctuation and collapse spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tok(s: str) -> list:
    s = _norm_text(s)
    return [t for t in s.split(" ") if t]

def _jaccard_tokens(a: str, b: str) -> float:
    ta = set(_tok(a))
    tb = set(_tok(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return inter / union

def _max_caption_similarity(media: Dict[str, Any], text: str, *, merged_text: Optional[Dict[str, Any]] = None) -> float:
    """Max token-jaccard similarity between the text node and media strings.
    Priority order:
    1) Current LLM result's merged_text (caption/subcaption and their concatenation)
    2) Media fields on node (image_caption/subcaption/description or table_caption/description)
    """
    cands: List[str] = []
    # Priority 1: use merged_text from this LLM result if available
    if isinstance(merged_text, dict):
        cap = merged_text.get("caption")
        sub = merged_text.get("subcaption")
        if isinstance(cap, str) and cap.strip():
            cands.append(cap)
        if isinstance(sub, str) and sub.strip():
            cands.append(sub)
        if isinstance(cap, str) and isinstance(sub, str) and cap.strip() and sub.strip():
            cands.append(f"{cap.strip()} {sub.strip()}")
    # Priority 2: fallback to media's existing fields
    if media.get("type") == "image":
        caps = media.get("image_caption")
        if isinstance(caps, list):
            cands.extend([c for c in caps if isinstance(c, str)])
        sub2 = media.get("image_subcaption")
        if isinstance(sub2, str):
            cands.append(sub2)
        # Add combined main caption + subcaption candidate
        if isinstance(caps, list) and caps:
            try:
                main_cap = next((c for c in caps if isinstance(c, str) and c.strip()), None)
            except Exception:
                main_cap = None
            if isinstance(main_cap, str) and isinstance(sub2, str) and main_cap.strip() and sub2.strip():
                cands.append(f"{main_cap.strip()} {sub2.strip()}")
    elif media.get("type") == "table":
        caps = media.get("table_caption")
        if isinstance(caps, list):
            cands.extend([c for c in caps if isinstance(c, str)])
    desc = media.get("description")
    if isinstance(desc, str):
        cands.append(desc)
    if not cands:
        return 0.0
    return max((_jaccard_tokens(text, c) for c in cands), default=0.0)


def _is_media(node: Dict[str, Any]) -> bool:
    return node.get("type") in ("image", "table")


def _is_text(node: Dict[str, Any]) -> bool:
    return node.get("type") == "text" and isinstance(node.get("text"), str)


def _collect_page_neighbors(children: List[Dict[str, Any]], idx: int, *, before: int = 3, after: int = 3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For media at position idx, collect up to `before` text nodes before and `after` after
    on the same page, stopping if another media node is encountered.
    """
    page = children[idx].get("page_idx")
    # walk backwards
    prev: List[Dict[str, Any]] = []
    i = idx - 1
    while i >= 0 and len(prev) < before:
        n = children[i]
        if n.get("page_idx") != page:
            break
        if _is_media(n):
            break
        if _is_text(n):
            prev.append(n)
        i -= 1
    prev.reverse()

    # walk forwards
    nxt: List[Dict[str, Any]] = []
    j = idx + 1
    while j < len(children) and len(nxt) < after:
        n = children[j]
        if n.get("page_idx") != page:
            break
        if _is_media(n):
            break
        if _is_text(n):
            nxt.append(n)
        j += 1

    return prev, nxt


def _payload_for_llm(media: Dict[str, Any], before_nodes: List[Dict[str, Any]], after_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    typ = media.get("type")
    node_idx = media.get("node_idx")
    page_idx = media.get("page_idx")
    payload: Dict[str, Any] = {
        "media": {
            "type": typ,
            "node_idx": node_idx,
            "page_idx": page_idx,
            "captions": [],
            "subcaption": None,
            "footnotes": [],
        },
        "candidates": [],
    }
    if typ == "image":
        if isinstance(media.get("image_caption"), list):
            payload["media"]["captions"] = [str(x) for x in media.get("image_caption", []) if isinstance(x, str)]
        if isinstance(media.get("image_subcaption"), str):
            payload["media"]["subcaption"] = media.get("image_subcaption")
        if isinstance(media.get("image_footnote"), list):
            payload["media"]["footnotes"] = [str(x) for x in media.get("image_footnote", []) if isinstance(x, str)]
    else:
        if isinstance(media.get("table_caption"), list):
            payload["media"]["captions"] = [str(x) for x in media.get("table_caption", []) if isinstance(x, str)]
        if isinstance(media.get("table_footnote"), list):
            payload["media"]["footnotes"] = [str(x) for x in media.get("table_footnote", []) if isinstance(x, str)]

    def _pack(nodes: List[Dict[str, Any]], where: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for k, n in enumerate(nodes, start=1):
            out.append({
                "node_idx": n.get("node_idx"),
                "text": n.get("text", ""),
                "node_level": n.get("node_level", -1),
                "position": where,
                "distance": k,
            })
        return out

    payload["candidates"].extend(_pack(before_nodes, "before"))
    payload["candidates"].extend(_pack(after_nodes, "after"))
    return payload


def _build_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "role_assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "node_idx": {"type": "integer"},
                        "role": {"type": "string", "enum": ["caption", "subcaption", "footnote", "ignore"]},
                        "part_index": {"type": ["integer", "null"]},
                    },
                    "required": ["node_idx", "role"],
                },
            },
            "merged_text": {
                "type": "object",
                "properties": {
                    "caption": {"type": ["string", "null"]},
                    "subcaption": {"type": ["string", "null"]},
                    "footnotes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["caption", "subcaption", "footnotes"],
            },
            "downgrade_nodes": {"type": "array", "items": {"type": "integer"}},
            "confidence": {"type": ["number", "null"]},
        },
        "required": ["role_assignments", "merged_text", "downgrade_nodes"],
    }

    prompt = (
        "You are given one media element (image/table) and nearby text candidates on the same page.\n"
        "Goal: decide which candidates are the media's caption, subcaption, or footnote.\n"
        "Rules:\n"
        "- Only use provided texts. Do not invent content.\n"
        "- If footnote spans multiple candidates, assign role=footnote to each and group via part_index.\n"
        "- If a candidate seems to be a mis-labeled title (node_level==1) but belongs to this media, include its node_idx in downgrade_nodes.\n"
        "- Keep ignore for unrelated candidates.\n"
        "Output strict JSON only matching the schema.\n"
    )
    messages = [
        {"role": "system", "content": "You only output JSON."},
        {"role": "user", "content": prompt + "\nPayload:\n" + json.dumps(payload, ensure_ascii=False)},
        {"role": "system", "content": json.dumps({"response_format": {"type": "json_schema", "json_schema": schema}})},
    ]
    return messages


def _call_llm(messages: List[Dict[str, Any]], *, model: str) -> Optional[Dict[str, Any]]:
    try:
        if model.lower().startswith("qwen"):
            from ..utils.llm_clients import qwen_llm_call
            raw = qwen_llm_call(messages, images=None, model=model, json_mode=True)
        else:
            from ..utils.llm_clients import gpt_llm_call
            raw = gpt_llm_call(messages, images=None, model=model, json_mode=True)
        return json.loads(raw)
    except Exception:
        return None


def _append_unique(lst: List[str], items: List[str]) -> List[str]:
    seen = set(lst)
    for it in items:
        s = (it or "").strip()
        if s and s not in seen:
            lst.append(s)
            seen.add(s)
    return lst


def _apply_result(
    media: Dict[str, Any],
    text_map: Dict[int, Dict[str, Any]],
    result: Dict[str, Any],
    *,
    writeback_media_fields: bool,
    require_confidence: bool,
    min_conf: float,
    geo_guard: bool,
    geo_params: Optional[Dict[str, float]] = None,
    geo_mode: str = "loose",
    min_text_sim: float = 0.5,
) -> Tuple[int, int, int]:
    downgraded = 0
    linked = 0
    written = 0

    confidence = result.get("confidence")
    can_downgrade = (not require_confidence) or (isinstance(confidence, (int, float)) and confidence >= min_conf)

    # links container on media
    links = media.setdefault("links", {})
    links.setdefault("captions", [])
    links.setdefault("subcaptions", [])
    links.setdefault("footnotes", [])

    # role assignments
    for ra in result.get("role_assignments", []) or []:
        nidx = ra.get("node_idx")
        role = ra.get("role")
        if not isinstance(nidx, int) or role not in ("caption", "subcaption", "footnote"):
            continue
        txt = text_map.get(nidx)
        if not txt:
            continue
        # mark link on text node
        txt["media_link"] = {
            "type": media.get("type"),
            "target_node_idx": media.get("node_idx"),
            "role": role,
            "via": "llm",
            "confidence": confidence,
        }
        linked += 1
        # Unconditional downgrade for caption/subcaption previously mislabeled as title
        try:
            if role in ("caption", "subcaption") and int(txt.get("node_level", -1)) == 1:
                allow = True
                if geo_guard:
                    if geo_mode == "loose":
                        allow = not _is_absurd_geometry(media, txt)
                    else:
                        params = geo_params or {}
                        allow = _is_same_column(media, txt, min_overlap=params.get("min_h_overlap", 0.25))
                if allow and float(min_text_sim) > 0:
                    sim = _max_caption_similarity(media, str(txt.get("text") or ""), merged_text=result.get("merged_text"))
                    allow = sim >= float(min_text_sim)
                    if not allow:
                        # annotate mismatch for debugging
                        ml = txt.setdefault("media_link", {})
                        ml["text_sim"] = sim
                    else:
                        # record sim even when allowed, for transparency
                        ml = txt.setdefault("media_link", {})
                        ml["text_sim"] = sim
                if allow:
                    txt["node_level"] = 0
                    txt["was_title"] = True
                    if isinstance(txt.get("media_link"), dict):
                        txt["media_link"]["was_title_downgraded"] = True
                    downgraded += 1
        except Exception:
            pass
        # add reverse link on media
        if role == "caption":
            if nidx not in links["captions"]:
                links["captions"].append(nidx)
        elif role == "subcaption":
            if nidx not in links["subcaptions"]:
                links["subcaptions"].append(nidx)
        else:
            if nidx not in links["footnotes"]:
                links["footnotes"].append(nidx)

    # downgrades
    if can_downgrade:
        for nidx in result.get("downgrade_nodes", []) or []:
            txt = text_map.get(nidx)
            if not txt:
                continue
            try:
                if int(txt.get("node_level", -1)) == 1:
                    allow = True
                    if geo_guard:
                        if geo_mode == "loose":
                            allow = not _is_absurd_geometry(media, txt)
                        else:
                            params = geo_params or {}
                            allow = _is_same_column(media, txt, min_overlap=params.get("min_h_overlap", 0.25))
                    if allow and float(min_text_sim) > 0:
                        sim = _max_caption_similarity(media, str(txt.get("text") or ""), merged_text=result.get("merged_text"))
                        allow = sim >= float(min_text_sim)
                        if not allow:
                            ml = txt.setdefault("media_link", {})
                            ml["text_sim"] = sim
                        else:
                            ml = txt.setdefault("media_link", {})
                            ml["text_sim"] = sim
                    if allow:
                        txt["node_level"] = 0
                        txt["was_title"] = True
                        if "media_link" in txt:
                            txt["media_link"]["was_title_downgraded"] = True
                        downgraded += 1
            except Exception:
                continue

    # optional write back to media fields
    if writeback_media_fields:
        merged = result.get("merged_text") or {}
        cap = merged.get("caption")
        subcap = merged.get("subcaption")
        foots = merged.get("footnotes") or []

        if media.get("type") == "image":
            if isinstance(cap, str) and cap.strip():
                media.setdefault("image_caption", [])
                media["image_caption"] = _append_unique(media["image_caption"], [cap.strip()])
                written += 1
            if isinstance(subcap, str) and subcap.strip():
                media["image_subcaption"] = subcap.strip()
                written += 1
            if isinstance(foots, list) and foots:
                media.setdefault("image_footnote", [])
                media["image_footnote"] = _append_unique(media["image_footnote"], [f.strip() for f in foots if isinstance(f, str)])
                written += 1
        else:
            if isinstance(cap, str) and cap.strip():
                media.setdefault("table_caption", [])
                media["table_caption"] = _append_unique(media["table_caption"], [cap.strip()])
                written += 1
            if isinstance(foots, list) and foots:
                media.setdefault("table_footnote", [])
                media["table_footnote"] = _append_unique(media["table_footnote"], [f.strip() for f in foots if isinstance(f, str)])
                written += 1

    return downgraded, linked, written


def cleanup_captions_with_llm(
    root: Dict[str, Any],
    *,
    llm_model: str = "qwen-vl-max",
    window_before: int = 3,
    window_after: int = 3,
    writeback_media_fields: bool = True,
    require_confidence: bool = True,
    min_confidence: float = 0.6,
    geo_guard: bool = True,
    geo_max_vgap: Optional[float] = 120.0,
    geo_min_h_overlap: float = 0.25,
    geo_max_width_ratio: float = 2.0,
    min_text_sim_to_downgrade: float = 0.5,
    geo_mode: str = "loose",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Use LLM to disambiguate caption/subcaption/footnote around media and fix title mislabels.
    Returns updated root and stats.
    """
    children: List[Dict[str, Any]] = list(root.get("children", []))
    # map for quick lookup of text nodes by node_idx
    text_by_idx: Dict[int, Dict[str, Any]] = {int(n.get("node_idx")): n for n in children if _is_text(n) and isinstance(n.get("node_idx"), int)}

    total_media = 0
    classified = 0
    downgraded_total = 0
    linked_total = 0
    writeback_total = 0
    llm_fail = 0

    for pos, node in enumerate(children):
        if not _is_media(node):
            continue
        total_media += 1

        before_nodes, after_nodes = _collect_page_neighbors(children, pos, before=window_before, after=window_after)
        # If no neighbors and media has no intrinsic fields, skip
        if not before_nodes and not after_nodes and not any(k in node for k in ("image_caption", "image_subcaption", "image_footnote", "table_caption", "table_footnote")):
            continue

        payload = _payload_for_llm(node, before_nodes, after_nodes)
        messages = _build_messages(payload)
        result = _call_llm(messages, model=llm_model)
        if not isinstance(result, dict):
            llm_fail += 1
            continue

        d, l, w = _apply_result(
            node,
            text_by_idx,
            result,
            writeback_media_fields=writeback_media_fields,
            require_confidence=require_confidence,
            min_conf=min_confidence,
            geo_guard=geo_guard,
            geo_params={
                "max_vgap": geo_max_vgap if geo_max_vgap is not None else 1e9,
                "min_h_overlap": geo_min_h_overlap,
                "max_width_ratio": geo_max_width_ratio,
            },
            geo_mode=geo_mode,
            min_text_sim=min_text_sim_to_downgrade,
        )
        downgraded_total += d
        linked_total += l
        writeback_total += w
        classified += 1

    root["children"] = children
    stats = {
        "total_media": total_media,
        "classified_media": classified,
        "downgraded_texts": downgraded_total,
        "linked_texts": linked_total,
        "writeback_updates": writeback_total,
        "llm_failures": llm_fail,
        "model": llm_model,
    }
    return root, stats
