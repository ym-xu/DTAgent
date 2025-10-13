from typing import Any, Dict, List, Optional, Tuple, Set
import json
import re

from .toc import _reindex_flat
from .llm_clients import gpt_llm_call


# def _norm_text(s: str) -> str:
#     t = (s or "").strip()
#     # remove leading numbering like "1.2 ", "A. ", "I. ", and trailing punctuation
#     t = re.sub(r"^(?:[A-Z]\.\s+|[IVXLCM]+\.\s+|\d+(?:\.\d+)*\s+)", "", t)
#     t = re.sub(r"\s+", " ", t)
#     t = t.strip(" .:;、。；：")
#     return t.lower()


# def _tokenize(s: str) -> List[str]:
#     return [w for w in re.split(r"[^A-Za-z0-9]+", s.lower()) if w]


# def _jaccard(a: List[str], b: List[str]) -> float:
#     if not a or not b:
#         return 0.0
#     sa, sb = set(a), set(b)
#     inter = len(sa & sb)
#     union = len(sa | sb)
#     if union == 0:
#         return 0.0
#     return inter / union


# def _sim_score(a: str, b: str) -> float:
#     if not a or not b:
#         return 0.0
#     a0, b0 = a.strip().lower(), b.strip().lower()
#     if a0 == b0:
#         return 1.0
#     a1, b1 = _norm_text(a), _norm_text(b)
#     if a1 and b1 and a1 == b1:
#         return 0.98
#     # substring containment
#     short, long = (a1, b1) if len(a1) <= len(b1) else (b1, a1)
#     if short and short in long:
#         # scale by coverage of long
#         try:
#             return max(0.7, min(0.97, len(short) / max(1, len(long))))
#         except Exception:
#             return 0.75
#     # token jaccard
#     return _jaccard(_tokenize(a1), _tokenize(b1)) * 0.9


def _logical_page_matches(node_value: Any, target_page: int) -> bool:
    if node_value is None:
        return False
    s = str(node_value).strip().lower()
    if not s or s == "cover":
        return False
    # allow "12" or "12-13"
    parts = [p.strip() for p in s.split("-") if p.strip()]
    return str(target_page) in parts


def _find_toc_node(children: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for n in children:
        if isinstance(n, dict) and n.get("type") == "toc" and isinstance(n.get("headings"), list):
            return n
    return None


def _flatten_toc(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    def dfs(nodes: List[Dict[str, Any]]):
        for h in nodes:
            title = h.get("title")
            level = h.get("level")
            page = h.get("page") if isinstance(h.get("page"), int) else None
            if isinstance(title, str) and isinstance(level, int) and level >= 1:
                out.append({"title": title, "level": int(level), "page": page})
            if isinstance(h.get("children"), list):
                dfs(h["children"])  
    dfs(headings)
    return out


def _first_index_of_logical_page(children: List[Dict[str, Any]], target_page: int) -> Optional[int]:
    for i, n in enumerate(children):
        if _logical_page_matches(n.get("logical_page"), target_page):
            return i
    return None


def _page_text_candidates(children: List[Dict[str, Any]], target_page: int) -> List[Tuple[int, Dict[str, Any]]]:
    cands: List[Tuple[int, Dict[str, Any]]] = []
    for i, n in enumerate(children):
        if n.get("type") != "text":
            continue
        if not _logical_page_matches(n.get("logical_page"), target_page):
            continue
        txt = n.get("text")
        if not isinstance(txt, str) or not txt.strip():
            continue
        cands.append((i, n))
    return cands


def _llm_choose_toc_candidate(
    title: str,
    candidates: List[Dict[str, Any]],
    *,
    model: str = "gpt-4o",
) -> Tuple[Optional[int], Optional[float]]:
    """
    Ask LLM to choose the best matching candidate node_idx for the given TOC title.
    candidates: list of {node_idx:int, text:str}. Returns (node_idx|None, confidence|None).
    """
    if not candidates:
        return None, None
    payload = {
        "title": str(title or ""),
        "candidates": [
            {
                "node_idx": int(c.get("node_idx")),
                "text": str(c.get("text", "")),
            }
            for c in candidates
            if isinstance(c.get("node_idx"), int)
        ],
    }
    messages = [
        {"role": "system", "content": "You only output JSON."},
        {
            "role": "user",
            "content": (
                "You are given one Table of Contents (TOC) title and a list of candidate headings on the same page.\n"
                "Task: choose the single candidate that corresponds to the TOC title.\n"
                "Guidance:\n"
                "- Consider fuzzy and semantic similarity; minor wording/numbering differences are acceptable.\n"
                "- If NONE is clearly corresponding, return null. Do NOT guess.\n"
                "Output strictly one JSON object: {\"choice\": node_idx|null, \"confidence\": number}.\n\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]
    try:
        raw = gpt_llm_call(messages, images=None, model=model, json_mode=True)
        obj = json.loads(raw)
        choice = obj.get("choice", None)
        conf = obj.get("confidence", None)
        node_idx = int(choice) if isinstance(choice, int) else None
        try:
            conf_f = float(conf) if conf is not None else None
        except Exception:
            conf_f = None
        return node_idx, conf_f
    except Exception:
        return None, None


def _collect_non_toc_candidate_positions(children: List[Dict[str, Any]]) -> List[int]:
    """Return a list of child indices that are MinerU-labeled title candidates (level==1),
    excluding frozen anchors and media-linked notes. Reuses _is_non_toc_candidate criteria.
    """
    pos: List[int] = []
    for i, n in enumerate(children):
        try:
            if _is_non_toc_candidate(n):
                pos.append(i)
        except Exception:
            continue
    return pos


def assign_heading_levels_case2(
    flat_root: Dict[str, Any],
    *,
    window_size: int = 20,
    llm_model: str = "gpt-4o",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Case 2: TOC present without page numbers.
    Align TOC headings to body candidates in reading order using LLM within sliding windows.
    - Does NOT insert missing TOC headings by default.
    - Freezes matched candidates with toc level.
    Returns updated root and stats.
    """
    if not isinstance(flat_root, dict):
        return flat_root, {"skipped": True}
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))
    if not children:
        return flat_root, {"skipped": True}

    toc_node = _find_toc_node(children)
    if not toc_node:
        return flat_root, {"skipped": True, "reason": "no_toc"}
    heads = toc_node.get("headings")
    if not isinstance(heads, list) or not heads:
        return flat_root, {"skipped": True, "reason": "empty_toc"}
    toc_entries = _flatten_toc(heads)
    if not toc_entries:
        return flat_root, {"skipped": True, "reason": "empty_toc"}

    # Body candidates: indices into children
    cand_positions = _collect_non_toc_candidate_positions(children)
    # Map node_idx -> child index for quick lookup
    nodeidx_to_childidx: Dict[int, int] = {}
    for ci in cand_positions:
        try:
            ni = int(children[ci].get("node_idx"))
            nodeidx_to_childidx[ni] = ci
        except Exception:
            continue

    matched = 0
    total = 0
    llm_calls = 0
    pos_ptr = 0  # pointer into cand_positions (monotonic)

    for entry in toc_entries:
        title = entry.get("title")
        level = entry.get("level")
        if not isinstance(title, str) or not isinstance(level, int) or level < 1:
            continue
        total += 1
        if pos_ptr >= len(cand_positions):
            break
        # Window of candidates in reading order
        win_end = min(len(cand_positions), pos_ptr + max(1, window_size))
        window_idxs = cand_positions[pos_ptr:win_end]
        llm_cands: List[Dict[str, Any]] = []
        for ci in window_idxs:
            node = children[ci]
            ni = node.get("node_idx")
            if isinstance(ni, int):
                llm_cands.append({"node_idx": ni, "text": node.get("text", "")})
        if not llm_cands:
            continue
        choice, conf = _llm_choose_toc_candidate(title, llm_cands, model=llm_model)
        llm_calls += 1
        if isinstance(choice, int) and choice in nodeidx_to_childidx:
            child_idx = nodeidx_to_childidx[choice]
            # Ensure chosen is within current window and after pointer
            if child_idx in window_idxs:
                node = children[child_idx]
                node["node_level"] = int(level)
                meta = node.setdefault("heading_meta", {})
                meta.update({
                    "via": "toc",
                    "frozen": True,
                    "toc_title": title,
                })
                if conf is not None:
                    try:
                        meta["confidence"] = float(conf)
                    except Exception:
                        pass
                matched += 1
                # advance pointer past this child
                # find its position in cand_positions
                try:
                    k = window_idxs.index(child_idx)
                    pos_ptr = pos_ptr + k + 1
                except Exception:
                    # fallback: move by 1
                    pos_ptr += 1
            else:
                # chosen outside window: ignore and continue (do not advance pointer)
                pass
        else:
            # no confident choice; keep pointer to allow next TOC to try from same spot
            pass

    flat_root["children"] = children
    stats = {
        "anchors_total": total,
        "anchors_matched": matched,
        "llm_calls": llm_calls,
    }
    return flat_root, stats


# === Case 3: No TOC present — assign heading levels from candidates only ===

def _collect_case3_candidates(children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    for i, n in enumerate(children):
        if _is_non_toc_candidate(n):
            # non-frozen MinerU=1, not caption/subcaption/footnote
            cands.append({
                "pos": i,
                "node_idx": n.get("node_idx"),
                "text": n.get("text", ""),
            })
    return cands


# (case3) legacy helpers removed


def _apply_case3_assignments(children: List[Dict[str, Any]], assigns: List[Dict[str, Any]]) -> int:
    """Apply LLM assignments (node_idx->level) to children. Returns count updated."""
    by_node_idx: Dict[int, Dict[str, Any]] = {}
    for n in children:
        try:
            ni = int(n.get("node_idx"))
            by_node_idx[ni] = n
        except Exception:
            continue
    updated = 0
    for a in assigns or []:
        nidx = a.get("node_idx")
        lvl = a.get("level")
        if not isinstance(nidx, int) or not isinstance(lvl, int):
            continue
        node = by_node_idx.get(nidx)
        if not node:
            continue
        # Only touch non-frozen text candidates
        if node.get("type") != "text" or _is_frozen_heading(node):
            continue
        node["node_level"] = max(0, int(lvl))
        meta = node.setdefault("heading_meta", {})
        meta["via"] = "llm_case3"
        updated += 1
    return updated

    # Case 3 helpers removed for redesign


def assign_heading_levels_case3(
    flat_root: Dict[str, Any],
    *,
    llm_model: str = "gpt-4o",
    max_level: int = 4,
    promote_media: bool = False,
    chunk_size: int = 60,
    overlap: int = 8,
    minimal: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Case 3 (redesign): build rich candidate payload (title + numbering + page + previews),
    ask LLM to assign structural levels and duplicates. Duplicates are converted to ordinary
    text nodes (node_level=0). Non-duplicates must remain headings (>=1).
    """
    if not isinstance(flat_root, dict):
        return flat_root, {"skipped": True}
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))
    if not children:
        return flat_root, {"skipped": True}
    # Skip when TOC exists (handled by Case 1/2)
    if _find_toc_node(children):
        return flat_root, {"skipped": True, "reason": "toc_present"}

    # Collect MinerU title candidates (level==1), excluding frozen headings and media-linked notes
    cands: List[int] = []
    for i, n in enumerate(children):
        if n.get("type") != "text":
            continue
        if _is_frozen_heading(n):
            continue
        try:
            if int(n.get("node_level", -1)) != 1:
                continue
        except Exception:
            continue
        ml = n.get("media_link")
        if isinstance(ml, dict) and ml.get("role") in ("caption", "subcaption", "footnote"):
            continue
        if not isinstance(n.get("node_idx"), int):
            continue
        cands.append(i)
    if not cands:
        return flat_root, {"skipped": True, "reason": "no_candidates"}

    # Simple numbering parser
    def parse_numbering(title: str) -> Dict[str, Any]:
        s = (title or "").strip()
        # numeric like 1.2.3 or 2. or 2)
        m = re.match(r"^(\d+(?:[\.:]\d+)*)[\s\)]", s)
        if m:
            toks = re.split(r"[\.:]", m.group(1))
            depth = len([t for t in toks if t])
            return {"type": "numeric", "depth": depth}
        # roman like I. II. IV.
        if re.match(r"^[IVXLCM]+\.[\s]", s):
            return {"type": "roman", "depth": 1}
        # alpha like A. B.
        if re.match(r"^[A-Z]\.[\s]", s):
            return {"type": "alpha", "depth": 1}
        return {"type": "none", "depth": 0}

    # Following text preview: first sentence of next body paragraph
    def following_preview(start_idx: int) -> str:
        j = start_idx + 1
        while j < len(children):
            m = children[j]
            if m.get("type") == "text":
                try:
                    lvl = int(m.get("node_level", -1))
                except Exception:
                    lvl = -1
                if lvl <= 0 and isinstance(m.get("text"), str):
                    t = re.sub(r"\s+", " ", m.get("text", "").strip())
                    # take first sentence
                    for sep in [". ", "? ", "! ", "。", "？", "！"]:
                        pos = t.find(sep)
                        if pos != -1:
                            t = t[: pos + 1]
                            break
                    return t[:160]
            j += 1
        return ""

    # Build entries payload
    entries: List[Dict[str, Any]] = []
    prev_titles: List[str] = []
    # helper for normalization
    def _norm_text_simple(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()
    # collect frequency of normalized titles across logical pages to hint duplicates
    norm_title_to_nodes: Dict[str, List[int]] = {}
    for idx in cands:
        n = children[idx]
        node_idx = int(n.get("node_idx"))
        title = str(n.get("text", ""))
        logical_page = n.get("logical_page") if isinstance(n.get("logical_page"), str) else (
            str(n.get("page_idx")) if isinstance(n.get("page_idx"), int) else None
        )
        num = parse_numbering(title)
        preview = following_preview(idx)
        entries.append({
            "node_idx": node_idx,
            "title": title,
            "logical_page": logical_page,
            "numbering": num,
            "following_text_preview": preview,
            "predecessor_titles": prev_titles[-5:],
        })
        prev_titles.append(title)
        norm_title_to_nodes.setdefault(_norm_text_simple(title), []).append(node_idx)

    # Compose LLM messages
    system_msg = (
        "You are a document structure analyzer. Assign a heading level (1.." + str(max_level) + ") to each title,\n"
        "and identify duplicates (non-structural repeats like headers/footers or repeated cover titles).\n"
        "Rules:\n"
        "- Level 1 is top-level chapter. Level 2 under level 1; level 3 under level 2; etc.\n"
        "- Avoid deep jumps (>+1) compared to the previous non-zero level in reading order.\n"
        "- Do NOT assign level 0 in assignments; use duplicates list to indicate items to ignore.\n"
        "- For academic/survey reports, titles like 'Acknowledgments', 'Research team', 'Methodology', 'Appendix', 'Glossary' are often module headings (likely high-level).\n"
        "Output strictly one JSON object: {\"assignments\": [{\"node_idx\": int, \"level\": int}], \"duplicates\": [{\"node_idx\": int, \"reason\": string}]}."
    )
    # Build dedup hints: titles that repeat across the document
    dedup_hints = [
        {"title": t, "count": len(ids)}
        for t, ids in norm_title_to_nodes.items()
        if len(ids) >= 2 and len(t) <= 60
    ]
    user_msg = json.dumps({"entries": entries, "dedup_hints": dedup_hints, "max_level": max_level}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    assigned = 0
    dup_count = 0
    try:
        raw = gpt_llm_call(messages, images=None, model=llm_model, json_mode=True)
        obj = json.loads(raw)
        assigns = obj.get("assignments") if isinstance(obj.get("assignments"), list) else []
        dups = obj.get("duplicates") if isinstance(obj.get("duplicates"), list) else []
        # Apply duplicates → convert to ordinary nodes (node_level=0)
        dup_set: Set[int] = set()
        for d in dups:
            nidx = d.get("node_idx")
            if not isinstance(nidx, int):
                continue
            dup_set.add(nidx)
            # locate node
            for n in children:
                if isinstance(n.get("node_idx"), int) and int(n.get("node_idx")) == nidx:
                    n["node_level"] = 0
                    meta = n.setdefault("heading_meta", {})
                    meta["duplicate"] = {
                        "ignored": True,
                        "reason": str(d.get("reason", "duplicate"))[:200],
                    }
                    break
        dup_count = len(dup_set)

        # Apply assignments (force non-duplicates to be >=1)
        for a in assigns:
            nidx = a.get("node_idx")
            lvl = a.get("level")
            if not isinstance(nidx, int) or not isinstance(lvl, int):
                continue
            if nidx in dup_set:
                continue
            final_lvl = max(1, min(int(lvl), max_level))
            # locate node
            for n in children:
                if isinstance(n.get("node_idx"), int) and int(n.get("node_idx")) == nidx:
                    n["node_level"] = final_lvl
                    meta = n.setdefault("heading_meta", {})
                    meta["via"] = "llm_case3_v2"
                    break
            assigned += 1

        # Fallback dedup (safety net): if some short/upper-case-like titles repeat and still not deduped, mark as duplicates
        def _is_uppercase_short(norm_title: str) -> bool:
            raw = norm_title.strip()
            # consider letters-only ratio uppercase in original title would be ideal; approximate: if words <= 5 and all tokens 2+ chars
            return len(raw.split()) <= 5 and raw.isupper()
        for norm_title, ids in norm_title_to_nodes.items():
            if len(ids) >= 2 and (len(norm_title.split()) <= 4 or _is_uppercase_short(norm_title)):
                for nid in ids:
                    if nid in dup_set:
                        continue
                    # convert to duplicate
                    for n in children:
                        if isinstance(n.get("node_idx"), int) and int(n.get("node_idx")) == nid:
                            n["node_level"] = 0
                            meta = n.setdefault("heading_meta", {})
                            meta["duplicate"] = {"ignored": True, "reason": "auto_dedup_by_frequency"}
                            dup_set.add(nid)
                            break
        dup_count = len(dup_set)
    except Exception:
        # minimal fallback: set the first candidate to L1
        n = children[cands[0]]
        n["node_level"] = 1
        meta = n.setdefault("heading_meta", {})
        meta["via"] = "llm_case3_seed"

    # Normalize globally (avoid deep jumps); by design, we do not promote media here
    children, corrections = _stack_normalize(children, max_level=max_level)
    doc_id = flat_root.get("doc_id") or "document"
    new_root = _reindex_flat(doc_id, children, flat_root.get("source_path"), preserve_meta=flat_root)
    stats = {
        "assigned": assigned,
        "duplicates": dup_count,
        "corrections": corrections,
        "model": llm_model,
    }
    return new_root, stats


## (legacy Case 3 block fully removed)


def _assign_or_insert_for_entry(
    children: List[Dict[str, Any]],
    entry: Dict[str, Any],
    assigned_indices: set,
    *,
    high_thr: float = 0.85,
    mid_thr: float = 0.72,
) -> Tuple[List[Dict[str, Any]], bool, Optional[int]]:
    """
    Try assign a TOC entry to an existing text node on the same logical page by similarity.
    If none exceeds threshold, insert a new heading node at the beginning of that page segment.
    Returns (children_updated, matched, node_index_assigned_or_insert_pos).
    """
    title = entry.get("title", "")
    level = int(entry.get("level", 1))
    page = entry.get("page")
    if not isinstance(page, int):
        return children, False, None

    # collect candidates on that page (text nodes only), excluding already assigned indices
    cands = _page_text_candidates(children, page)
    filtered: List[Tuple[int, Dict[str, Any]]] = [(idx, n) for (idx, n) in cands if idx not in assigned_indices]
    if filtered:
        # Build LLM candidates: node_idx + text
        llm_cands = []
        idx_by_nodeidx: Dict[int, int] = {}
        for child_idx, node in filtered:
            node_idx_val = node.get("node_idx")
            if isinstance(node_idx_val, int):
                llm_cands.append({"node_idx": node_idx_val, "text": node.get("text", "")})
                idx_by_nodeidx[int(node_idx_val)] = child_idx
        chosen_node_idx, conf = _llm_choose_toc_candidate(title, llm_cands, model="gpt-4o")
        if isinstance(chosen_node_idx, int) and chosen_node_idx in idx_by_nodeidx:
            top_idx = idx_by_nodeidx[chosen_node_idx]
            node = children[top_idx]
            node["node_level"] = level
            meta = node.setdefault("heading_meta", {})
            if conf is not None:
                meta["confidence"] = float(conf)
            meta.update({
                "via": "toc",
                "frozen": True,
                "toc_title": entry.get("title"),
                "toc_page": page,
            })
            assigned_indices.add(top_idx)
            return children, True, top_idx

    # not matched: insert at page block start
    insert_pos = _first_index_of_logical_page(children, page)
    if insert_pos is None:
        # page not present in children (should not happen), fallback append
        insert_pos = len(children)
    # try to reuse a page_idx from the first node on that logical page
    page_idx_val = None
    for j in range(insert_pos, len(children)):
        if _logical_page_matches(children[j].get("logical_page"), page):
            page_idx_val = children[j].get("page_idx")
            break
    new_node: Dict[str, Any] = {
        "type": "text",
        "text": entry.get("title", ""),
        "page_idx": page_idx_val,
        "logical_page": str(page),
        "node_level": level,
        "heading_meta": {
            "via": "toc",
            "inserted": True,
            "frozen": True,
            "confidence": 1.0,
            "toc_title": entry.get("title"),
            "toc_page": page,
        },
    }
    children = children[:insert_pos] + [new_node] + children[insert_pos:]
    # assigned index is insert position (will be reindexed later globally)
    return children, True, insert_pos


def _stack_normalize(children: List[Dict[str, Any]], max_level: int = 4) -> Tuple[List[Dict[str, Any]], int]:
    """
    Enforce non-jumping levels for headings while respecting frozen toc anchors.
    Returns (children_updated, corrections).
    """
    corrections = 0
    last_level = 0
    for n in children:
        if n.get("type") != "text":
            continue
        try:
            lvl = int(n.get("node_level", -1))
        except Exception:
            continue
        if lvl < 1:
            continue
        frozen = bool(n.get("heading_meta", {}).get("frozen"))
        if frozen:
            last_level = max(1, min(int(lvl), max_level))
            n["node_level"] = last_level
            continue
        desired = max(1, min(int(lvl), max_level))
        if desired > last_level + 1:
            desired = last_level + 1
            corrections += 1
            meta = n.setdefault("heading_meta", {})
            meta["corrected_cross_level"] = True
            meta.setdefault("via", "heuristic")
        n["node_level"] = desired
        last_level = desired
    return children, corrections


def assign_heading_levels_case1(
    flat_root: Dict[str, Any],
    *,
    high_threshold: float = 0.85,
    mid_threshold: float = 0.72,
    max_level: int = 4,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Case 1: TOC present with printed page numbers and nodes have logical_page assigned.
    - For each TOC heading with a page number, find a matching title on the same logical_page, assign its level and freeze it.
    - If none found, insert a new title node at the start of that logical_page's segment.
    - Finally, normalize heading levels globally without changing frozen anchors.
    Returns updated root and stats.
    """
    if not isinstance(flat_root, dict):
        return flat_root, {"skipped": True}
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))
    if not children:
        return flat_root, {"skipped": True}

    toc_node = _find_toc_node(children)
    if not toc_node:
        return flat_root, {"skipped": True, "reason": "no_toc"}
    headings = toc_node.get("headings")
    if not isinstance(headings, list) or not headings:
        return flat_root, {"skipped": True, "reason": "empty_toc"}

    toc_entries = _flatten_toc(headings)
    # keep only entries with page number
    toc_entries = [e for e in toc_entries if isinstance(e.get("page"), int)]
    if not toc_entries:
        return flat_root, {"skipped": True, "reason": "no_page_in_toc"}

    assigned_indices: set = set()
    toc_matched = 0
    toc_inserted = 0

    # Process in TOC order
    for e in toc_entries:
        before_len = len(children)
        children, matched, pos_idx = _assign_or_insert_for_entry(
            children,
            e,
            assigned_indices,
            high_thr=high_threshold,
            mid_thr=mid_threshold,
        )
        if matched:
            if len(children) > before_len:
                toc_inserted += 1
                # insertion shifts later indices; adjust assigned_indices greater than or equal to insert pos
                new_assigned: set = set()
                for idx in assigned_indices:
                    new_assigned.add(idx + 1 if pos_idx is not None and idx >= pos_idx else idx)
                assigned_indices = new_assigned
            else:
                toc_matched += 1

    # Normalize globally (respect frozen anchors)
    children, corrections = _stack_normalize(children, max_level=max_level)

    # Reindex
    doc_id = flat_root.get("doc_id") or "document"
    new_root = _reindex_flat(doc_id, children, flat_root.get("source_path"), preserve_meta=flat_root)
    stats = {
        "toc_matched": toc_matched,
        "toc_inserted": toc_inserted,
        "corrections": corrections,
        "total_toc_entries_with_page": len(toc_entries),
    }
    return new_root, stats


# === Case 1 (part 2): assign levels for non-TOC headings via LLM under TOC constraints ===

def _is_frozen_heading(n: Dict[str, Any]) -> bool:
    if n.get("type") != "text":
        return False
    try:
        lvl = int(n.get("node_level", -1))
    except Exception:
        return False
    if lvl < 1:
        return False
    return bool(n.get("heading_meta", {}).get("frozen", False))


def _is_non_toc_candidate(n: Dict[str, Any]) -> bool:
    if n.get("type") != "text":
        return False
    # Only MinerU-labeled headings (level==1); captions/subcaptions are treated as noise
    try:
        if int(n.get("node_level", -1)) != 1:
            return False
    except Exception:
        return False
    # skip frozen (i.e., TOC-anchored) and any media-linked notes
    if _is_frozen_heading(n):
        return False
    ml = n.get("media_link")
    if isinstance(ml, dict) and ml.get("role") in ("caption", "subcaption", "footnote"):
        return False
    # basic sanity: has shortish text
    t = (n.get("text") or "").strip()
    if not t:
        return False
    return True


def _idx_to_pos(children: List[Dict[str, Any]]) -> Dict[int, int]:
    m: Dict[int, int] = {}
    for i, n in enumerate(children):
        try:
            ni = int(n.get("node_idx"))
        except Exception:
            continue
        m[ni] = i
    return m


def _promote_media_in_span(children: List[Dict[str, Any]], start_pos: int, end_pos: int, level: int) -> int:
    """Set image/table between start_pos..end_pos (inclusive, clamped) to given level
    ONLY when the media has a caption (image_caption/table_caption). Returns count.
    """
    def _has_caption(n: Dict[str, Any]) -> bool:
        t = n.get("type")
        if t == "image":
            caps = n.get("image_caption")
            if isinstance(caps, list):
                return any(isinstance(x, str) and x.strip() for x in caps)
            return False
        if t == "table":
            caps = n.get("table_caption")
            if isinstance(caps, list):
                return any(isinstance(x, str) and x.strip() for x in caps)
            return False
        return False
    if start_pos > end_pos:
        start_pos, end_pos = end_pos, start_pos
    start_pos = max(0, start_pos)
    end_pos = min(len(children) - 1, end_pos)
    changed = 0
    for j in range(start_pos, end_pos + 1):
        n = children[j]
        if n.get("type") in ("image", "table") and _has_caption(n):
            n["node_level"] = level
            meta = n.setdefault("heading_meta", {})
            meta["via"] = "media_as_heading"
            changed += 1
    return changed


def _extract_anchor_path(children: List[Dict[str, Any]], upto_index: int) -> List[Dict[str, Any]]:
    """Return current frozen heading path (stack) just before position upto_index."""
    path: List[Dict[str, Any]] = []
    for i in range(upto_index + 1):
        n = children[i]
        if _is_frozen_heading(n):
            lvl = int(n.get("node_level", 1))
            # pop until strictly less than lvl
            while path and int(path[-1]["level"]) >= lvl:
                path.pop()
            path.append({"level": lvl, "title": str(n.get("text", ""))})
    return path


def _build_non_toc_groups(children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Anchor-based grouping: create a group for each segment between frozen headings.
    Each group stores [start_pos, end_pos], the anchor_path at start, and any
    non-TOC text candidates within. This ensures media promotion applies even
    when there are no text candidates in a segment.
    """
    anchors: List[int] = [i for i, n in enumerate(children) if _is_frozen_heading(n)]
    if not anchors:
        return []

    groups: List[Dict[str, Any]] = []
    for ai, pos in enumerate(anchors):
        start_pos = pos
        end_pos = (anchors[ai + 1] - 1) if ai + 1 < len(anchors) else (len(children) - 1)
        path = _extract_anchor_path(children, start_pos)
        cands: List[Dict[str, Any]] = []
        for j in range(start_pos, end_pos + 1):
            n = children[j]
            if _is_non_toc_candidate(n):
                cands.append(n)
        groups.append({
            "anchor_path": path,
            "candidates": cands,
            "start_pos": start_pos,
            "end_pos": end_pos,
        })
    return groups


def _toc_max_level_from_node(children: List[Dict[str, Any]]) -> int:
    toc = _find_toc_node(children)
    if not toc:
        return 1
    heads = toc.get("headings")
    if not isinstance(heads, list):
        return 1
    flat = _flatten_toc(heads)
    mx = 1
    for e in flat:
        try:
            lv = int(e.get("level", 1))
            if lv > mx:
                mx = lv
        except Exception:
            pass
    return max(1, mx)


def _build_llm_payload_for_group(group: Dict[str, Any], toc_max_level: int, max_level: int) -> Dict[str, Any]:
    path = group.get("anchor_path") or []
    base_level = int(path[-1]["level"]) if path else 0
    min_level = max(base_level + 1, toc_max_level + 1)
    forbid = list(range(1, toc_max_level + 1))
    cands = []
    for n in group.get("candidates", []):
        cands.append({
            "node_idx": n.get("node_idx"),
            "text": n.get("text", ""),
            "logical_page": n.get("logical_page"),
        })
    return {
        "anchor_path": path,
        "toc_max_level": toc_max_level,
        "constraints": {
            "min_level": min_level,
            "max_level": max_level,
            "forbid_levels": forbid,
        },
        "candidates": cands,
    }


def _messages_for_group(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = (
        "You are given a current heading path (from the Table of Contents anchors) and a list of non-TOC candidate headings.\n"
        "Task: assign a heading level for each candidate.\n"
        "Rules:\n"
        "- You MUST respect constraints: candidates cannot be assigned to any level in 'forbid_levels' (e.g., 1..toc_max_level).\n"
        "- Prefer continuous levels. If unsure whether it is a heading, use 0 (body).\n"
        "- Heuristic: If a candidate is a question or contains a question mark ('?' or '？'), it is LIKELY a section heading.\n"
        "  Therefore, avoid level 0 for such candidates unless there is strong evidence it is body text.\n"
        "- Otherwise, assign at least the 'min_level' and at most 'max_level'.\n"
        "Output JSON only: {\"assignments\": [{\"node_idx\": int, \"level\": int}]}. No comments.\n"
        "Payload follows.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": "You only output JSON."},
        {"role": "user", "content": prompt},
    ]


def assign_non_toc_levels_with_llm(
    flat_root: Dict[str, Any],
    *,
    llm_model: str = "gpt-4o",
    max_level: int = 4,
    chunk_size: int = 40,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Under Case 1: use TOC anchors to constrain levels for non-TOC headings (MinerU level==1),
    and ask LLM to decide between 0 (body) or levels in [min_level..max_level].
    Returns updated root and stats.
    """
    if not isinstance(flat_root, dict):
        return flat_root, {"skipped": True}
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))
    if not children:
        return flat_root, {"skipped": True}

    groups = _build_non_toc_groups(children)
    if not groups:
        return flat_root, {"skipped": True, "reason": "no_anchors"}

    toc_max = _toc_max_level_from_node(children)

    updated = 0
    llm_calls = 0
    llm_fail = 0
    clamped_min = 0
    clamped_max = 0
    to_zero = 0
    fallback_promoted = 0

    # process each group, splitting into chunks if too large
    idx_pos = _idx_to_pos(children)
    for g in groups:
        cands = g.get("candidates", [])
        # compute group's min_level from full group (not chunked payload)
        group_payload = _build_llm_payload_for_group(g, toc_max, max_level)
        min_level_group = group_payload["constraints"]["min_level"]
        # 1) Promote media (image/table) within this group's span to the group's min level
        span_start = int(g.get("start_pos", 0))
        span_end = int(g.get("end_pos", span_start))
        # media_promoted = _promote_media_in_span(children, span_start, span_end, min_level_group)
        media_promoted = 0
        kept_for_llm: List[Dict[str, Any]] = list(cands)
        # Track which candidates received an explicit LLM assignment
        assigned_nodes: Set[int] = set()
        for start in range(0, len(kept_for_llm), max(1, chunk_size)):
            part = {"anchor_path": g.get("anchor_path", []), "candidates": kept_for_llm[start:start + chunk_size]}
            payload = _build_llm_payload_for_group(part, toc_max, max_level)
            messages = _messages_for_group(payload)
            try:
                raw = gpt_llm_call(messages, images=None, model=llm_model, json_mode=True)
                obj = json.loads(raw)
                llm_calls += 1
            except Exception:
                llm_fail += 1
                continue
            assigns = obj.get("assignments") if isinstance(obj.get("assignments"), list) else []
            for a in assigns:
                nidx = a.get("node_idx")
                lvl = a.get("level")
                if not isinstance(nidx, int) or not isinstance(lvl, int):
                    continue
                node = next((n for n in kept_for_llm if n.get("node_idx") == nidx), None)
                if not node:
                    continue
                # apply constraints: allow 0, else clamp to [min..max]
                min_level = payload["constraints"]["min_level"]
                if lvl == 0:
                    node["node_level"] = 0
                    meta = node.setdefault("heading_meta", {})
                    meta["via"] = "llm_non_toc"
                    to_zero += 1
                else:
                    final = max(min_level, min(int(lvl), max_level))
                    if final == min_level and lvl < min_level:
                        clamped_min += 1
                    if final == max_level and lvl > max_level:
                        clamped_max += 1
                    node["node_level"] = final
                    meta = node.setdefault("heading_meta", {})
                    meta["via"] = "llm_non_toc"
                updated += 1
                assigned_nodes.add(int(nidx))

        # Fallback: if LLM did not assign a candidate, promote it to the group's minimum level
        # to respect TOC constraints and avoid leaving MinerU's level=1 under H1/H2 anchors.
        if kept_for_llm:
            for n in kept_for_llm:
                try:
                    if int(n.get("node_level", -1)) == 1 and int(n.get("node_idx")) not in assigned_nodes:
                        n["node_level"] = min_level_group
                        meta = n.setdefault("heading_meta", {})
                        if "via" not in meta:
                            meta["via"] = "heuristic_non_toc"
                        fallback_promoted += 1
                except Exception:
                    continue
        # add media promotions to updated counter
        updated += media_promoted

    # Normalize globally while respecting frozen anchors
    children, corrections = _stack_normalize(children, max_level=max_level)
    doc_id = flat_root.get("doc_id") or "document"
    new_root = _reindex_flat(doc_id, children, flat_root.get("source_path"), preserve_meta=flat_root)
    stats = {
        "groups": len(groups),
        "updated": updated,
        "llm_calls": llm_calls,
        "llm_failures": llm_fail,
        "clamped_to_min": clamped_min,
        "clamped_to_max": clamped_max,
        "to_body": to_zero,
        "fallback_promoted": fallback_promoted,
        "toc_max_level": toc_max,
    }
    return new_root, stats
