from typing import Any, Dict, List, Sequence, Set, Tuple


def validate_and_normalize_jsonlist(
    items_obj: Any,
    *,
    allowed_ids: Sequence[str],
    level_max: int = 4,
) -> Tuple[bool, List[Dict[str, int]], List[str]]:
    """
    Validate a jsonlist (top-level array) of {node_id, level} and normalize it.

    Rules:
    - Must be a full permutation of allowed_ids (no missing/extra/duplicate ids)
    - level must be int in [0..level_max] (0 means non-title)
    """
    errors: List[str] = []
    if not isinstance(items_obj, list):
        return False, [], ["jsonlist must be a top-level array"]

    allowed: Set[str] = set(allowed_ids)
    seen: Set[str] = set()
    norm: List[Dict[str, int]] = []

    for it in items_obj:
        if not isinstance(it, dict):
            errors.append("jsonlist entries must be objects")
            continue
        nid = it.get("node_id")
        lvl = it.get("level")
        if not isinstance(nid, str):
            errors.append("item.node_id must be string")
            continue
        if nid not in allowed:
            errors.append(f"item.node_id not in page: {nid}")
            continue
        if nid in seen:
            errors.append(f"duplicate node_id in items: {nid}")
            continue
        if not isinstance(lvl, int) or not (0 <= lvl <= level_max):
            errors.append(f"level must be int in [0..{level_max}] for {nid}")
            continue
        seen.add(nid)
        norm.append({"node_id": nid, "level": int(lvl)})

    if len(seen) != len(allowed):
        missing = list(allowed - seen)
        errors.append(f"items must cover all nodes on page; missing={missing}")

    ok = len(errors) == 0
    return ok, norm, errors


def validate_and_normalize_plan(
    plan_obj: Any,
    *,
    allowed_ids: Sequence[str],
    types_by_id: Dict[str, str],
    page_idx: int,
    level_max: int = 4,
    max_merge_from: int = 3,
    max_virtual_titles: int = 4,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Validate and normalize a plan object with keys:
      page_idx, items, merges?, virtual_titles?
    """
    errors: List[str] = []
    allowed: Set[str] = set(allowed_ids)

    if not isinstance(plan_obj, dict):
        return False, {}, ["plan must be an object with keys: page_idx, items, merges?, virtual_titles?"]

    norm: Dict[str, Any] = {"page_idx": page_idx, "items": [], "merges": [], "virtual_titles": []}

    # page index
    pidx = plan_obj.get("page_idx")
    if pidx is not None and pidx != page_idx:
        errors.append(f"page_idx mismatch: plan={pidx} target={page_idx}")

    # items
    ok_items, norm_items, errs_items = validate_and_normalize_jsonlist(
        plan_obj.get("items"), allowed_ids=allowed_ids, level_max=level_max
    )
    if not ok_items:
        errors.extend(errs_items)
    norm["items"] = norm_items

    # adjacency order map
    order = {it["node_id"]: i for i, it in enumerate(norm_items)}

    # merges
    merges_in = plan_obj.get("merges", [])
    if merges_in:
        if not isinstance(merges_in, list):
            errors.append("merges must be a list")
        else:
            for m in merges_in:
                if not isinstance(m, dict):
                    errors.append("merge entry must be object")
                    continue
                into = m.get("into")
                frm = m.get("from")
                if not isinstance(into, str) or not isinstance(frm, list) or not frm:
                    errors.append("merge requires 'into': str and 'from': non-empty list")
                    continue
                if into not in allowed or any(not isinstance(x, str) or x not in allowed for x in frm):
                    errors.append("merge node_ids must be from this page")
                    continue
                # type check: only text
                if types_by_id.get(into) != "text" or any(types_by_id.get(x) != "text" for x in frm):
                    errors.append("merge only allowed for text nodes")
                    continue
                # adjacency check in order
                pos = order.get(into)
                adj_ok = pos is not None
                if adj_ok:
                    cur = pos
                    for x in frm:
                        px = order.get(x)
                        if px is None or abs(px - cur) != 1:
                            adj_ok = False
                            break
                        cur = min(cur, px)
                if not adj_ok:
                    errors.append(f"merge nodes must be adjacent in items order: into={into} from={frm}")
                    continue
                if len(frm) > max_merge_from:
                    errors.append(f"merge 'from' too long (>{max_merge_from}) for into={into}")
                    continue
                # record
                norm.setdefault("merges", []).append({"into": into, "from": frm})

    # virtual_titles
    virtuals_in = plan_obj.get("virtual_titles", [])
    if virtuals_in:
        if not isinstance(virtuals_in, list):
            errors.append("virtual_titles must be a list")
        else:
            vt_norm: List[Dict[str, Any]] = []
            for v in virtuals_in:
                if not isinstance(v, dict):
                    errors.append("virtual_title must be object")
                    continue
                text = v.get("text")
                lvl = v.get("level")
                insert_after = v.get("insert_after")
                if not isinstance(text, str) or not text.strip():
                    errors.append("virtual_title.text must be non-empty string")
                    continue
                if not isinstance(lvl, int) or not (1 <= lvl <= level_max):
                    errors.append("virtual_title.level must be int in [1..level_max]")
                    continue
                if insert_after is not None and insert_after not in allowed:
                    errors.append("virtual_title.insert_after must be a node_id from this page")
                    continue
                t = text.strip()
                if len(t) > 80:
                    errors.append("virtual_title.text too long (max 80)")
                    continue
                vt_norm.append({"text": t, "level": int(lvl), "insert_after": insert_after})
            if len(vt_norm) > max_virtual_titles:
                errors.append(f"too many virtual_titles (>{max_virtual_titles})")
            else:
                norm["virtual_titles"] = vt_norm

    ok = len(errors) == 0
    return ok, norm, errors

