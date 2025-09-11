from typing import Any, Dict, List, Optional


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def apply_items_order_and_levels(
    page_children: List[Dict[str, Any]],
    items: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for ch in page_children:
        nid = ch.get("node_id")
        if isinstance(nid, str):
            by_id[nid] = ch
    new_list: List[Dict[str, Any]] = []
    for it in items:
        ch = by_id.get(it["node_id"])  # type: ignore[index]
        if ch is None:
            continue
        lvl = int(it["level"])  # 0..N
        ch["node_level"] = (-1 if lvl == 0 else lvl)
        new_list.append(ch)
    return new_list


def apply_merges(
    page_children: List[Dict[str, Any]],
    merges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    id_to_idx = {ch.get("node_id"): i for i, ch in enumerate(page_children)}
    for m in merges or []:
        into = m.get("into")
        from_list = list(m.get("from", []))
        for fid in from_list:
            i_into = id_to_idx.get(into)
            i_from = id_to_idx.get(fid)
            if i_into is None or i_from is None:
                continue
            if abs(i_from - i_into) != 1:
                continue
            ch_into = page_children[i_into]
            ch_from = page_children[i_from]
            t_into = _normalize_ws(ch_into.get("text", ""))
            t_from = _normalize_ws(ch_from.get("text", ""))
            if i_from < i_into:
                ch_into["text"] = _normalize_ws(f"{t_from} {t_into}") if t_from else t_into
                page_children.pop(i_from)
                i_into -= 1
            else:
                ch_into["text"] = _normalize_ws(f"{t_into} {t_from}") if t_from else t_into
                page_children.pop(i_from)
            id_to_idx = {ch.get("node_id"): i for i, ch in enumerate(page_children)}
    return page_children


def apply_virtual_titles(
    page_children: List[Dict[str, Any]],
    *,
    page_idx: int,
    virtuals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not virtuals:
        return page_children
    id_to_idx = {ch.get("node_id"): i for i, ch in enumerate(page_children)}
    for v in virtuals:
        text = _normalize_ws(v.get("text", ""))
        level = int(v.get("level", 1))
        insert_after = v.get("insert_after")
        new_node: Dict[str, Any] = {
            "type": "text",
            "text": text,
            "page_idx": page_idx,
            "node_level": level,
        }
        if insert_after is not None and insert_after in id_to_idx:
            pos = id_to_idx[insert_after] + 1
        else:
            pos = 0
        page_children.insert(pos, new_node)
        id_to_idx = {ch.get("node_id"): i for i, ch in enumerate(page_children)}
    return page_children


def replace_page_children(
    children: List[Dict[str, Any]],
    page_idx: int,
    new_page_children: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    inserted = False
    i = 0
    n = len(children)
    while i < n:
        ch = children[i]
        if ch.get("page_idx") != page_idx:
            out.append(ch)
            i += 1
            continue
        if not inserted:
            out.extend(new_page_children)
            inserted = True
        while i < n and children[i].get("page_idx") == page_idx:
            i += 1
    if not inserted:
        out.extend(new_page_children)
    return out


def _reindex_and_indices(
    doc_id: str,
    children: List[Dict[str, Any]],
    *,
    source_path: Optional[str] = None,
    include_indices: bool = True,
) -> Dict[str, Any]:
    for i, ch in enumerate(children):
        ch["node_idx"] = i
        ch["node_id"] = f"{doc_id}#{i}"
    root: Dict[str, Any] = {"type": "document", "doc_id": doc_id, "children": children}
    if source_path:
        root["source_path"] = source_path
    if include_indices:
        by_page: Dict[int, List[int]] = {}
        by_type: Dict[str, List[int]] = {}
        id_to_idx: Dict[str, int] = {}
        for i, ch in enumerate(children):
            p = ch.get("page_idx")
            if isinstance(p, int):
                by_page.setdefault(p, []).append(i)
            t = ch.get("type")
            if isinstance(t, str):
                by_type.setdefault(t, []).append(i)
            id_to_idx[f"{doc_id}#{i}"] = i
        root["indices"] = {"by_page": by_page, "by_type": by_type, "id_to_idx": id_to_idx}
    return root


def apply_plan_to_document(
    root: Dict[str, Any],
    page_idx: int,
    norm_plan: Dict[str, Any],
    *,
    include_indices: bool = True,
) -> Dict[str, Any]:
    children: List[Dict[str, Any]] = list(root.get("children", []))
    # Slice this page
    page_children = [ch for ch in children if ch.get("page_idx") == page_idx]
    if not page_children:
        return root
    # Items (reorder + levels)
    page_children = apply_items_order_and_levels(page_children, norm_plan.get("items", []))
    # Merges
    if norm_plan.get("merges"):
        page_children = apply_merges(page_children, norm_plan.get("merges", []))
    # Virtual titles
    if norm_plan.get("virtual_titles"):
        page_children = apply_virtual_titles(page_children, page_idx=page_idx, virtuals=norm_plan.get("virtual_titles", []))
    # Replace
    new_children = replace_page_children(children, page_idx, page_children)
    # Reindex
    return _reindex_and_indices(
        root.get("doc_id") or "document", new_children, source_path=root.get("source_path"), include_indices=include_indices
    )


def _is_allowed_node_keys(node: Dict[str, Any]) -> bool:
    allowed = {
        "type",
        "text",
        "page_idx",
        "node_level",
        "img_path",
        "outline",
        "image_caption",
        "image_footnote",
        "table_caption",
        "table_footnote",
        "table_body",
        "table_text",
        "text_format",
    }
    # node_id/node_idx are re-created by reindex, but allow if present
    allowed_optional = {"node_id", "node_idx"}
    return set(node.keys()).issubset(allowed | allowed_optional)


def apply_nodes_to_document(
    root: Dict[str, Any],
    page_idx: int,
    page_nodes: List[Dict[str, Any]],
    *,
    include_indices: bool = True,
) -> Dict[str, Any]:
    # Basic validation: keys subset, types, page_idx match
    valid_nodes: List[Dict[str, Any]] = []
    for n in page_nodes:
        if not isinstance(n, dict):
            continue
        if not _is_allowed_node_keys(n):
            continue
        if n.get("page_idx") != page_idx:
            continue
        t = n.get("type")
        if not isinstance(t, str):
            continue
        # Ensure node_level exists; default to -1
        if not isinstance(n.get("node_level"), int):
            n["node_level"] = -1
        valid_nodes.append(dict(n))

    if not valid_nodes:
        return root

    children: List[Dict[str, Any]] = list(root.get("children", []))
    new_children = replace_page_children(children, page_idx, valid_nodes)
    return _reindex_and_indices(
        root.get("doc_id") or "document", new_children, source_path=root.get("source_path"), include_indices=include_indices
    )

