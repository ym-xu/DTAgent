"""
Index Loaders
=============

从真实 DocTree 索引文件构建检索与导航所需的数据结构。
"""

from __future__ import annotations

import json
from collections import defaultdict
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

from .planner import DocGraphNavigator
from .utils.toc_utils import format_heading_outline
from .retriever import RetrieverResources
from .observer import NodeEvidence


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_resources_from_index(doc_dir: Path) -> Tuple[RetrieverResources, DocGraphNavigator]:
    """
    从索引目录加载 RetrieverResources 与 DocGraphNavigator。

    期望目录包含：
    - summary.json
    - graph_edges.jsonl
    """
    summary_path = doc_dir / "summary.json"
    edges_path = doc_dir / "graph_edges.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)

    summary = _read_json(summary_path)
    nodes = summary.get("nodes", [])
    doctree_map, parent_map, doctree_root = _load_doctree_map(doc_dir)

    label_index: Dict[str, str] = {}
    page_index: Dict[int, List[str]] = defaultdict(list)
    text_index: Dict[str, str] = {}
    table_structs: Dict[str, Dict[str, object]] = {}
    node_roles: Dict[str, str] = {}
    image_paths: Dict[str, str] = {}
    image_meta: Dict[str, Dict[str, object]] = {}
    node_pages: Dict[str, int] = {}
    node_physical_pages: Dict[str, int] = {}

    FIGURE_LABEL_RE = re.compile(r"\b(?:Figure|Fig\.?)\s*([A-Za-z0-9]+)", re.IGNORECASE)
    TABLE_LABEL_RE = re.compile(r"\bTable\s*([A-Za-z0-9]+)", re.IGNORECASE)

    def _register_label(key: Optional[str], node_id: str, *, prioritize: bool = False) -> None:
        if not isinstance(key, str):
            return
        trimmed = key.strip()
        if not trimmed:
            return
        if prioritize or trimmed not in label_index:
            label_index[trimmed] = node_id
        lower = trimmed.lower()
        if prioritize or lower not in label_index:
            label_index[lower] = node_id

    def _extract_caption_aliases(node: dict, tree_node: Optional[dict]) -> List[str]:
        texts: List[str] = []
        for key in ("title", "summary", "dense_text"):
            value = node.get(key)
            if isinstance(value, str):
                texts.append(value)
        hints = node.get("hints")
        if isinstance(hints, list):
            texts.extend(str(h) for h in hints if isinstance(h, str))
        if tree_node:
            caption = _extract_caption(tree_node)
            if isinstance(caption, str):
                texts.append(caption)
        aliases: List[str] = []
        for text in texts:
            for match in FIGURE_LABEL_RE.findall(text):
                label = f"Figure {match}"
                aliases.append(label)
            for match in TABLE_LABEL_RE.findall(text):
                label = f"Table {match}"
                aliases.append(label)
        return aliases

    def _register_page(page_value, node_id: str) -> None:
        if page_value is None:
            return
        if isinstance(page_value, int):
            if node_id not in page_index[page_value]:
                page_index[page_value].append(node_id)
        elif isinstance(page_value, str):
            cleaned = page_value.strip()
            if not cleaned:
                return
            if cleaned.isdigit():
                key = int(cleaned)
                if node_id not in page_index[key]:
                    page_index[key].append(node_id)
            else:
                roman_val = _roman_to_int(cleaned)
                if roman_val is not None:
                    if node_id not in page_index[roman_val]:
                        page_index[roman_val].append(node_id)

    summary_ids: Set[str] = set()

    for node in nodes:
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        summary_ids.add(node_id)
        label = node.get("label")
        if isinstance(label, str):
            _register_label(label, node_id, prioritize=True)
        title = node.get("title") or node.get("heading")
        _register_label(title or None, node_id)
        title_norm = node.get("title_norm")
        _register_label(title_norm or None, node_id)
        role = node.get("role")
        if isinstance(role, str):
            node_roles[node_id] = role
        if role in {"image", "table"}:
            tree_node = doctree_map.get(node_id) if doctree_map else None
            for alias in _extract_caption_aliases(node, tree_node):
                _register_label(alias, node_id, prioritize=True)
        tree_node = doctree_map.get(node_id) if doctree_map else None
        page_idx = node.get("page_idx")
        if isinstance(page_idx, int):
            _register_page(page_idx, node_id)
            _register_page(page_idx + 1, node_id)
            if page_idx >= 0:
                node_physical_pages.setdefault(node_id, page_idx + 1)
        logical_page = _resolve_logical_page(node_id, tree_node, doctree_map, parent_map)
        if logical_page is not None:
            _register_page(logical_page, node_id)
            node_pages[node_id] = int(logical_page)
        dense_text = node.get("dense_text") or node.get("summary") or node.get("title")
        if isinstance(dense_text, str) and dense_text.strip():
            text_index[node_id] = dense_text.strip()
        if role == "table":
            table_info = _parse_table_node(tree_node) if tree_node else {}
            if table_info:
                table_structs[node_id] = table_info
        elif role == "image":
            image_info = _parse_image_node(tree_node, base_dir=doc_dir) if tree_node else None
            if image_info:
                image_meta[node_id] = image_info
                if image_info.get("path"):
                    image_paths[node_id] = str(image_info["path"])

    if doctree_map:
        for node_id, tree_node in doctree_map.items():
            if not isinstance(node_id, str):
                continue
            if node_id in summary_ids:
                continue
            role = tree_node.get("role") or tree_node.get("type")
            if isinstance(role, str):
                node_roles.setdefault(node_id, role)
            logical_page = _resolve_logical_page(node_id, tree_node, doctree_map, parent_map)
            if logical_page is not None:
                _register_page(logical_page, node_id)
                node_pages[node_id] = int(logical_page)
            page_idx = tree_node.get("page_idx")
            if isinstance(page_idx, int):
                _register_page(page_idx + 1, node_id)
                _register_page(page_idx, node_id)
                if page_idx >= 0:
                    node_physical_pages.setdefault(node_id, page_idx + 1)
            if role == "image":
                image_info = _parse_image_node(tree_node, base_dir=doc_dir)
                if image_info:
                    text_index[node_id] = (image_info.get("summary") or image_info.get("caption") or "").strip()
                    table_structs.pop(node_id, None)
                    image_meta[node_id] = image_info
                    if image_info.get("path"):
                        image_paths[node_id] = str(image_info["path"])
            elif role == "table":
                table_info = _parse_table_node(tree_node)
                if table_info:
                    table_structs[node_id] = table_info
                    preview = table_info.get("preview") or table_info.get("caption")
                    if preview:
                        text_index[node_id] = preview
            else:
                text = tree_node.get("text")
                if isinstance(text, str) and text.strip():
                    text_index.setdefault(node_id, text.strip())

    dense_views, dense_base_ids = _load_dense_views(doc_dir)
    sparse_docs = _load_sparse_docs(doc_dir)
    figure_spans_map, figure_tokens_map = _load_figure_spans(doc_dir)

    resources = RetrieverResources(
        label_index=label_index,
        page_index=dict(page_index),
        text_index=text_index,
        dense_views=dense_views,
        dense_base_ids=dense_base_ids,
        sparse_docs=sparse_docs,
        tables=table_structs,
        node_roles=node_roles,
        image_paths=image_paths,
        image_meta=image_meta,
        node_pages=node_pages,
        node_physical_pages=node_physical_pages,
        figure_spans=figure_spans_map,
        figure_tokens=figure_tokens_map,
        base_dir=doc_dir,
        toc_outline=[],
        heading_index={},
    )

    if doctree_root:
        heading_index: Dict[str, List[str]] = defaultdict(list)
        heading_titles: Dict[str, str] = {}
        heading_children: Dict[str, List[str]] = defaultdict(list)

        def _walk(node: Dict, parent_id: Optional[str]) -> None:
            if not isinstance(node, dict):
                return
            node_id = node.get("node_id")
            node_type = str(node.get("role") or node.get("type") or "").lower()
            heading_text = _extract_heading_text(node)
            if isinstance(node_id, str) and heading_text and node_type in {"section", "heading"}:
                key = _normalize_heading_key(heading_text)
                heading_index[key].append(node_id)
                heading_titles[node_id] = heading_text
                if parent_id:
                    heading_children[parent_id].append(node_id)

            for child in node.get("children", []) or []:
                if isinstance(child, dict):
                    child_id = child.get("node_id") if isinstance(child.get("node_id"), str) else None
                    _walk(child, node_id if isinstance(node_id, str) else parent_id)

        _walk(doctree_root, None)
        resources.heading_index = {k: v for k, v in heading_index.items()}
        resources.heading_titles = heading_titles
        resources.heading_children = {k: v for k, v in heading_children.items()}

    graph = _build_graph(edges_path)
    if doctree_root:
        try:
            toc = format_heading_outline(
                doctree_root,
                by_logical_page=True,
                include_meta=False,
                include_media=False,
                max_len=80,
            )
            resources.toc_outline = toc[:200]
        except Exception:
            resources.toc_outline = []
    return resources, graph


def _load_dense_views(doc_dir: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    path = doc_dir / "dense_coarse.jsonl"
    dense_views: Dict[str, Dict[str, str]] = defaultdict(dict)
    dense_base_ids: Dict[str, Dict[str, str]] = defaultdict(dict)
    if not path.exists():
        return {}, {}, None
    for rec in _read_jsonl(path):
        node_id = rec.get("node_id")
        dense_text = rec.get("dense_text")
        if not isinstance(node_id, str) or not isinstance(dense_text, str):
            continue
        variant = rec.get("variant") or rec.get("role") or "default"
        variant = str(variant)
        dense_views[variant][node_id] = dense_text
        base_id = node_id.split("#", 1)[0]
        dense_base_ids[variant][node_id] = base_id
    return dict(dense_views), dict(dense_base_ids)


def _load_sparse_docs(doc_dir: Path) -> Dict[str, Dict[str, str]]:
    path = doc_dir / "sparse_coarse.jsonl"
    docs: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return docs
    useful_fields = ["title", "caption", "table_schema", "aliases", "body", "heading", "headings_path"]
    for rec in _read_jsonl(path):
        doc_id = rec.get("id")
        if not isinstance(doc_id, str):
            continue
        fields: Dict[str, str] = {}
        for key in useful_fields:
            val = rec.get(key)
            if isinstance(val, str) and val.strip():
                fields[key] = val.strip()
        docs[doc_id] = fields
    return docs


def _load_figure_spans(doc_dir: Path) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, List[str]]]:
    path = doc_dir / "figure_spans.jsonl"
    spans_map: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    tokens_map: Dict[str, List[str]] = defaultdict(list)
    if not path.exists():
        return {}, {}
    for rec in _read_jsonl(path):
        node_id = rec.get("node_id")
        if not isinstance(node_id, str):
            continue
        spans_map[node_id].append(rec)
        tokens = _collect_span_tokens(rec)
        if tokens:
            tokens_map[node_id].extend(tokens)
    return {node: spans for node, spans in spans_map.items()}, {node: tokens for node, tokens in tokens_map.items()}


def _collect_span_tokens(span: Dict[str, object]) -> List[str]:
    tokens: List[str] = []
    dense = span.get("dense_text")
    if isinstance(dense, str):
        tokens.extend(_expanded_tokens(dense))
    sparse = span.get("sparse_tokens")
    if isinstance(sparse, list):
        for item in sparse:
            if isinstance(item, str):
                tokens.append(item.lower())
    chart_type = span.get("chart_type")
    if isinstance(chart_type, str):
        tokens.extend(_expanded_tokens(chart_type))
    return tokens


def _tokenize_text(text: str) -> List[str]:
    text = text.lower()
    return [token for token in re.findall(r"[a-z0-9]+", text)]


def _expanded_tokens(text: str) -> List[str]:
    words = _tokenize_text(text)
    tokens = list(words)
    tokens.extend(f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1))
    return tokens


def _build_graph(edges_path: Path) -> DocGraphNavigator:
    children: Dict[str, List[str]] = defaultdict(list)
    parents: Dict[str, str] = {}
    same_page: Dict[str, List[str]] = defaultdict(list)

    for edge in _read_jsonl(edges_path):
        src = edge.get("src")
        dst = edge.get("dst")
        etype = edge.get("type")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        if etype == "child":
            children[src].append(dst)
            parents[dst] = src
        elif etype == "same_page":
            same_page[src].append(dst)
            same_page[dst].append(src)

    siblings: Dict[str, List[str]] = defaultdict(list)
    for parent, kids in children.items():
        for kid in kids:
            siblings[kid].extend(x for x in kids if x != kid)

    return DocGraphNavigator(
        children=dict(children),
        parents=parents,
        same_page=dict(same_page),
        siblings=dict(siblings),
    )


def build_observer_store(doc_dir: Path) -> Dict[str, NodeEvidence]:
    """根据 summary 和 dense_leaf 构建节点证据映射。"""
    summary_path = doc_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = _read_json(summary_path)
    nodes = summary.get("nodes", [])

    store: Dict[str, NodeEvidence] = {}
    for node in nodes:
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        role = node.get("role") or "text"
        modality = "text"
        if role == "image":
            modality = "image"
        elif role == "table":
            modality = "table"
        content = node.get("dense_text") or node.get("summary") or node.get("title")
        extra = {
            "title": node.get("title"),
            "hints": node.get("hints"),
            "page_idx": node.get("page_idx"),
            "label": node.get("label"),
        }
        store[node_id] = NodeEvidence(
            node_id=node_id,
            modality=modality,
            content=content,
            extra={k: v for k, v in extra.items() if v is not None},
        )

    doctree_map, _, _ = _load_doctree_map(doc_dir)
    if doctree_map:
        for node_id, node in doctree_map.items():
            role = node.get("role") or node.get("type")
            if node_id in store:
                ev = store[node_id]
            else:
                modality = "text"
                if role == "image":
                    modality = "image"
                elif role == "table":
                    modality = "table"
                content = node.get("text") if isinstance(node.get("text"), str) else None
                page_idx = node.get("page_idx")
                ev = NodeEvidence(
                    node_id=node_id,
                    modality=modality,
                    content=content,
                    extra={k: v for k, v in {"page_idx": page_idx, "label": node.get("label")}.items() if v is not None},
                )
                store[node_id] = ev

            if role == "table":
                table_info = _parse_table_node(node)
                if table_info:
                    ev.extra.setdefault("table", table_info)
                    if table_info.get("preview"):
                        ev.content = table_info["preview"]
            elif role == "image":
                image_info = _parse_image_node(node, base_dir=doc_dir)
                if image_info:
                    ev.modality = "image"
                    ev.extra.setdefault("image", image_info)
                    if image_info.get("path"):
                        ev.extra.setdefault("image_path", image_info["path"])
                    if image_info.get("summary"):
                        ev.content = image_info["summary"]
            elif role == "list":
                list_info = _parse_list_node(node)
                if list_info:
                    items = list_info.get("items") or []
                    if items:
                        ev.extra.setdefault("structured_list", items)
                    ev.content = "\n".join(items) if items else list_info.get("preview") or ev.content

    dense_leaf_path = doc_dir / "dense_leaf.jsonl"
    if dense_leaf_path.exists():
        for rec in _read_jsonl(dense_leaf_path):
            node_id = rec.get("node_id")
            if not isinstance(node_id, str):
                continue
            content = rec.get("raw_text") or rec.get("dense_text")
            extra = {
                "parent_section": rec.get("filters", {}).get("parent_section") if isinstance(rec.get("filters"), dict) else None,
                "page_idx": rec.get("filters", {}).get("page_idx") if isinstance(rec.get("filters"), dict) else None,
            }
            store[node_id] = NodeEvidence(
                node_id=node_id,
                modality="text",
                content=content,
                extra={k: v for k, v in extra.items() if v is not None},
            )
    return store


def _load_doctree_map(doc_dir: Path) -> Tuple[Dict[str, dict], Dict[str, str], Optional[dict]]:
    path = doc_dir / "doctree.mm.json"
    if not path.exists():
        parent = doc_dir.parent
        candidate = parent / "doctree.mm.json"
        if candidate.exists():
            path = candidate
    if not path.exists():
        return {}, {}
    root = _read_json(path)
    stack = [(root, None)]
    id_map: Dict[str, dict] = {}
    parent_map: Dict[str, str] = {}
    while stack:
        cur, parent_id = stack.pop()
        if not isinstance(cur, dict):
            continue
        nid = cur.get("node_id")
        if isinstance(nid, str):
            id_map[nid] = cur
            if parent_id:
                parent_map[nid] = parent_id
        for ch in cur.get("children", []) or []:
            if isinstance(ch, dict):
                stack.append((ch, nid if isinstance(nid, str) else parent_id))
    return id_map, parent_map, root


def _parse_table_node(node: dict) -> Dict[str, object] | None:
    data = node.get("data")
    columns: List[str] = []
    rows: List[List[str]] = []
    if isinstance(data, list):
        for idx, row in enumerate(data):
            if isinstance(row, dict):
                if not columns:
                    columns = list(row.keys())
                rows.append([str(row.get(col, "")) for col in columns])
            elif isinstance(row, (list, tuple)):
                if idx == 0 and not columns:
                    columns = [str(col) for col in row]
                else:
                    rows.append([str(cell) for cell in row])
    elif isinstance(data, str) and "<table" in data.lower():
        parsed = _parse_html_table(data)
        columns = parsed.get("columns", [])
        rows = parsed.get("rows", [])
    preview = None
    if rows and columns:
        preview = ", ".join(f"{col}: {rows[0][i]}" for i, col in enumerate(columns) if rows[0][i])
    caption = _extract_caption(node)
    return {
        "columns": columns,
        "rows": rows,
        "caption": caption,
        "preview": preview or caption or None,
    }


def _parse_list_node(node: dict) -> Dict[str, object] | None:
    items_raw = node.get("items")
    items: List[str] = []
    if isinstance(items_raw, list):
        for entry in items_raw:
            if isinstance(entry, dict):
                text = entry.get("text")
            else:
                text = entry
            if isinstance(text, str):
                cleaned = text.strip()
                if cleaned:
                    items.append(cleaned)
    if not items:
        return None
    preview = "; ".join(items[:2]) if items else None
    return {
        "items": items,
        "preview": preview,
    }


def _parse_image_node(node: dict, *, base_dir: Path) -> Dict[str, object] | None:
    caption = _extract_caption(node)
    desc = node.get("description")
    img_path = node.get("image_path") or node.get("path")
    resolved_path = _resolve_image_path(img_path, base_dir) if isinstance(img_path, str) else None
    out = {
        "caption": caption,
        "description": desc,
        "summary": "; ".join(filter(None, [caption, desc])) or None,
        "path": resolved_path,
        "bbox": node.get("bbox"),
    }
    return out


def _roman_to_int(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None
    s = text.lower()
    roman_map = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        value = roman_map.get(ch)
        if value is None:
            return None
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total if total > 0 else None


def _resolve_logical_page(
    node_id: str,
    tree_node: Optional[dict],
    doctree_map: Dict[str, dict],
    parent_map: Dict[str, str],
) -> Optional[int]:
    current_id = node_id
    current_node = tree_node or doctree_map.get(current_id)
    visited: set[str] = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        if isinstance(current_node, dict):
            lp = current_node.get("logical_page")
            if lp is not None:
                if isinstance(lp, int):
                    return lp
                if isinstance(lp, str):
                    cleaned = lp.strip()
                    if cleaned.isdigit():
                        return int(cleaned)
                    roman_val = _roman_to_int(cleaned)
                    if roman_val is not None:
                        return roman_val
        parent_id = parent_map.get(current_id)
        if not parent_id:
            break
        current_id = parent_id
        current_node = doctree_map.get(current_id)
    return None


def _extract_caption(node: dict) -> Optional[str]:
    for ch in node.get("children", []) or []:
        if isinstance(ch, dict) and (ch.get("role") == "caption" or ch.get("type") == "text"):
            text = ch.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    text = node.get("caption")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _parse_html_table(html: str) -> Dict[str, List[List[str]]]:
    from html.parser import HTMLParser

    class TableParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.rows: List[List[str]] = []
            self.row_tags: List[List[str]] = []
            self._current_row: List[str] = []
            self._current_tags: List[str] = []
            self._buffer: List[str] = []
            self._in_cell = False
            self._cell_tag = "td"

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self._current_row = []
                self._current_tags = []
            elif tag in ("td", "th"):
                self._in_cell = True
                self._cell_tag = tag
                self._buffer = []

        def handle_data(self, data):
            if self._in_cell:
                self._buffer.append(data)

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self._in_cell:
                text = "".join(self._buffer).strip()
                self._current_row.append(text)
                self._current_tags.append(self._cell_tag)
                self._in_cell = False
                self._buffer = []
            elif tag == "tr" and self._current_row:
                self.rows.append(self._current_row)
                self.row_tags.append(self._current_tags)
                self._current_row = []
                self._current_tags = []

    parser = TableParser()
    parser.feed(html)

    columns: List[str] = []
    data_rows: List[List[str]] = []
    for row, tags in zip(parser.rows, parser.row_tags):
        if not row:
            continue
        if not columns:
            if any(t == "th" for t in tags) or "<th" in html.lower():
                columns = row
                continue
            columns = row
            continue
        # pad or trim to match columns length
        if len(row) < len(columns):
            row = row + [""] * (len(columns) - len(row))
        elif len(row) > len(columns):
            row = row[: len(columns)]
        data_rows.append(row)

    return {"columns": columns, "rows": data_rows}


def _resolve_image_path(image_path: Optional[str], base_dir: Path) -> Optional[str]:
    if not image_path:
        return None
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    search_roots = [base_dir]
    parents = list(base_dir.parents)[:3]
    search_roots.extend(parents)
    for root in search_roots:
        resolved = (root / image_path).resolve()
        if resolved.exists():
            return str(resolved)
    return None


def _extract_heading_text(node: dict) -> Optional[str]:
    for key in ("heading_text", "text", "title", "label"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    caption = _extract_caption(node)
    if caption:
        return caption
    node_id = node.get("node_id")
    if isinstance(node_id, str) and node_id.strip():
        return node_id.strip()
    return None


def _normalize_heading_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


__all__ = ["build_resources_from_index", "build_observer_store"]
