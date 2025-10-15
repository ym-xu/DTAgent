"""Figure finder adapter.

根据页面信息生成图像/图表 ROI 列表。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ...retriever.manager import RetrieverManager
from ..types import ToolCall, ToolResult


def find_regions(call: ToolCall) -> ToolResult:
    manager = _require_manager(call.args.get("_retriever_manager"))
    context = call.args.get("_context") or {}

    source_id = call.args.get("source") or call.args.get("from")
    source_pages: Sequence[dict] = []
    if isinstance(source_id, str):
        src = context.get(source_id)
        if isinstance(src, ToolResult):
            data = src.data if isinstance(src.data, dict) else {}
            source_pages = data.get("pages") or []

    if not source_pages:
        pages = call.args.get("pages")
        source_pages = [{"page": page, "nodes": manager.resources.page_index.get(page, [])} for page in _normalize_pages(pages, manager.resources.page_index.keys())]

    allowed = call.args.get("kinds") or call.args.get("want") or ["image"]
    allowed_set = {str(kind).lower() for kind in allowed}
    labels = call.args.get("labels") or call.args.get("label")
    label_targets: set[str] = set()
    if isinstance(labels, (list, tuple, set)):
        for label in labels:
            if isinstance(label, str):
                normalized = label.strip()
                if not normalized:
                    continue
                node_id = manager.resources.label_index.get(normalized) or manager.resources.label_index.get(normalized.lower())
                if node_id:
                    label_targets.add(node_id)
    elif isinstance(labels, str) and labels.strip():
        normalized = labels.strip()
        node_id = manager.resources.label_index.get(normalized) or manager.resources.label_index.get(normalized.lower())
        if node_id:
            label_targets.add(node_id)

    rois: List[Dict[str, object]] = []
    for entry in source_pages:
        page = entry.get("page")
        nodes = entry.get("nodes") or []
        for node_id in nodes:
            role = manager.resources.node_roles.get(node_id)
            if not role or role.lower() not in allowed_set:
                continue
            if label_targets and node_id not in label_targets:
                continue
            meta = manager.resources.image_meta.get(node_id, {})
            roi = {
                "page": page,
                "node_id": node_id,
                "bbox": meta.get("bbox"),
                "caption": meta.get("caption"),
                "description": meta.get("description"),
                "path": manager.resources.image_paths.get(node_id),
                "role": role,
            }
            rois.append(roi)

    status = "ok" if rois else "empty"
    metrics = {"n_rois": len(rois)}
    return ToolResult(status=status, data={"rois": rois}, metrics=metrics)


def _require_manager(manager: Optional[RetrieverManager]) -> RetrieverManager:
    if not isinstance(manager, RetrieverManager):
        raise RuntimeError("figure_finder.find_regions requires RetrieverManager instance")
    return manager


def _normalize_pages(pages, available: Sequence[int]) -> List[int]:
    if pages in (None, "all"):
        return sorted({int(p) for p in available if isinstance(p, int)})
    values: List[int] = []
    iterable = pages if isinstance(pages, (list, tuple, set)) else [pages]
    for item in iterable:
        try:
            page = int(item)
        except (TypeError, ValueError):
            continue
        if page not in values:
            values.append(page)
    return values


__all__ = ["find_regions"]
