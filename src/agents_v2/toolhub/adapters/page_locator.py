"""Page locator adapter.

根据页码返回页面节点与图像资源，供视觉链路使用。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ...retriever.manager import RetrieverManager
from ..types import ToolCall, ToolResult


def locate(call: ToolCall) -> ToolResult:
    manager = _require_manager(call.args.get("_retriever_manager"))
    resources = manager.resources

    pages = _normalize_pages(call.args.get("pages"), resources.page_index.keys())
    if not pages:
        return ToolResult(status="empty", data={"pages": []}, metrics={"n_pages": 0})

    page_records: List[dict] = []
    image_records: List[dict] = []

    for page in pages:
        nodes = list(resources.page_index.get(page, []))
        page_records.append({"page": page, "nodes": nodes})
        for node_id in nodes:
            if resources.node_roles.get(node_id) != "image":
                continue
            meta = resources.image_meta.get(node_id, {})
            image_records.append(
                {
                    "page": page,
                    "node_id": node_id,
                    "path": resources.image_paths.get(node_id),
                    "meta": meta,
                }
            )

    metrics = {"n_pages": len(pages), "n_images": len(image_records)}
    return ToolResult(
        status="ok" if page_records else "empty",
        data={
            "pages": page_records,
            "images": image_records,
        },
        metrics=metrics,
    )


def _require_manager(manager: Optional[RetrieverManager]) -> RetrieverManager:
    if not isinstance(manager, RetrieverManager):
        raise RuntimeError("page_locator.locate requires RetrieverManager instance")
    return manager


def _normalize_pages(pages, available: Sequence[int]) -> List[int]:
    if pages in (None, "all"):
        return sorted({int(p) for p in available if isinstance(p, int)})
    normalized: List[int] = []
    if isinstance(pages, (list, tuple, set)):
        items = pages
    else:
        items = [pages]
    for item in items:
        try:
            page = int(item)
        except (TypeError, ValueError):
            continue
        if page not in normalized:
            normalized.append(page)
    return normalized


__all__ = ["locate"]
