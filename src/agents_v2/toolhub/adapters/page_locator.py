"""Page locator adapter.

根据页码返回页面节点与图像资源，供视觉链路使用。
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import re

from ...retriever.manager import RetrieverManager
from ..types import CommonHit, ToolCall, ToolResult


def locate(call: ToolCall) -> ToolResult:
    manager = _require_manager(call.args.get("_retriever_manager"))
    resources = manager.resources

    pages = _normalize_pages(call.args.get("pages"), resources.page_index.keys())
    if not pages:
        return ToolResult(status="empty", hits=[], metrics={"n_pages": 0})

    page_records: List[dict] = []
    image_records: List[dict] = []
    hits: List[CommonHit] = []

    for page in pages:
        nodes = list(resources.page_index.get(page, []))
        page_records.append({"page": page, "nodes": nodes})
        for node_id in nodes:
            role = resources.node_roles.get(node_id, "text")
            evidence_type = "graphics" if role in {"image", "figure"} else "text"
            if role in {"table"}:
                evidence_type = "table"
            modality = "image" if role in {"image", "figure"} else ("table" if role == "table" else "text")
            provenance = {
                "tool": "page_locator.locate",
                "page_idx": page,
                "role": role,
            }
            hits.append(
                CommonHit(
                    node_id=node_id,
                    evidence_type=evidence_type,
                    score=1.0,
                    provenance=provenance,
                    modality=modality,
                    affordances=["page_neighbor"],
                    meta={"page": page, "role": role},
                )
            )
            if role == "image":
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
        status="ok" if hits else "empty",
        hits=hits,
        metrics=metrics,
        info={"pages": page_records, "images": image_records},
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
        if item is None:
            continue
        candidate = None
        if isinstance(item, str):
            text = item.strip().lower()
            if not text:
                continue
            m = re.search(r"(slide|page)\s*(\d+)", text)
            if m:
                candidate = int(m.group(2))
            elif text.isdigit():
                candidate = int(text)
        else:
            try:
                candidate = int(item)
            except (TypeError, ValueError):
                candidate = None
        if candidate is None:
            continue
        page = candidate
        if page not in normalized:
            normalized.append(page)
    return normalized


__all__ = ["locate"]
