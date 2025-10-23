"""Figure finder adapter.

根据页面信息生成图像/图表 ROI 列表。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from ...retriever.manager import RetrieverManager, RetrieverResources
from ..types import CommonHit, ToolCall, ToolResult


def find_regions(call: ToolCall) -> ToolResult:
    resources = _resolve_resources(call)
    if resources is None:
        raise RuntimeError("figure_finder.find_regions requires RetrieverResources")
    context = call.args.get("_context") or {}

    source_id = call.args.get("source") or call.args.get("from")
    source_pages: Sequence[dict] = []
    source_hits: Optional[Sequence[CommonHit]] = None
    if isinstance(source_id, str):
        src = context.get(source_id)
        if isinstance(src, ToolResult):
            data = src.data if isinstance(src.data, dict) else {}
            source_pages = data.get("pages") or []
            source_hits = src.hits
            if not source_pages:
                derived = _pages_from_hits(src.hits, resources)
                if derived:
                    source_pages = derived

    node_candidates: Set[str] = set()
    if source_hits:
        for hit in source_hits:
            node_id = getattr(hit, "node_id", None)
            if isinstance(node_id, str):
                node_candidates.add(node_id)
    for entry in source_pages or []:
        for node_id in entry.get("nodes") or []:
            if isinstance(node_id, str):
                node_candidates.add(node_id)

    if not node_candidates:
        pages = call.args.get("pages")
        if pages is not None:
            for page in _normalize_pages(pages, resources.page_index.keys()):
                node_candidates.update(resources.page_index.get(page, []))
        else:
            hints = _pages_from_hits(source_hits or [], resources)
            for entry in hints:
                for node_id in entry.get("nodes", []):
                    if isinstance(node_id, str):
                        node_candidates.add(node_id)
    if not node_candidates:
        return ToolResult(status="empty", hits=[], metrics={"reason": "no-nodes"})

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
                node_id = resources.label_index.get(normalized) or resources.label_index.get(normalized.lower())
                if node_id:
                    label_targets.add(node_id)
    elif isinstance(labels, str) and labels.strip():
        normalized = labels.strip()
        node_id = resources.label_index.get(normalized) or resources.label_index.get(normalized.lower())
        if node_id:
            label_targets.add(node_id)

    rois: List[Dict[str, object]] = []
    hits: List[CommonHit] = []
    for node_id in sorted(node_candidates):
        role = resources.node_roles.get(node_id)
        if not role or role.lower() not in allowed_set:
            continue
        if label_targets and node_id not in label_targets:
            continue
        meta = resources.image_meta.get(node_id, {})
        spans = resources.figure_spans.get(node_id, [])
        page_idx = resources.node_pages.get(node_id)
        roi = {
            "page": page_idx,
            "node_id": node_id,
            "bbox": meta.get("bbox"),
            "caption": meta.get("caption"),
            "description": meta.get("description"),
            "path": resources.image_paths.get(node_id),
            "role": role,
            "spans": spans,
        }
        rois.append(roi)
        provenance = {
            "tool": "figure_finder.find_regions",
            "page_idx": page_idx,
            "bbox": roi.get("bbox"),
            "image_path": roi.get("path"),
        }
        hit_meta = {
            "caption": roi.get("caption"),
            "description": roi.get("description"),
            "role": role,
            "label_targets": list(label_targets) if label_targets else None,
            "span_roles": sorted({span.get("role") for span in spans if isinstance(span, dict) and span.get("role")}),
        }
        hit_meta = {k: v for k, v in hit_meta.items() if v}
        hits.append(
            CommonHit(
                node_id=node_id,
                evidence_type="graphics",
                score=1.0,
                provenance={k: v for k, v in provenance.items() if v is not None},
                modality="image",
                affordances=["roi"],
                meta=hit_meta,
            )
        )

    status = "ok" if hits else "empty"
    metrics = {"n_rois": len(rois)}
    return ToolResult(status=status, hits=hits, metrics=metrics, info={"rois": rois})


def _resolve_resources(call: ToolCall) -> Optional[RetrieverResources]:
    resources = call.args.get("_resources")
    if isinstance(resources, RetrieverResources):
        return resources
    manager = call.args.get("_retriever_manager")
    if isinstance(manager, RetrieverManager):
        return manager.resources
    return None


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


def _pages_from_hits(hits: Sequence[CommonHit] | None, resources: RetrieverResources) -> List[Dict[str, object]]:
    if not hits:
        return []
    page_map: Dict[int, Set[str]] = {}
    for hit in hits:
        node_id = getattr(hit, "node_id", None)
        if not isinstance(node_id, str):
            continue
        page_idx = hit.provenance.get("page_idx") if isinstance(hit.provenance, dict) else None
        if page_idx is None:
            page_idx = resources.node_pages.get(node_id)
        if isinstance(page_idx, int):
            page_map.setdefault(page_idx, set()).add(node_id)
    return [{"page": page, "nodes": list(nodes)} for page, nodes in page_map.items()]


__all__ = ["find_regions"]
