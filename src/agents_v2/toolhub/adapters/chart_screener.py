"""Chart screener adapter."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...retriever.manager import RetrieverManager, RetrieverResources
from ..types import CommonHit, ToolCall, ToolResult


def screen(call: ToolCall) -> ToolResult:
    resources = _resolve_resources(call)
    if resources is None:
        raise RuntimeError("chart_screener.screen requires RetrieverResources")

    context = call.args.get("_context") or {}
    source_key = call.args.get("source") or call.args.get("from")
    nodes: Set[str] = set()
    rois: List[Dict[str, object]] = []

    if isinstance(source_key, str):
        src = context.get(source_key)
        if isinstance(src, ToolResult):
            info = src.info if isinstance(src.info, dict) else {}
            rois.extend(info.get("rois") or [])
            for hit in src.hits or []:
                node_id = getattr(hit, "node_id", None)
                if isinstance(node_id, str):
                    nodes.add(node_id)

    extra_nodes = call.args.get("nodes")
    if isinstance(extra_nodes, str):
        nodes.add(extra_nodes)
    elif isinstance(extra_nodes, Iterable):
        for item in extra_nodes:
            if isinstance(item, str):
                nodes.add(item)

    for roi in rois:
        node_id = roi.get("node_id")
        if isinstance(node_id, str):
            nodes.add(node_id)

    if not nodes:
        return ToolResult(status="empty", hits=[], metrics={"n_candidates": 0})

    detections: List[Dict[str, object]] = []
    for node_id in sorted(nodes):
        analysis = _analyze_node(node_id, resources)
        if analysis is None:
            continue
        detections.append(analysis)

    if not detections:
        return ToolResult(status="empty", hits=[], metrics={"n_candidates": len(nodes)})

    only_positive = bool(call.args.get("only_positive", True))
    hits: List[CommonHit] = []
    positives = 0

    for det in detections:
        if only_positive and not det["has_chart"]:
            continue
        score = 1.0 if det["has_chart"] else 0.2
        if det["has_chart"]:
            positives += 1
        provenance = {
            "tool": "chart_screener.screen",
            "page_idx": det.get("page_idx"),
            "signals": {
                "has_axes": det["has_axes"],
                "has_legend": det["has_legend"],
                "keyword_hit": det["keyword_hit"],
            },
        }
        meta = {
            "chart_type": det["chart_type"],
            "span_roles": det["span_roles"],
            "caption": det.get("caption"),
        }
        hits.append(
            CommonHit(
                node_id=det["node_id"],
                evidence_type="graphics",
                score=score,
                provenance={k: v for k, v in provenance.items() if v is not None},
                modality="image",
                affordances=["chart_detect"],
                meta={k: v for k, v in meta.items() if v},
            )
        )

    status = "ok" if positives or (hits and not only_positive) else "empty"
    metrics = {
        "n_candidates": len(detections),
        "n_positive": positives,
    }
    return ToolResult(status=status, hits=hits, metrics=metrics, info={"results": detections})


def _analyze_node(node_id: str, resources: RetrieverResources) -> Optional[Dict[str, object]]:
    spans = resources.figure_spans.get(node_id, [])
    meta = resources.image_meta.get(node_id, {})
    span_roles = {
        span.get("role")
        for span in spans
        if isinstance(span, dict) and isinstance(span.get("role"), str)
    }
    chart_type = _first_non_empty(
        [meta.get("chart_type")]
        + [span.get("chart_type") for span in spans if isinstance(span, dict)]
    )
    caption = _first_non_empty(
        [
            meta.get("caption"),
            _first_span_text(spans, "figure_caption"),
            _first_span_text(spans, "figure_description"),
        ]
    )
    description = _first_non_empty([meta.get("description")])
    text_blob = " ".join(
        str(part)
        for part in (
            caption or "",
            description or "",
            _join_span_texts(spans, ("figure_caption", "figure_legend", "figure_ocr")),
        )
    ).lower()
    keyword_hit = any(token in text_blob for token in ("chart", "graph", "plot", "diagram"))
    has_axes = "axis_label" in span_roles
    has_legend = "figure_legend" in span_roles

    has_chart = bool(chart_type or has_axes or has_legend or keyword_hit)
    page_idx = resources.node_pages.get(node_id) or meta.get("page_idx")

    return {
        "node_id": node_id,
        "has_chart": has_chart,
        "chart_type": chart_type,
        "has_axes": has_axes,
        "has_legend": has_legend,
        "keyword_hit": keyword_hit,
        "span_roles": sorted(role for role in span_roles if role),
        "caption": caption,
        "page_idx": page_idx,
    }


def _resolve_resources(call: ToolCall) -> Optional[RetrieverResources]:
    resources = call.args.get("_resources")
    if isinstance(resources, RetrieverResources):
        return resources
    manager = call.args.get("_retriever_manager")
    if isinstance(manager, RetrieverManager):
        return manager.resources
    return None


def _first_non_empty(candidates: Sequence[object]) -> Optional[str]:
    for value in candidates:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _first_span_text(spans: Sequence[dict], role: str) -> Optional[str]:
    for span in spans:
        if isinstance(span, dict) and span.get("role") == role:
            text = span.get("dense_text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _join_span_texts(spans: Sequence[dict], roles: Tuple[str, ...]) -> str:
    parts: List[str] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        if span.get("role") in roles:
            text = span.get("dense_text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return " ".join(parts)


__all__ = ["screen"]
