"""Structure adjacency adapters."""

from __future__ import annotations

from typing import Iterable, List, Optional, Set

from ...retriever.manager import RetrieverManager
from ..types import CommonHit, ToolCall, ToolResult


def expand(call: ToolCall) -> ToolResult:
    graph = _require_graph(call.args.get("_doc_graph"))
    seeds = _collect_seed_nodes(call)
    include_self = bool(call.args.get("include_self"))
    manager = _require_manager(call.args.get("_retriever_manager"))
    heading_children = getattr(manager.resources, "heading_children", {}) or {}

    neighbors: List[str] = []
    seen: Set[str] = set()
    for seed in seeds:
        for nid in graph.expand(seed, include_self=include_self):
            if nid not in seen:
                seen.add(nid)
                neighbors.append(nid)
        for nid in heading_children.get(seed, []):
            if nid not in seen:
                seen.add(nid)
                neighbors.append(nid)
    hits = [
        CommonHit(
            node_id=nid,
            evidence_type="layout",
            score=0.6,
            provenance={"tool": "structure.expand"},
            modality="text",
            affordances=["neighbors"],
            meta={"source": list(seeds)},
        )
        for nid in neighbors
    ]
    status = "ok" if hits else "empty"
    return ToolResult(status=status, hits=hits, metrics={"n_hits": len(hits)})


def children(call: ToolCall) -> ToolResult:
    graph = _require_graph(call.args.get("_doc_graph"))
    manager = _require_manager(call.args.get("_retriever_manager"))
    seeds = _collect_seed_nodes(call)
    node_roles = manager.resources.node_roles
    heading_children = getattr(manager.resources, "heading_children", {}) or {}

    desired_role = call.args.get("role")
    level = str(call.args.get("level") or "child").lower()
    results: List[str] = []
    seen: Set[str] = set()
    for seed in seeds:
        candidates: List[str] = []
        if level == "subheading" and heading_children:
            candidates = heading_children.get(seed, []) or []
        else:
            candidates = graph.children.get(seed, []) or []
        for child in candidates:
            if child in seen:
                continue
            if desired_role and node_roles.get(child) != desired_role:
                continue
            seen.add(child)
            results.append(child)
    hits = [
        CommonHit(
            node_id=nid,
            evidence_type="layout",
            score=0.7,
            provenance={"tool": "structure.children", "parent": list(seeds)},
            modality="text",
            affordances=["child"],
            meta={"role": node_roles.get(nid)},
        )
        for nid in results
    ]
    status = "ok" if hits else "empty"
    return ToolResult(status=status, hits=hits, metrics={"n_hits": len(hits)})


def _collect_seed_nodes(call: ToolCall) -> List[str]:
    seeds: List[str] = []
    reference = call.args.get("from") or call.args.get("node") or call.args.get("nodes")
    if isinstance(reference, str):
        seeds.extend(_resolve_reference(reference, call))
    elif isinstance(reference, Iterable):
        for ref in reference:
            if isinstance(ref, str):
                seeds.extend(_resolve_reference(ref, call))
    if not seeds and isinstance(reference, str):
        seeds.append(reference)
    return seeds


def _resolve_reference(ref: str, call: ToolCall) -> List[str]:
    context = call.args.get("_context") or {}
    result = context.get(ref)
    nodes: List[str] = []
    if isinstance(result, ToolResult):
        for hit in getattr(result, "hits", []) or []:
            node_id = getattr(hit, "node_id", None)
            if isinstance(node_id, str):
                nodes.append(node_id)
        if not nodes and isinstance(getattr(result, "data", None), dict):
            for hit in result.data.get("hits", []):
                node_id = getattr(hit, "node_id", None)
                if isinstance(node_id, str):
                    nodes.append(node_id)
    if not nodes and ref:
        nodes.append(ref)
    return nodes


def _require_manager(manager: Optional[RetrieverManager]) -> RetrieverManager:
    if not isinstance(manager, RetrieverManager):
        raise RuntimeError("structure children tool requires RetrieverManager instance")
    return manager


def _require_graph(graph) -> "DocGraphNavigator":
    from ...planner.doc_graph import DocGraphNavigator

    if not isinstance(graph, DocGraphNavigator):
        raise RuntimeError("structure tool requires DocGraphNavigator")
    return graph


__all__ = ["expand", "children"]
