"""Table index adapter.

复用 RetrieverManager 的 dense/sparse 检索，为表格查询提供统一入口。
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from ...retriever.manager import RetrieverManager
from ...schemas import RetrievalHit, StrategyStep
from ..types import ToolCall, ToolResult


def search(call: ToolCall) -> ToolResult:
    manager = _require_manager(call.args.get("_retriever_manager"))
    step = _require_step(call.args.get("_step"))
    memory = call.args.get("_memory")

    query = _clean_str(call.args.get("query"))
    column_hints = _to_str_list(call.args.get("column_hints"))
    keywords = _to_str_list(call.args.get("keywords"))
    filters = call.args.get("filters") if isinstance(call.args.get("filters"), dict) else {}

    queries: List[str] = []
    if query:
        queries.append(query)
    for hint in column_hints:
        if hint not in queries:
            queries.append(hint)
    for kw in keywords:
        if kw not in queries:
            queries.append(kw)

    dense_args = {
        "query": queries[0] if queries else query,
        "queries": queries[1:] if len(queries) > 1 else None,
        "view": "table",
    }
    dense_args = {k: v for k, v in dense_args.items() if v}
    dense_step = replace(step, tool="dense_search", args=dense_args)
    hits = manager.execute(dense_step, memory)

    annotated_hits = []
    for hit in hits:
        metadata = dict(hit.metadata)
        if column_hints:
            metadata["column_hints"] = column_hints
        if filters:
            metadata["filters"] = filters
        annotated_hits.append(
            RetrievalHit(
                node_id=hit.node_id,
                score=hit.score,
                tool="table_index.search",
                metadata=metadata,
            )
        )

    status = "ok" if annotated_hits else "empty"
    metrics = {
        "n_hits": len(annotated_hits),
        "n_queries": len(queries),
    }
    return ToolResult(
        status=status,
        data={"hits": annotated_hits, "queries": queries, "filters": filters},
        metrics=metrics,
    )


def _require_manager(manager: Optional[RetrieverManager]) -> RetrieverManager:
    if not isinstance(manager, RetrieverManager):
        raise RuntimeError("table_index.search requires RetrieverManager instance")
    return manager


def _require_step(step: Optional[StrategyStep]) -> StrategyStep:
    if not isinstance(step, StrategyStep):
        raise RuntimeError("table_index.search requires StrategyStep context")
    return step


def _clean_str(value) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _to_str_list(value) -> List[str]:
    items: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    items.append(cleaned)
    return items


__all__ = ["search"]
