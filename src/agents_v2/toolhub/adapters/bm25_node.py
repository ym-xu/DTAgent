"""BM25 node search adapter.

当前实现直接复用 RetrieverManager 的 dense/sparse/hybrid 检索逻辑，
方便逐步迁移到 ToolHub 执行链。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from ...retriever.manager import RetrieverManager
from ...schemas import StrategyStep
from ..common import (
    common_hit_from_retrieval,
    map_view_to_evidence,
    map_view_to_modality,
)
from ..types import ToolCall, ToolResult


def search(call: ToolCall) -> ToolResult:
    manager = _require_manager(call.args.get("_retriever_manager"))
    step = _require_step(call.args.get("_step"))
    memory = call.args.get("_memory")

    # 允许通过 mode 指定实际执行的检索工具（dense/sparse/hybrid）
    mode = str(call.args.get("_source_tool") or step.tool)
    legacy_tool = call.args.get("_legacy_tool")
    exec_step = replace(step, tool=legacy_tool) if legacy_tool else step

    hits_raw = manager.execute(exec_step, memory)
    view = str(call.args.get("view") or step.args.get("view") or "section#child")
    common_hits = [
        common_hit_from_retrieval(hit, tool_name=legacy_tool or mode, view=view)
        for hit in hits_raw
    ]
    status = "ok" if common_hits else "empty"
    metrics = {
        "mode": mode,
        "n_hits": len(common_hits),
        "legacy_tool": legacy_tool,
    }
    return ToolResult(
        status=status,
        metrics=metrics,
        hits=common_hits,
        info={
            "mode": mode,
            "view": view,
            "evidence_type": map_view_to_evidence(view),
            "modality": map_view_to_modality(view),
        },
    )


def _require_manager(manager: Optional[RetrieverManager]) -> RetrieverManager:
    if not isinstance(manager, RetrieverManager):
        raise RuntimeError("bm25_node.search requires RetrieverManager instance")
    return manager


def _require_step(step: Optional[StrategyStep]) -> StrategyStep:
    if not isinstance(step, StrategyStep):
        raise RuntimeError("bm25_node.search requires StrategyStep context")
    return step


__all__ = ["search"]
