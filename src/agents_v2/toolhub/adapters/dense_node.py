"""Dense node search adapter.

复用 RetrieverManager 的 dense_search 能力，逐步迁移到 ToolHub。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from ...retriever.manager import RetrieverManager
from ...schemas import StrategyStep
from ..types import ToolCall, ToolResult


def search(call: ToolCall) -> ToolResult:
    manager = _require_manager(call.args.get("_retriever_manager"))
    step = _require_step(call.args.get("_step"))
    memory = call.args.get("_memory")

    mode = str(call.args.get("_source_tool") or step.tool)
    legacy_tool = call.args.get("_legacy_tool") or "dense_search"
    exec_step = replace(step, tool=legacy_tool)

    hits = manager.execute(exec_step, memory)
    status = "ok" if hits else "empty"
    metrics = {
        "mode": mode,
        "n_hits": len(hits),
        "legacy_tool": legacy_tool,
    }
    return ToolResult(
        status=status,
        data={"hits": hits, "mode": mode},
        metrics=metrics,
    )


def _require_manager(manager: Optional[RetrieverManager]) -> RetrieverManager:
    if not isinstance(manager, RetrieverManager):
        raise RuntimeError("dense_node.search requires RetrieverManager instance")
    return manager


def _require_step(step: Optional[StrategyStep]) -> StrategyStep:
    if not isinstance(step, StrategyStep):
        raise RuntimeError("dense_node.search requires StrategyStep context")
    return step


__all__ = ["search"]
