"""
ToolRegistry
============

简单的工具注册与查找中心。
"""

from __future__ import annotations

from typing import Callable, Dict

from .types import ToolCall, ToolResult

ToolFn = Callable[[ToolCall], ToolResult]


class ToolRegistry:
    """工具函数注册表。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, tool_id: str, fn: ToolFn) -> None:
        if tool_id in self._tools:
            raise ValueError(f"duplicate tool_id: {tool_id}")
        self._tools[tool_id] = fn

    def register_bulk(self, tool_map: Dict[str, ToolFn]) -> None:
        for tool_id, fn in tool_map.items():
            self.register(tool_id, fn)

    def get(self, tool_id: str) -> ToolFn:
        if tool_id not in self._tools:
            raise KeyError(f"tool not found: {tool_id}")
        return self._tools[tool_id]

    def __contains__(self, tool_id: str) -> bool:  # pragma: no cover - simple proxy
        return tool_id in self._tools


__all__ = ["ToolRegistry", "ToolFn"]
