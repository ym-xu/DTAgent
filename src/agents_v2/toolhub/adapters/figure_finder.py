"""Figure finder adapter (placeholder)."""

from __future__ import annotations

from ..types import ToolCall, ToolResult


def find_regions(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        data={},
        metrics={},
        error="NOT_IMPLEMENTED",
    )


__all__ = ["find_regions"]
