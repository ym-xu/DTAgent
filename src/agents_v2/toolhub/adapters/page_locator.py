"""Page locator adapter (placeholder)."""

from __future__ import annotations

from ..types import ToolCall, ToolResult


def locate(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        data={},
        metrics={},
        error="NOT_IMPLEMENTED",
    )


__all__ = ["locate"]
