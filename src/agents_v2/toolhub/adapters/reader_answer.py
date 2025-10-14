"""Reader answer adapter (placeholder)."""

from __future__ import annotations

from ..types import ToolCall, ToolResult


def answer(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        data={},
        metrics={},
        error="NOT_IMPLEMENTED",
    )


__all__ = ["answer"]
