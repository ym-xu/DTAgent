"""Judger verify adapter (placeholder)."""

from __future__ import annotations

from ..types import CommonError, ToolCall, ToolResult


def verify(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        errors=[CommonError(error="NOT_IMPLEMENTED", message="judger.verify not implemented", retryable=False)],
        error="NOT_IMPLEMENTED",
    )


__all__ = ["verify"]
