"""Reader answer adapter (placeholder)."""

from __future__ import annotations

from ..types import CommonError, ToolCall, ToolResult


def answer(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        errors=[CommonError(error="NOT_IMPLEMENTED", message="reader.answer not implemented", retryable=False)],
        error="NOT_IMPLEMENTED",
    )


__all__ = ["answer"]
