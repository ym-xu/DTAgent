"""BM25 page search adapter (placeholder)."""

from __future__ import annotations

from ..types import CommonError, ToolCall, ToolResult


def search(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        errors=[CommonError(error="NOT_IMPLEMENTED", message="bm25_page.search not implemented", retryable=False)],
        error="NOT_IMPLEMENTED",
    )


__all__ = ["search"]
