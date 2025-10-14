"""Extraction adapters (placeholder)."""

from __future__ import annotations

from ..types import ToolCall, ToolResult


def extract(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        data={},
        metrics={},
        error="NOT_IMPLEMENTED",
    )


def regex(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        data={},
        metrics={},
        error="NOT_IMPLEMENTED",
    )


def chart_read_axis(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        data={},
        metrics={},
        error="NOT_IMPLEMENTED",
    )


__all__ = ["extract", "regex", "chart_read_axis"]
