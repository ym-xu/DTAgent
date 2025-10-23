"""
ToolExecutor
============

负责调用已注册的工具函数，并在需要时统一记账与重试。
"""

from __future__ import annotations

import time
from typing import Optional

from .registry import ToolRegistry
from .types import CommonError, ToolCall, ToolResult


class ToolExecutor:
    """工具执行器，封装错误处理与性能指标。"""

    def __init__(self, registry: ToolRegistry, max_retries: int = 0) -> None:
        self.registry = registry
        self.max_retries = max_retries

    def run(self, call: ToolCall) -> ToolResult:
        fn = self.registry.get(call.tool_id)
        attempt = 0
        last_error: Optional[str] = None
        while attempt <= self.max_retries:
            t0 = time.time()
            try:
                result = fn(call)
            except Exception as exc:  # pragma: no cover - defensive
                latency_ms = int((time.time() - t0) * 1000)
                last_error = str(exc)
                attempt += 1
                if attempt > self.max_retries:
                    return ToolResult(
                        status="error",
                        errors=[CommonError(error="RUNTIME_ERROR", message=last_error or "runtime error", retryable=False)],
                        metrics={"latency_ms": latency_ms, "attempt": attempt},
                        error=last_error,
                    )
                continue

            result.metrics.setdefault("latency_ms", int((time.time() - t0) * 1000))
            result.metrics.setdefault("attempt", attempt + 1)
            return result

        return ToolResult(
            status="error",
            errors=[CommonError(error="MAX_RETRIES", message=last_error or "unknown error", retryable=False)],
            metrics={"attempt": attempt},
            error=last_error or "UNKNOWN_ERROR",
        )


__all__ = ["ToolExecutor"]
