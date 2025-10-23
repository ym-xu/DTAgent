"""
工具类型定义
============

统一 ToolCall / ToolResult / Hit 数据结构，规范工具调用输入输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ToolStatus = Literal["ok", "empty", "error"]


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用请求。"""

    tool_id: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommonError:
    """统一错误结构。"""

    error: str
    message: str
    retryable: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommonHit:
    """统一命中结构，围绕 DocTree 节点或其子单元。"""

    node_id: str
    evidence_type: str
    score: float
    provenance: Dict[str, Any] = field(default_factory=dict)
    modality: Optional[str] = None
    affordances: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行结果：命中、错误、统计指标统一暴露。"""

    status: ToolStatus
    hits: List[CommonHit] = field(default_factory=list)
    errors: List[CommonError] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None  # legacy兼容

    @property
    def stats(self) -> Dict[str, Any]:
        """兼容命名：stats 与 metrics 等效。"""
        return self.metrics

    @property
    def data(self) -> Dict[str, Any]:
        """兼容旧接口：统一暴露 hits/errors/stats 等。"""
        base = {
            "hits": self.hits,
            "errors": self.errors,
            "stats": self.metrics,
        }
        if self.info:
            base.update(self.info)
        return base


ToolCallList = List[ToolCall]

__all__ = [
    "ToolStatus",
    "ToolCall",
    "ToolResult",
    "CommonHit",
    "CommonError",
    "ToolCallList",
]
