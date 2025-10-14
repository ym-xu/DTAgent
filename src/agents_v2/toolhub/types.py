"""
工具类型定义
============

统一 ToolCall / ToolResult / Hit 数据结构，规范工具调用输入输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict


ToolStatus = Literal["ok", "empty", "error"]


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用请求。"""

    tool_id: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行结果。"""

    status: ToolStatus
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Hit(TypedDict, total=False):
    """统一命中结构。"""

    eid: str
    doc_id: str
    page_idx: Optional[int]
    node_id: Optional[str]
    modality: Optional[str]
    snippet: Optional[str]
    bbox: Optional[List[float]]
    score_raw: Optional[float]
    method: Optional[str]
    extra: Dict[str, Any]


ToolCallList = List[ToolCall]

__all__ = [
    "ToolStatus",
    "ToolCall",
    "ToolResult",
    "Hit",
    "ToolCallList",
]
