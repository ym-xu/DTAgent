"""
数据结构定义
================

统一定义 Agent 各模块间传递的数据结构，包含检索策略、Planner 动作、
检索结果、观察结果以及 Reasoner 输出等。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class StrategyKind(str, Enum):
    """支持的检索策略类型。"""

    SINGLE = "SINGLE"
    COMPOSITE = "COMPOSITE"


StrategyToolName = Literal[
    "jump_to_label",
    "jump_to_page",
    "dense_search",
    "sparse_search",
    "hybrid_search",
]


@dataclass(frozen=True)
class StrategyStep:
    """单个检索步骤声明。"""

    tool: StrategyToolName
    args: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    step_id: Optional[str] = None
    step_type: Optional[str] = None
    save_as: Optional[str] = None
    when: Optional[str] = None
    uses: Optional[List[str]] = None

    def describe(self) -> str:
        """返回易读描述，便于日志与调试。"""
        kv = ", ".join(f"{k}={v!r}" for k, v in sorted(self.args.items()))
        return f"{self.tool}(weight={self.weight:.2f}, {kv})"


@dataclass(frozen=True)
class StrategyStage:
    """检索阶段配置。"""

    stage: str
    methods: List[str] = field(default_factory=list)
    k_pages: int = 0
    k_nodes: int = 0
    page_window: int = 0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankSpec:
    fuse: str
    features: List[str] = field(default_factory=list)
    diversify_by: Optional[str] = None


@dataclass(frozen=True)
class PackSpec:
    mmr_lambda: float = 0.7
    ctx_tokens: int = 1500
    per_page_limit: int = 2
    attach: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FallbackSpec:
    condition: str
    action: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalSpec:
    answer_var: str
    format: str = "string"


@dataclass(frozen=True)
class StrategyPlan:
    """完整检索策略。"""

    strategy: StrategyKind
    steps: List[StrategyStep] = field(default_factory=list)
    confidence: float = 0.0
    notes: Optional[str] = None
    hints: Optional[List[str]] = None
    thinking: Optional[str] = None
    retrieval_keys: Optional[List[str]] = None
    stages: List[StrategyStage] = field(default_factory=list)
    rerank: Optional[RerankSpec] = None
    pack: Optional[PackSpec] = None
    coverage_gate: float = 0.0
    fallbacks: List[FallbackSpec] = field(default_factory=list)
    final: Optional[FinalSpec] = None

    def is_empty(self) -> bool:
        return len(self.steps) == 0


PlannerActionType = Literal["retrieve", "observe", "move", "noop"]


@dataclass(frozen=True)
class PlannerAction:
    """Planner 输出的动作。"""

    type: PlannerActionType
    payload: Dict[str, Any] = field(default_factory=dict)
    source_step: Optional[StrategyStep] = None

    def require(self, key: str) -> Any:
        """获取 payload 中的必备字段。"""
        if key not in self.payload:
            raise KeyError(f"PlannerAction payload 缺少字段: {key}")
        return self.payload[key]


@dataclass(frozen=True)
class RetrievalHit:
    """检索命中结果。"""

    node_id: str
    score: float
    tool: StrategyToolName
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """观察模块返回的证据项。"""

    node_id: str
    modality: Literal["text", "image", "table", "other"]
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ReasonerAnswer:
    """Reasoner 给出的最终答复。"""

    answer: str
    confidence: float
    support_nodes: List[str]
    reasoning_trace: List[str] = field(default_factory=list)
    thinking: Optional[str] = None
    action: Optional[Literal["REPLAN"]] = None
    missing_intent: Optional[Dict[str, Any]] = None


RouterQueryType = Literal[
    "text",
    "table",
    "chart",
    "visual_count",
    "visual_presence",
    "definition",
    "metadata",
    "numeric_compute",
    "price_total",
    "cross_page",
    "list",
    "visual_compare",
    "reasoning",
    "multi_modal",
    "hybrid",
    "metadata_visual",
    "map_spatial",
    "visual_trend",
]


@dataclass(frozen=True)
class RouterSignals:
    page_hint: List[int] = field(default_factory=list)
    figure_hint: List[str] = field(default_factory=list)
    table_hint: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    expected_format: Optional[str] = None
    section_cues: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    objects_scope: Optional[str] = None


@dataclass(frozen=True)
class RouterRisk:
    ambiguity: float = 0.0
    need_visual: bool = False
    need_table: bool = False
    need_chart: bool = False


@dataclass(frozen=True)
class RouterConstraints:
    allow_unanswerable: bool = False
    must_cite: bool = False


@dataclass(frozen=True)
class RouterDecision:
    """Router 对问题的结构化判定。"""

    query: str
    query_type: str
    signals: RouterSignals = field(default_factory=RouterSignals)
    risk: RouterRisk = field(default_factory=RouterRisk)
    constraints: RouterConstraints = field(default_factory=RouterConstraints)
    confidence: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"type={self.query_type}, confidence={self.confidence:.2f}, "
            f"signals={self.signals}, risk={self.risk}"
        )


__all__ = [
    "StrategyKind",
    "StrategyToolName",
    "StrategyStep",
    "StrategyPlan",
    "StrategyStage",
    "RerankSpec",
    "PackSpec",
    "FallbackSpec",
    "FinalSpec",
    "PlannerAction",
    "RetrievalHit",
    "Observation",
    "ReasonerAnswer",
    "RouterDecision",
    "RouterSignals",
    "RouterRisk",
    "RouterConstraints",
]
