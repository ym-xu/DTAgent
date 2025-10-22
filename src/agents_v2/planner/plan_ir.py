from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanNode:
    id: str
    op: str
    args: Dict[str, Any]
    save_as: Optional[str] = None
    uses: Optional[List[str]] = None
    when: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStage:
    name: str
    evidence_type: str
    run_if: Optional[str]
    graph: List[PlanNode] = field(default_factory=list)


@dataclass
class PlanIR:
    plan_id: str
    stages: List[PlanStage]
    final: str
    constraints: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def copy_with_meta(self, **kwargs: Any) -> "PlanIR":
        merged = dict(self.meta)
        merged.update(kwargs)
        return PlanIR(
            plan_id=self.plan_id,
            stages=self.stages,
            final=self.final,
            constraints=self.constraints,
            meta=merged,
        )
