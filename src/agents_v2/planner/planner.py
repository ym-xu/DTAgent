from __future__ import annotations

from typing import Any, Dict, List

from ..router import RouterDecision
from ..schemas import (
    FinalSpec,
    StrategyKind,
    StrategyPlan,
    StrategyStage,
    StrategyStep,
)
from .plan_ir import PlanIR
from .planner_llm import sample_plans


EVIDENCE_FALLBACK = {"text", "layout", "table", "graphics"}


def plan_from_router(decision: RouterDecision, *, max_candidates: int = 2) -> List[PlanIR]:
    stage_prior = order_stages_from_router(decision)
    plans = sample_plans(decision, stage_prior, k=max_candidates)
    if not plans:
        raise RuntimeError("LLM planner did not return any Plan-IR candidates")
    enriched: List[PlanIR] = []
    for plan in plans:
        meta = dict(plan.meta)
        meta.setdefault("generated_by", "llm")
        meta.setdefault("router_hint", decision.signals.evidence_hint or "unknown")
        meta.setdefault("router_conf", decision.confidence)
        enriched.append(plan.copy_with_meta(**meta))
    return enriched


def order_stages_from_router(decision: RouterDecision) -> List[str]:
    hint = (decision.signals.evidence_hint or "unknown").lower()
    base = {
        "text": ["text", "layout", "table", "graphics"],
        "layout": ["layout", "text", "table", "graphics"],
        "table": ["table", "layout", "text", "graphics"],
        "graphics": ["graphics", "chart", "text", "table"],
        "unknown": ["table", "chart", "text", "graphics"],
    }.get(hint, ["table", "chart", "text", "graphics"])
    return base


def plan_ir_to_strategy_plan(plan: PlanIR, decision: RouterDecision) -> StrategyPlan:
    steps: List[StrategyStep] = []
    stages: List[StrategyStage] = []

    for stage in plan.stages:
        stage_methods: List[str] = []
        stage_step_ids: List[str] = []
        evidence_type = stage.evidence_type.lower() if stage.evidence_type else "text"
        if evidence_type not in EVIDENCE_FALLBACK:
            evidence_type = "graphics" if "chart" in evidence_type else "text"
        for node in stage.graph:
            step = StrategyStep(
                tool=node.op,
                args=node.args,
                step_id=node.id,
                save_as=node.save_as,
                when=node.when,
                uses=node.uses,
                meta=dict(node.meta),
            )
            steps.append(step)
            stage_methods.append(node.op)
            stage_step_ids.append(node.id)
        stages.append(
            StrategyStage(
                stage=stage.name,
                methods=stage_methods,
                k_pages=0,
                k_nodes=0,
                page_window=0,
                params={},
                run_if=stage.run_if,
                step_ids=stage_step_ids,
                evidence_type=evidence_type,
            )
        )

    strategy_kind = StrategyKind.SINGLE if len(stages) <= 1 else StrategyKind.COMPOSITE
    final_spec = FinalSpec(
        answer_var=plan.final or "reasoner",
        format=str(plan.constraints.get("format", "string")),
    )

    return StrategyPlan(
        strategy=strategy_kind,
        steps=steps,
        confidence=decision.confidence,
        notes=None,
        hints=None,
        thinking=None,
        retrieval_keys=[decision.query] if decision.query else None,
        stages=stages,
        rerank=None,
        pack=None,
        coverage_gate=0.0,
        fallbacks=[],
        final=final_spec,
    )


__all__ = [
    "plan_from_router",
    "order_stages_from_router",
    "plan_ir_to_strategy_plan",
]
