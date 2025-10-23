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

    table_stage_context: Optional[Dict[str, Any]] = None

    for stage in plan.stages:
        stage_params: Dict[str, Any] = dict(stage.params or {})
        stage_methods: List[str] = []
        stage_step_ids: List[str] = []
        evidence_type = stage.evidence_type.lower() if stage.evidence_type else "text"
        if evidence_type not in EVIDENCE_FALLBACK:
            evidence_type = "graphics" if "chart" in evidence_type else "text"
        captured_steps: List[StrategyStep] = []
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
            captured_steps.append(step)

        pack_tool = "pack.mmr_knapsack"
        pack_disabled = bool(stage_params.get("disable_pack"))
        pack_present = pack_tool in stage_methods
        if stage_step_ids and not pack_disabled and not pack_present:
            pack_cfg_raw = stage_params.get("pack") if isinstance(stage_params.get("pack"), dict) else {}
            pack_args: Dict[str, Any] = dict(pack_cfg_raw)
            pack_sources = pack_args.get("source") or pack_args.get("sources")
            if isinstance(pack_sources, list):
                pack_args["source"] = pack_sources
            else:
                pack_args["source"] = list(stage_step_ids)
            default_limit = _to_int(stage_params.get("k_nodes"), 40)
            if default_limit <= 0:
                default_limit = 40
            pack_args.setdefault("limit", default_limit)
            pack_args.setdefault("per_page_limit", _to_int(stage_params.get("per_page_limit"), 2) or 2)
            pack_args.setdefault("mmr_lambda", stage_params.get("mmr_lambda") or 0.72)
            pack_args.setdefault("coverage_target", stage_params.get("k_nodes") or pack_args["limit"])
            pack_save_as = pack_args.pop("save_as", None)
            pack_step_id = f"{stage.name}_PACK".replace(" ", "_")
            steps.append(
                StrategyStep(
                    tool=pack_tool,
                    args=pack_args,
                    step_id=pack_step_id,
                    save_as=pack_save_as,
                    uses=list(stage_step_ids),
                    meta={"auto_pack": True},
                )
            )
            stage_methods.append(pack_tool)
            stage_step_ids.append(pack_step_id)

        stages.append(
            StrategyStage(
                stage=stage.name,
                methods=stage_methods,
                k_pages=_to_int(stage_params.get("k_pages"), 0),
                k_nodes=_to_int(stage_params.get("k_nodes"), 0),
                page_window=_to_int(stage_params.get("page_window"), 0),
                params=stage_params,
                run_if=stage.run_if,
                step_ids=stage_step_ids,
                evidence_type=evidence_type,
            )
        )

        if evidence_type == "table" and table_stage_context is None:
            table_stage_context = {
                "stage": stage,
                "stage_params": stage_params,
                "steps": captured_steps,
                "stage_name": stage.name,
                "stage_step_ids": list(stage_step_ids),
            }

    strategy_kind = StrategyKind.SINGLE if len(stages) <= 1 else StrategyKind.COMPOSITE
    final_spec = FinalSpec(
        answer_var=plan.final or "reasoner",
        format=str(plan.constraints.get("format", "string")),
    )

    # Auto append chart fallback if only table stage provided
    if table_stage_context and len(plan.stages) == 1:
        fallback_steps, fallback_stage = _build_chart_fallback(
            decision,
            table_stage_context,
        )
        for step in fallback_steps:
            steps.append(step)
        stages.append(fallback_stage)

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


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_chart_fallback(
    decision: RouterDecision,
    context: Dict[str, Any],
) -> tuple[List[StrategyStep], StrategyStage]:
    stage_name = context.get("stage_name", "table_stage")
    base_id = stage_name.upper()
    search_step_id = f"{base_id}_CHART"
    figure_step_id = f"{base_id}_FIGURE"
    screen_step_id = f"{base_id}_SCREEN"
    vlm_step_id = f"{base_id}_VLM"

    table_steps: List[StrategyStep] = context.get("steps", [])
    keywords: List[str] = []
    for step in table_steps:
        if isinstance(step.args, dict):
            keys = step.args.get("keys")
            if isinstance(keys, dict):
                for value in keys.values():
                    if isinstance(value, str):
                        keywords.append(value)
                    elif isinstance(value, list):
                        keywords.extend([str(v) for v in value if isinstance(v, (str, int))])

    chart_args = {
        "keys": {
            "keywords": list({kw for kw in keywords if isinstance(kw, str)}),
            "entities": list({kw for kw in keywords if isinstance(kw, str)})[:6],
            "years": decision.signals.years or None,
        },
        "filters": {
            "page_idx": (decision.signals.page_hint[0] if decision.signals.page_hint else None),
        },
    }
    chart_args["keys"] = {k: v for k, v in chart_args["keys"].items() if v}
    chart_args["filters"] = {k: v for k, v in chart_args["filters"].items() if v is not None}

    fallback_steps = [
        StrategyStep(
            tool="chart_index.search",
            args=chart_args,
            step_id=search_step_id,
            meta={"auto_fallback": True},
        ),
        StrategyStep(
            tool="figure_finder.find_regions",
            args={"from": search_step_id},
            step_id=figure_step_id,
            uses=[search_step_id],
            meta={"auto_fallback": True},
        ),
        StrategyStep(
            tool="chart_screener.screen",
            args={"source": figure_step_id},
            step_id=screen_step_id,
            uses=[figure_step_id],
            meta={"auto_fallback": True},
        ),
        StrategyStep(
            tool="vlm.answer",
            args={"question": decision.query},
            step_id=vlm_step_id,
            uses=[figure_step_id],
            meta={"auto_fallback": True},
        ),
    ]

    fallback_stage = StrategyStage(
        stage=f"{stage_name}_chart_fallback",
        methods=[step.tool for step in fallback_steps],
        k_pages=0,
        k_nodes=1,
        page_window=0,
        params={"coverage_gate": 0.0},
        run_if="prev_coverage < 0.5",
        step_ids=[step.step_id for step in fallback_steps],
        evidence_type="graphics",
    )
    return fallback_steps, fallback_stage
