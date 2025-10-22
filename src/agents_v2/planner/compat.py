"""
Compatibility Planner wrapper
------------------------------

Bridges existing code (Orchestrator, CLI, tests) that expect a ``Planner``
object returning ``StrategyPlan`` while the new planning stack produces
Plan-IR objects.  Internally this wrapper invokes the LLM-based planner and
translates the resulting Plan-IR into legacy strategy data structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..memory import AgentMemory
from ..schemas import PlannerAction, RetrievalHit, RouterDecision, StrategyPlan, StrategyStep
from .plan_ir import PlanIR
from .planner import plan_from_router as generate_plan_candidates
from .planner import plan_ir_to_strategy_plan
from .selector import select_plan

try:  # optional; in some contexts doc_graph may not be needed
    from .doc_graph import DocGraphNavigator  # type: ignore
except Exception:  # pragma: no cover
    DocGraphNavigator = object  # type: ignore


RETRIEVE_TOOLS = {
    "bm25_node.search",
    "dense_node.search",
    "table_index.search",
    "chart_index.search",
    "page_locator.locate",
    "figure_finder.find_regions",
    "chart_screener.screen",
    "toc_anchor.locate",
    "structure.expand",
    "structure.children",
    "extract.column",
    "extract.table",
    "extract.chart_read_axis",
    "extract.regex",
    "compute.filter",
    "vlm.answer",
}


@dataclass
class Planner:
    selector_mode: str = "first"
    observation_limit: int = 6
    graph: Optional[DocGraphNavigator] = None

    def plan_from_router(self, decision: RouterDecision, *, max_calls: int = 8) -> StrategyPlan:
        plans = generate_plan_candidates(decision, max_candidates=1)
        selected = select_plan(plans, mode=self.selector_mode)
        return plan_ir_to_strategy_plan(selected, decision)

    def plan_from_strategy(self, plan: StrategyPlan) -> List[PlannerAction]:
        return [self._step_to_action(step) for step in plan.steps]

    def observation_plan(
        self,
        *,
        hits: List[RetrievalHit],
        memory: AgentMemory,
        include_neighbors: bool = True,  # noqa: D417, kept for legacy signature
    ) -> List[PlannerAction]:
        unseen: List[str] = []
        for hit in hits:
            if hit.node_id and not memory.has_seen(hit.node_id):
                unseen.append(hit.node_id)
            if len(unseen) >= self.observation_limit:
                break
        if not unseen:
            return [PlannerAction(type="noop", payload={"reason": "no-new-targets"})]
        return [PlannerAction(type="observe", payload={"nodes": unseen})]

    def _step_to_action(self, step: StrategyStep) -> PlannerAction:
        action_type = "retrieve" if step.tool in RETRIEVE_TOOLS else "move"
        payload: Dict[str, object] = {"tool": step.tool, **step.args}
        if step.save_as:
            payload["save_as"] = step.save_as
        if step.when:
            payload["when"] = step.when
        if step.uses:
            payload["uses"] = list(step.uses)
        return PlannerAction(type=action_type, payload=payload, source_step=step)


__all__ = ["Planner", "PlanIR"]
