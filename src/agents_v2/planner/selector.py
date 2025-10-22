from __future__ import annotations

from typing import List

from .plan_ir import PlanIR


def select_plan(plans: List[PlanIR], mode: str = "first") -> PlanIR:
    if not plans:
        raise ValueError("select_plan requires at least one candidate plan")
    if mode != "first":
        # Placeholder for future strategies (probe/parallel, etc.)
        raise NotImplementedError(f"selection mode {mode!r} is not supported yet")
    return plans[0]


__all__ = ["select_plan"]
