"""Default ToolHub registry helpers."""

from __future__ import annotations

from .registry import ToolRegistry
from .adapters import (
    bm25_node,
    chart_index,
    compute,
    dense_node,
    extract,
    figure_finder,
    chart_screener,
    page_locator,
    table_index,
    vlm_answer,
    pack_mmr,
    judger_verify,
    toc_anchor,
    structure_expand,
)


def build_default_registry() -> ToolRegistry:
    """注册核心检索/跳转工具，供初始迁移使用。"""
    registry = ToolRegistry()
    registry.register("bm25_node.search", bm25_node.search)
    registry.register("dense_node.search", dense_node.search)
    registry.register("page_locator.locate", page_locator.locate)
    registry.register("figure_finder.find_regions", figure_finder.find_regions)
    registry.register("structure.expand", structure_expand.expand)
    registry.register("structure.children", structure_expand.children)
    registry.register("toc_anchor.locate", toc_anchor.locate)
    registry.register("table_index.search", table_index.search)
    registry.register("pack.mmr_knapsack", pack_mmr.mmr_knapsack)
    registry.register("chart_index.search", chart_index.search)
    registry.register("extract.column", extract.column)
    registry.register("extract.chart_read_axis", extract.chart_read_axis)
    registry.register("extract.regex", extract.regex)
    registry.register("compute.filter", compute.filter)
    registry.register("compute.eval", compute.eval)
    registry.register("vlm.answer", vlm_answer.answer)
    registry.register("judger.verify", judger_verify.verify)
    registry.register("chart_screener.screen", chart_screener.screen)
    return registry


__all__ = ["build_default_registry"]
