"""
Planner
=======

负责将检索策略转化为具体的检索与观察动作，
并利用 DocTree 图结构扩展邻域节点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .memory import AgentMemory
from .schemas import (
    FallbackSpec,
    FinalSpec,
    PackSpec,
    PlannerAction,
    RetrievalHit,
    RerankSpec,
    RouterDecision,
    StrategyKind,
    StrategyPlan,
    StrategyStage,
    StrategyStep,
)


DEFAULTS = {
    "k_pages": 8,
    "k_nodes": 50,
    "page_window": 1,
    "ctx_tokens": 1500,
    "mmr_lambda": 0.7,
    "coverage_gate": 0.5,
}


@dataclass
class DocGraphNavigator:
    """基于简单邻接表的树导航工具。"""

    children: Dict[str, List[str]] = field(default_factory=dict)
    parents: Dict[str, str] = field(default_factory=dict)
    same_page: Dict[str, List[str]] = field(default_factory=dict)
    siblings: Dict[str, List[str]] = field(default_factory=dict)

    def expand(self, node_id: str, *, include_self: bool = False) -> List[str]:
        """返回与 node_id 相关的邻域节点（去重）。"""
        related: List[str] = []
        seen: Set[str] = set()

        def _push(seq: Iterable[str]) -> None:
            for nid in seq:
                if nid and nid not in seen:
                    seen.add(nid)
                    related.append(nid)

        if include_self:
            _push([node_id])

        _push(self.children.get(node_id, []))
        parent = self.parents.get(node_id)
        if parent:
            _push([parent])
            _push(self.children.get(parent, []))  # siblings via parent
        _push(self.same_page.get(node_id, []))
        _push(self.siblings.get(node_id, []))
        # 去除自身
        if node_id in seen and not include_self:
            related = [nid for nid in related if nid != node_id]
        return related


@dataclass
class Planner:
    """基于策略计划生成动作，并在命中的节点附近扩展。"""

    graph: DocGraphNavigator
    max_observation_neighbors: int = 6

    def plan_from_strategy(self, plan: StrategyPlan) -> List[PlannerAction]:
        actions: List[PlannerAction] = []
        for step in plan.steps:
            actions.append(self._step_to_action(step))
        return actions

    def _step_to_action(self, step: StrategyStep) -> PlannerAction:
        if step.tool in {"jump_to_label", "jump_to_page"}:
            payload = {"tool": step.tool, **step.args}
            action_type = "move"
        else:
            payload = {"tool": step.tool, **step.args}
            action_type = "retrieve"
        return PlannerAction(type=action_type, payload=payload, source_step=step)

    def observation_plan(
        self,
        *,
        hits: List[RetrievalHit],
        memory: AgentMemory,
        include_neighbors: bool = True,
    ) -> List[PlannerAction]:
        """根据检索命中生成观察动作，可扩展邻域节点。"""
        scoring: Dict[str, float] = {}
        order: Dict[str, int] = {}

        def _register(node: str, score: float) -> None:
            if node is None or memory.has_seen(node):
                return
            current = scoring.get(node)
            if current is None or score > current:
                scoring[node] = score
            order.setdefault(node, len(order))

        for hit in hits:
            _register(hit.node_id, hit.score)
            if include_neighbors:
                neighbors = self._neighbors_for(hit.node_id, memory)
                for neighbor in neighbors:
                    _register(neighbor, hit.score * 0.7)

        if not scoring:
            return [PlannerAction(type="noop", payload={"reason": "no-new-targets"})]

        ranked = sorted(scoring.items(), key=lambda item: (-item[1], order[item[0]]))
        selected = [nid for nid, _ in ranked[: self.max_observation_neighbors]]
        return [
            PlannerAction(
                type="observe",
                payload={"nodes": selected},
            )
        ]

    def _neighbors_for(self, node_id: str, memory: AgentMemory) -> List[str]:
        neighbors = self.graph.expand(node_id)
        pruned = [nid for nid in neighbors if not memory.has_seen(nid)]
        return pruned

    # --- Router 接入 ---

    def plan_from_router(
        self,
        decision: RouterDecision,
        *,
        max_calls: int = 8,
    ) -> StrategyPlan:
        """根据 RouterDecision 生成 StrategyPlan。"""

        signals = decision.signals
        risk = decision.risk

        methods = _methods_for(
            decision.query_type,
            need_table=risk.need_table,
            need_chart=risk.need_chart,
            need_visual=risk.need_visual,
        )

        stages = [
            StrategyStage(
                stage="primary",
                methods=methods,
                k_pages=DEFAULTS["k_pages"],
                k_nodes=DEFAULTS["k_nodes"],
                page_window=DEFAULTS["page_window"],
                params={"per_page_limit": 2},
            )
        ]

        steps = _build_steps_from_router(decision, max_calls=max_calls)
        step_count = len(steps)
        strategy_kind = StrategyKind.SINGLE if step_count <= 1 else StrategyKind.COMPOSITE

        rerank = RerankSpec(
            fuse="RRF",
            features=_features_for(signals),
            diversify_by="section",
        )

        pack = PackSpec(
            mmr_lambda=DEFAULTS["mmr_lambda"],
            ctx_tokens=DEFAULTS["ctx_tokens"],
            per_page_limit=2,
            attach=["caption", "table_header"],
        )

        fallbacks = [
            FallbackSpec(condition="coverage<0.5", action={"expand_window": 1}),
            FallbackSpec(condition="low_diversity", action={"diversify_by": "section"}),
        ]
        if risk.need_visual:
            fallbacks.append(
                FallbackSpec(
                    condition="visual_low_conf",
                    action={"enable": "vlm_verify_topk", "topk": 6},
                )
            )

        expected_format = signals.expected_format or "string"
        final = FinalSpec(answer_var="ANS", format=expected_format)

        retrieval_keys = _collect_retrieval_keys(decision, steps)

        return StrategyPlan(
            strategy=strategy_kind,
            steps=steps,
            confidence=decision.confidence,
            thinking=None,
            retrieval_keys=retrieval_keys,
            stages=stages,
            rerank=rerank,
            pack=pack,
            coverage_gate=DEFAULTS["coverage_gate"],
            fallbacks=fallbacks,
            final=final,
        )


def _methods_for(
    query_type: str,
    need_table: bool,
    need_chart: bool,
    need_visual: bool,
) -> List[str]:
    qt = (query_type or "").lower()
    if qt in {"text", "definition", "list"}:
        return ["bm25_node", "dense_node", "toc_anchor"]
    if qt == "table" or need_table:
        base = ["bm25_node", "table_index"]
        if need_chart:
            base.append("chart_index")
        return base
    if qt == "chart" or (need_chart and not need_table):
        return ["bm25_node", "chart_index"]
    if qt in {"visual_count", "visual_presence", "visual_compare", "visual_trend", "metadata_visual"} or need_visual:
        return ["page_locator", "figure_finder", "chart_index"]
    if qt in {"price_total", "numeric_compute", "cross_page", "multi_modal", "hybrid"}:
        return ["bm25_node", "dense_node", "table_index", "chart_index"]
    return ["bm25_node", "dense_node"]


def _features_for(signals) -> List[str]:
    feats = ["toc_distance"]
    if signals.years:
        feats.append("year")
    if signals.units:
        feats.append("unit")
    if signals.page_hint:
        feats.append("page")
    return feats


def _build_steps_from_router(decision: RouterDecision, *, max_calls: int) -> List[StrategyStep]:
    signals = decision.signals
    query_type = (decision.query_type or "").lower()
    steps: List[StrategyStep] = []
    added: Set[Tuple[str, Tuple[Tuple[str, object], ...]]] = set()

    keywords = _compose_keywords(decision)

    def _add_step(
        tool: str,
        args: Dict[str, object],
        *,
        step_id: Optional[str] = None,
        step_type: Optional[str] = None,
        save_as: Optional[str] = None,
        when: Optional[str] = None,
        uses: Optional[List[str]] = None,
    ) -> None:
        if len(steps) >= max_calls:
            return
        key = _step_key(tool, args)
        if key in added:
            return
        added.add(key)
        steps.append(
            StrategyStep(
                tool=tool,
                args=args,
                step_id=step_id,
                step_type=step_type,
                save_as=save_as,
                when=when,
                uses=uses,
            )
        )

    # Page / label hints
    for page in signals.page_hint or []:
        try:
            page_int = int(page)
        except (TypeError, ValueError):
            continue
        _add_step(
            "jump_to_page",
            {"page": page_int},
            step_id=f"page_{page_int}",
            step_type="move",
        )

    for label in (signals.figure_hint or []) + (signals.table_hint or []):
        cleaned = label.strip()
        if not cleaned:
            continue
        _add_step(
            "jump_to_label",
            {"label": cleaned},
            step_id=f"label_{cleaned}",
            step_type="move",
        )

    # Main retrieval
    base_query = decision.query.strip()
    year_phrase = " ".join(str(year) for year in (signals.years or []))
    operations_phrase = " ".join(signals.operations or [])
    object_phrase = " ".join(signals.objects or [])
    units_phrase = " ".join(signals.units or [])

    enriched_query = " ".join(
        part
        for part in [base_query, units_phrase, year_phrase, operations_phrase, object_phrase]
        if part
    ).strip() or base_query

    if query_type in {"text", "definition", "list"}:
        _add_step(
            "dense_search",
            {"query": enriched_query, "view": "section#gist"},
            step_id="dense_primary",
            step_type="search",
        )
        if keywords:
            _add_step(
                "sparse_search",
                {"keywords": keywords[:6], "view": "section#child"},
                step_id="sparse_keywords",
                step_type="search",
            )
        _add_step(
            "dense_search",
            {"query": enriched_query, "view": "section#heading"},
            step_id="dense_heading",
            step_type="search",
        )

    elif query_type == "table":
        if keywords:
            _add_step(
                "sparse_search",
                {"keywords": keywords[:8], "view": "section#child"},
                step_id="table_sparse",
                step_type="search",
            )
        _add_step(
            "dense_search",
            {"query": enriched_query, "view": "table"},
            step_id="table_dense",
            step_type="search",
        )
        _add_step(
            "hybrid_search",
            {"query": enriched_query, "keywords": keywords[:6], "view": "section#gist"},
            step_id="table_hybrid",
            step_type="search",
        )

    elif query_type == "chart":
        _add_step(
            "dense_search",
            {"query": enriched_query, "view": "image"},
            step_id="chart_dense",
            step_type="search",
        )
        if keywords:
            _add_step(
                "sparse_search",
                {"keywords": keywords[:6], "view": "section#child"},
                step_id="chart_sparse",
                step_type="search",
            )

    elif query_type in {"visual_count", "visual_presence", "visual_compare", "visual_trend", "metadata_visual"}:
        visual_query = object_phrase or enriched_query or base_query
        _add_step(
            "dense_search",
            {"query": visual_query, "view": "image"},
            step_id="visual_dense",
            step_type="search",
        )
        if keywords:
            _add_step(
                "sparse_search",
                {"keywords": keywords[:6], "view": "section#child"},
                step_id="visual_sparse",
                step_type="search",
            )

    else:
        _add_step(
            "dense_search",
            {"query": enriched_query, "view": "section#gist"},
            step_id="dense_primary",
            step_type="search",
        )
        if keywords:
            _add_step(
                "sparse_search",
                {"keywords": keywords[:6], "view": "section#child"},
                step_id="sparse_keywords",
                step_type="search",
            )
        if signals.page_hint:
            _add_step(
                "dense_search",
                {"query": enriched_query, "view": "section#heading"},
                step_id="dense_heading",
                step_type="search",
            )
        if query_type in {"numeric_compute", "price_total"} and units_phrase:
            _add_step(
                "sparse_search",
                {"keywords": [units_phrase], "view": "section#child"},
                step_id="unit_sparse",
                step_type="search",
            )

    return steps


def _compose_keywords(decision: RouterDecision) -> List[str]:
    signals = decision.signals
    keywords: List[str] = []
    for container in (
        signals.keywords or [],
        signals.section_cues or [],
        signals.objects or [],
    ):
        for item in container:
            cleaned = item.strip()
            if cleaned:
                keywords.append(cleaned)

    for year in signals.years or []:
        keywords.append(str(year))
    for unit in signals.units or []:
        cleaned = unit.strip()
        if cleaned:
            keywords.append(cleaned)

    seen: Set[str] = set()
    unique: List[str] = []
    for kw in keywords:
        lower = kw.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(kw)
    return unique


def _step_key(tool: str, args: Dict[str, object]) -> Tuple[str, Tuple[Tuple[str, object], ...]]:
    normalized_args: List[Tuple[str, object]] = []
    for key, value in sorted(args.items()):
        if isinstance(value, list):
            normalized_args.append((key, tuple(value)))
        else:
            normalized_args.append((key, value))
    return tool, tuple(normalized_args)


def _collect_retrieval_keys(decision: RouterDecision, steps: List[StrategyStep]) -> List[str]:
    keys: List[str] = []
    seen: Set[str] = set()

    for step in steps:
        query = step.args.get("query")
        if isinstance(query, str):
            trimmed = query.strip()
            if trimmed and trimmed not in seen:
                seen.add(trimmed)
                keys.append(trimmed)
        keywords = step.args.get("keywords")
        if isinstance(keywords, list):
            for kw in keywords:
                if isinstance(kw, str):
                    trimmed = kw.strip()
                    if trimmed and trimmed not in seen:
                        seen.add(trimmed)
                        keys.append(trimmed)

    if decision.query:
        base = decision.query.strip()
        if base and base not in seen:
            keys.insert(0, base)
    return keys[:12]


__all__ = ["DocGraphNavigator", "Planner"]
