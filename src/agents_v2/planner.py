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
    RouterSignals,
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

VISUAL_QUERY_TYPES = {
    "visual_count",
    "visual_presence",
    "visual_compare",
    "visual_trend",
    "metadata_visual",
}

VISUAL_KEYWORDS = {"image", "photo", "map", "figure"}
TEXT_GUARD_UNITS = {"%", "percent", "percentage"}


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
        candidates = _normalize_candidates(decision)
        ordered_candidates = reorder_with_textual_guard(decision, candidates)

        stages: List[StrategyStage] = []
        steps: List[StrategyStep] = []
        remaining_calls = max_calls

        if not ordered_candidates:
            ordered_candidates = [(decision.query_type, decision.confidence)]

        for idx, (qt, score) in enumerate(ordered_candidates[:2]):
            if remaining_calls <= 0:
                break
            stage_name = f"{qt}_stage_{idx + 1}"
            stage_suffix = f"_s{idx + 1}"
            stage_steps = _build_steps_from_router(
                decision,
                max_calls=remaining_calls,
                query_type_override=qt,
                step_suffix=stage_suffix,
            )
            if not stage_steps:
                continue
            steps.extend(stage_steps)
            remaining_calls = max(0, max_calls - len(steps))

            stage_step_ids = [step.step_id for step in stage_steps if step.step_id]
            methods = _methods_for(
                qt,
                need_table=risk.need_table or qt == "table",
                need_chart=risk.need_chart or qt == "chart",
                need_visual=risk.need_visual or qt in VISUAL_QUERY_TYPES,
            )
            run_if = None if idx == 0 else "prev_quality < 0.75"
            stage = StrategyStage(
                stage=stage_name,
                methods=methods,
                k_pages=DEFAULTS["k_pages"],
                k_nodes=DEFAULTS["k_nodes"],
                page_window=DEFAULTS["page_window"],
                params={"per_page_limit": 2, "candidate_score": score},
                run_if=run_if,
                step_ids=stage_step_ids,
            )
            stages.append(stage)

        if not steps:
            steps = _build_steps_from_router(decision, max_calls=max_calls)
            stages = [
                StrategyStage(
                    stage="primary",
                    methods=_methods_for(
                        decision.query_type,
                        need_table=risk.need_table,
                        need_chart=risk.need_chart,
                        need_visual=risk.need_visual,
                    ),
                    k_pages=DEFAULTS["k_pages"],
                    k_nodes=DEFAULTS["k_nodes"],
                    page_window=DEFAULTS["page_window"],
                    params={"per_page_limit": 2},
                )
            ]

        step_count = len(steps)
        strategy_kind = StrategyKind.SINGLE if len(stages) <= 1 and step_count <= 1 else StrategyKind.COMPOSITE

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
    if qt in {"table", "price_total", "numeric_compute"} or need_table:
        base = ["table_index.search", "extract.column", "compute.filter"]
        if need_chart:
            base.append("chart_index.search")
        return base
    if qt in {"chart"} or need_chart:
        return ["chart_index.search", "extract.chart_read_axis"]
    if qt in VISUAL_QUERY_TYPES or need_visual:
        return ["page_locator.locate", "figure_finder.find_regions", "vlm.answer"]
    # 默认文本/混合任务
    return ["bm25_node.search", "dense_node.search"]


def _features_for(signals) -> List[str]:
    feats = ["toc_distance"]
    if signals.years:
        feats.append("year")
    if signals.units:
        feats.append("unit")
        if signals.page_hint:
            feats.append("page")
        return feats


def _build_steps_from_router(
    decision: RouterDecision,
    *,
    max_calls: int,
    query_type_override: Optional[str] = None,
    step_suffix: str = "",
) -> List[StrategyStep]:
    signals = decision.signals
    query_type = (query_type_override or decision.query_type or "").lower()
    steps: List[StrategyStep] = []
    added: Set[Tuple[str, Tuple[Tuple[str, object], ...]]] = set()

    keywords = _compose_keywords(decision)
    base_query = decision.query.strip()
    year_phrase = " ".join(str(year) for year in (signals.years or []))
    operations_phrase = " ".join(signals.operations or [])
    object_phrase = " ".join(signals.objects or [])
    units_phrase = " ".join(signals.units or [])
    enriched_query = " ".join(
        part for part in [base_query, units_phrase, year_phrase, operations_phrase, object_phrase] if part
    ).strip() or base_query
    suffix = step_suffix
    table_like_types = {"table", "numeric_compute", "price_total"}
    chart_like_types = {"chart"}

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
        suffixed_step_id = f"{step_id}{step_suffix}" if step_id else None
        suffixed_save = f"{save_as}{step_suffix}" if save_as else save_as
        suffixed_uses = [f"{use}{step_suffix}" for use in uses] if uses else None
        steps.append(
            StrategyStep(
                tool=tool,
                args=args,
                step_id=suffixed_step_id,
                step_type=step_type,
                save_as=suffixed_save,
                when=when,
                uses=suffixed_uses,
            )
        )

    def _build_text_steps() -> None:
        _add_step(
            "dense_node.search",
            {"query": enriched_query, "view": "section#gist"},
            step_id="dense_primary",
            step_type="search",
        )
        if keywords:
            _add_step(
                "bm25_node.search",
                {"keywords": keywords[:8], "view": "section#child"},
                step_id="sparse_keywords",
                step_type="search",
            )
        _add_step(
            "dense_node.search",
            {"query": enriched_query, "view": "section#heading"},
            step_id="dense_heading",
            step_type="search",
        )

    def _build_table_steps() -> None:
        table_step_base = "table_search"
        table_step_id = f"{table_step_base}{suffix}"
        column_hints = _collect_column_hints(signals, keywords)
        value_hints = _collect_value_hints(signals, keywords)
        filters = _collect_filters(signals)
        _add_step(
            "table_index.search",
            {
                "query": enriched_query,
                "column_hints": column_hints[:6],
                "keywords": keywords[:8],
                "filters": filters or None,
            },
            step_id=table_step_base,
            step_type="search",
            save_as="table_hits",
        )
        extract_step_base = "table_extract"
        extract_args: Dict[str, object] = {
            "source": table_step_id,
            "value_hints": value_hints[:6],
            "label_hints": (signals.table_hint or [])[:6],
            "unit": signals.units[0] if signals.units else None,
            "years": signals.years[:6],
        }
        _add_step(
            "extract.column",
            {k: v for k, v in extract_args.items() if v},
            step_id=extract_step_base,
            step_type="extract",
            save_as="table_rows",
        )
        threshold = _safe_number(signals.threshold)
        comparator = _normalize_comparator(signals.comparator)
        if threshold is not None and comparator:
            _add_step(
                "compute.filter",
                {
                    "source": f"{extract_step_base}{suffix}",
                    "threshold": threshold,
                    "comparator": comparator,
                    "unit": signals.units[0] if signals.units else None,
                },
                step_id="table_filter",
                step_type="compute",
            )

    def _build_chart_steps() -> None:
        chart_step_base = "chart_search"
        chart_step_id = f"{chart_step_base}{suffix}"
        _add_step(
            "chart_index.search",
            {
                "query": enriched_query,
                "keywords": keywords[:8],
                "units": signals.units[:4],
                "years": signals.years[:4],
            },
            step_id=chart_step_base,
            step_type="search",
            save_as="chart_hits",
        )
        _add_step(
            "extract.chart_read_axis",
            {"source": chart_step_id},
            step_id="chart_axis",
            step_type="extract",
        )

    def _build_visual_steps() -> None:
        pages = _normalize_page_hints(signals.page_hint)
        figures = signals.figure_hint or []
        page_step_base = "page_loc"
        page_step_id = f"{page_step_base}{suffix}"
        if pages:
            _add_step(
                "page_locator.locate",
                {"pages": pages},
                step_id=page_step_base,
                step_type="move",
                save_as="page_records",
            )
        figure_step_base = "figure_regions"
        figure_step_id = f"{figure_step_base}{suffix}"
        figure_args: Dict[str, object] = {}
        if pages:
            figure_args["pages"] = pages
        if pages or figures:
            figure_args["source"] = page_step_id if pages else None
        if figures:
            figure_args.setdefault("labels", figures)
        figure_args["kinds"] = ["image"]
        _add_step(
            "figure_finder.find_regions",
            {k: v for k, v in figure_args.items() if v},
            step_id=figure_step_base,
            step_type="extract",
            uses=[page_step_base] if pages else None,
            save_as="figure_regions",
        )
        vlm_args: Dict[str, object] = {
            "question": decision.query,
        }
        if pages:
            vlm_args["page_hint"] = list(pages)
        if signals.objects:
            vlm_args["objects"] = list(signals.objects)
        if figures:
            vlm_args["figure_hint"] = list(figures)
        _add_step(
            "vlm.answer",
            vlm_args,
            step_id="vlm_answer",
            step_type="vision",
            uses=[figure_step_base] + ([page_step_base] if pages else []),
        )

    if query_type in VISUAL_QUERY_TYPES:
        _build_visual_steps()
    elif query_type in table_like_types:
        _build_table_steps()
    elif query_type in chart_like_types:
        _build_chart_steps()
    else:
        _build_text_steps()

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
        normalized_args.append((key, _to_hashable(value)))
    return tool, tuple(normalized_args)


def _to_hashable(value: object) -> object:
    if isinstance(value, list):
        return tuple(_to_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _to_hashable(v)) for k, v in sorted(value.items()))
    return value


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


def _collect_column_hints(signals: RouterSignals, keywords: List[str]) -> List[str]:
    hints: List[str] = []
    for container in (
        signals.table_hint or [],
        signals.section_cues or [],
        signals.objects or [],
    ):
        for item in container:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned and cleaned not in hints:
                    hints.append(cleaned)
    for kw in keywords:
        if kw not in hints:
            hints.append(kw)
    return hints


def _collect_value_hints(signals: RouterSignals, keywords: List[str]) -> List[str]:
    hints: List[str] = []
    for unit in signals.units or []:
        cleaned = unit.strip()
        if cleaned and cleaned not in hints:
            hints.append(cleaned)
    for op in signals.operations or []:
        cleaned = op.strip()
        if cleaned and cleaned not in hints:
            hints.append(cleaned)
    for kw in keywords:
        if kw not in hints:
            hints.append(kw)
    return hints


def _collect_filters(signals: RouterSignals) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    if signals.units:
        payload["unit"] = signals.units[0]
    if signals.years:
        payload["years"] = list(signals.years)
    comparator = _normalize_comparator(signals.comparator)
    threshold = _safe_number(signals.threshold)
    if comparator and threshold is not None:
        payload["comparator"] = comparator
        payload["threshold"] = threshold
    return {k: v for k, v in payload.items() if v not in (None, [], {})}


def _safe_number(value) -> Optional[float]:
    try:
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if cleaned.endswith("%"):
                cleaned = cleaned[:-1]
            return float(cleaned)
        if isinstance(value, (int, float)):
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def _normalize_comparator(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text == "=":
        text = "=="
    if text in {">", ">=", "<", "<=", "==", "!="}:
        return text
    return None


def _normalize_page_hints(hints: Optional[List[int]]) -> List[int]:
    normalized: List[int] = []
    if not hints:
        return normalized
    for item in hints:
        try:
            page = int(item)
        except (TypeError, ValueError):
            continue
        if page < 0:
            continue
        if page not in normalized:
            normalized.append(page)
    return normalized


def _normalize_candidates(decision: RouterDecision) -> List[Tuple[str, float]]:
    ordered: List[Tuple[str, float]] = []
    seen: Set[str] = set()

    for qt, score in decision.candidates:
        if not qt:
            continue
        normalized = qt.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        ordered.append((normalized, max(0.0, min(1.0, score_value))))

    primary = (decision.query_type or "").strip().lower()
    if primary and primary not in seen:
        try:
            primary_conf = float(decision.confidence)
        except (TypeError, ValueError):
            primary_conf = 0.0
        ordered.insert(0, (primary, max(0.0, min(1.0, primary_conf))))
        seen.add(primary)

    def _ensure_candidate(kind: str, score: float) -> None:
        if kind and kind not in seen:
            seen.add(kind)
            ordered.append((kind, max(0.0, min(1.0, score))))

    textual_intent = bool(decision.signals.units or decision.signals.operations or decision.signals.table_hint)
    chart_intent = bool(decision.signals.figure_hint) or any(
        kw for kw in (decision.signals.keywords or []) if "chart" in kw.lower() or "graph" in kw.lower()
    )
    if textual_intent:
        fallback_score = max(decision.confidence * 0.6, 0.35)
        _ensure_candidate("table", fallback_score)
    if chart_intent and "chart" not in seen:
        fallback_score = max(decision.confidence * 0.5, 0.3)
        _ensure_candidate("chart", fallback_score)
    if decision.risk.need_visual and "visual_count" not in seen and "visual_presence" not in seen:
        _ensure_candidate(decision.query_type.strip().lower() if decision.query_type else "visual_count", decision.confidence * 0.5)
    return ordered


def reorder_with_textual_guard(
    decision: RouterDecision,
    order: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    if not order:
        return order

    signals = decision.signals
    normalized = list(order)
    units = {str(unit).strip().lower() for unit in signals.units or [] if str(unit).strip()}
    has_percentage = any(unit in TEXT_GUARD_UNITS for unit in units)
    numerical_ops = {op.strip().lower() for op in (signals.operations or []) if isinstance(op, str)}
    textual_intent = bool(units or numerical_ops or signals.table_hint)

    question_text = decision.query.lower() if isinstance(decision.query, str) else ""
    signal_tokens = [token.lower() for token in (signals.keywords or []) if isinstance(token, str)]
    signal_tokens += [token.lower() for token in (signals.section_cues or []) if isinstance(token, str)]
    signal_tokens += [token.lower() for token in (signals.objects or []) if isinstance(token, str)]
    visual_terms = set(signal_tokens)
    if question_text:
        visual_terms.add(question_text)
    strong_visual = any(vk in token for token in visual_terms for vk in VISUAL_KEYWORDS)

    def _pop_candidate(kind: str) -> Optional[Tuple[str, float]]:
        for idx, item in enumerate(normalized):
            if item[0] == kind:
                return normalized.pop(idx)
        return None

    if textual_intent and has_percentage:
        table_candidate = _pop_candidate("table")
        if table_candidate:
            insert_idx = 1 if strong_visual and normalized else 0
            normalized.insert(insert_idx, table_candidate)

    chart_intent = decision.risk.need_chart or any("chart" in token for token in signal_tokens)
    if chart_intent:
        chart_candidate = _pop_candidate("chart")
        if chart_candidate:
            insert_idx = 1 if normalized and normalized[0][0] == "table" else 0
            normalized.insert(insert_idx, chart_candidate)

    return normalized


__all__ = ["DocGraphNavigator", "Planner"]
