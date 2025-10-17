"""
AgentOrchestrator
=================

协调问题处理、策略判定、规划、检索、观察与推理模块的主流程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .memory import AgentMemory
from .schemas import (
    PlannerAction,
    Observation,
    ReasonerAnswer,
    RouterDecision,
    FinalSpec,
    StrategyPlan,
    StrategyStage,
    StrategyStep,
    RetrievalHit,
)
from .toolhub import ToolCall, ToolExecutor, ToolResult
from .judger import LLMJudger


class MissingDependencyError(RuntimeError):
    """缺少依赖时抛出的异常。"""


@dataclass
class AgentConfig:
    """Agent 运行时配置。"""

    max_iterations: int = 4
    accept_threshold: float = 0.75


@dataclass
class StageOutcome:
    """单个阶段执行结果，用于质量评估与最终裁决。"""

    stage: StrategyStage
    hits: List[RetrievalHit] = field(default_factory=list)
    answer: Optional[ReasonerAnswer] = None
    quality: float = 0.0
    judger_pass: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    verdict: Dict[str, object] = field(default_factory=dict)


class AgentOrchestrator:
    """协调新一代 Agent 流程的顶层对象。"""

    def __init__(
        self,
        *,
        router,
        planner,
        retriever_manager,
        observer,
        reasoner,
        strategy_planner=None,
        llm_judger: Optional[LLMJudger] = None,
        tool_executor: Optional[ToolExecutor] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.memory = AgentMemory()
        self.router = router
        self.strategy_planner = strategy_planner
        self.planner = planner
        self.retriever_manager = retriever_manager
        self.observer = observer
        self.reasoner = reasoner
        self.llm_judger = llm_judger
        self.tool_executor = tool_executor
        self.last_answer: Optional[ReasonerAnswer] = None

    # --- 流程方法 ---

    def preprocess_question(self, question: str) -> str:
        """问题预处理：当前仅做去空格记录。"""
        normalized = question.strip()
        self.memory.record_question(normalized)
        return normalized

    def route_question(self, question: str):
        if not self.router:
            raise MissingDependencyError("router is missing")
        decision = self.router.route(question)
        self.memory.push_router(decision)
        return decision

    def decide_strategy(self, question: str) -> StrategyPlan:
        if not self.strategy_planner:
            raise MissingDependencyError("strategy_planner is missing")
        plan = self.strategy_planner.decide(question)
        self.memory.push_strategy(plan)
        return plan

    def run(self, question: str) -> ReasonerAnswer:
        return self.run_with_callback(question)

    def run_with_callback(
        self,
        question: str,
        *,
        on_plan: Optional[Callable[[StrategyPlan], None]] = None,
    ) -> ReasonerAnswer:
        """执行 Agent 主循环。"""
        normalized = self.preprocess_question(question)
        decision = self.route_question(normalized)
        plan = self._plan_from_router(decision, normalized)
        if on_plan and not plan.is_empty():
            on_plan(plan)

        for iteration in range(self.config.max_iterations):
            self.memory.next_iteration()
            outcomes, need_replan = self._run_plan_stages(plan, normalized)
            if need_replan:
                if iteration + 1 >= self.config.max_iterations:
                    break
                plan = self._plan_from_router(decision, normalized)
                if on_plan and not plan.is_empty():
                    on_plan(plan)
                continue

            if outcomes:
                accepted = next(
                    (
                        outcome
                        for outcome in outcomes
                        if outcome.answer
                        and outcome.judger_pass
                        and outcome.quality >= self.config.accept_threshold
                    ),
                    None,
                )
                target = accepted or max(
                    outcomes,
                    key=lambda o: (
                        1 if (o.judger_pass and o.answer) else 0,
                        o.quality,
                    ),
                )
                if target.answer:
                    self.last_answer = target.answer
                    return target.answer

            if iteration + 1 >= self.config.max_iterations:
                break
            plan = self._plan_from_router(decision, normalized)
            if on_plan and not plan.is_empty():
                on_plan(plan)

        final_answer = ReasonerAnswer(
            answer="Insufficient information to reach a conclusion",
            confidence=0.0,
            support_nodes=[],
            reasoning_trace=["Exceeded iteration loop"],
        )
        self.last_answer = final_answer
        return final_answer

    # --- 内部辅助 ---
    def _run_plan_stages(
        self,
        plan: StrategyPlan,
        question: str,
    ) -> Tuple[List[StageOutcome], bool]:
        if not self.planner or not self.retriever_manager:
            raise MissingDependencyError("planner or retriever_manager is missing")

        actions = self.planner.plan_from_strategy(plan)
        stages = plan.stages or [StrategyStage(stage="primary")]

        outcomes: List[StageOutcome] = []
        ctx: Dict[str, float] = {}
        need_replan = False

        for stage in stages:
            if not self._should_run_stage(stage, ctx):
                continue

            stage_actions = self._actions_for_stage(actions, stage)
            if not stage_actions:
                continue

            hits = self._execute_actions(stage_actions)
            outcome = StageOutcome(stage=stage, hits=hits)
            self._run_observers(hits)

            answer = self.reasoner.run(question, self.memory.snapshot())
            outcome.answer = answer

            if answer.action == "REPLAN":
                need_replan = True
                outcomes.append(outcome)
                break

            verdict = self._judger_verify(answer, plan, stage)
            outcome.verdict = verdict or {}

            if verdict and not verdict.get("pass", False):
                recommendation = verdict.get("recommendation", {})
                issues = verdict.get("issues", [])
                if recommendation.get("action") == "return_unanswerable":
                    detail_lines: List[str] = []
                    if "threshold_fail" in issues and verdict.get("details", {}).get("inequality"):
                        mismatch = verdict["details"]["inequality"]
                        detail_lines.append(
                            "No supporting value satisfies the required threshold "
                            f"({mismatch.get('value')} vs {mismatch.get('comparator')} {mismatch.get('threshold')})"
                        )
                    if "format_mismatch" in issues:
                        detail_lines.append("Answer format does not match expected output type")
                    if "unit_mismatch" in issues:
                        detail_lines.append("Units in answer/evidence differ from requested units")
                    explanation = verdict.get("explanation")
                    if explanation:
                        detail_lines.append(f"Judger explanation: {explanation}")
                    if not detail_lines:
                        detail_lines.append("Judger issues: " + ", ".join(issues))
                    outcome.answer = ReasonerAnswer(
                        answer="Insufficient information to satisfy the query constraints",
                        confidence=0.0,
                        support_nodes=answer.support_nodes,
                        reasoning_trace=detail_lines,
                    )
                elif recommendation.get("action") in {"replan", "redo_answer"}:
                    need_replan = True
                    outcomes.append(outcome)
                    break

            judger_pass = verdict.get("pass", False) if verdict else self._heuristic_judger(outcome.answer)
            outcome.judger_pass = judger_pass
            metrics = self._compute_quality_metrics(plan, stage, hits, outcome.answer, verdict)
            outcome.metrics = metrics
            outcome.quality = self._compute_quality_score(metrics, judger_pass)
            outcomes.append(outcome)

            ctx["prev_quality"] = outcome.quality
            ctx["prev_judger_pass"] = 1.0 if judger_pass else 0.0

            if judger_pass and outcome.quality >= self.config.accept_threshold:
                break

        return outcomes, need_replan

    def _plan_from_router(
        self,
        decision: RouterDecision,
        question: str,
    ) -> StrategyPlan:
        plan = self.planner.plan_from_router(decision)
        if not plan.is_empty():
            self.memory.push_strategy(plan)
            return plan
        if self.strategy_planner:
            return self.decide_strategy(question)
        self.memory.push_strategy(plan)
        return plan

    def _execute_plan(self, plan: StrategyPlan):
        if not self.planner or not self.retriever_manager:
            raise MissingDependencyError("planner or retriever_manager is missing")

        actions = self.planner.plan_from_strategy(plan)
        return self._execute_actions(actions)

    def _execute_actions(self, actions: Sequence["PlannerAction"]) -> List[RetrievalHit]:
        hits: List[RetrievalHit] = []
        if self.tool_executor:
            tool_hits = self._execute_plan_with_toolhub(actions)
            if tool_hits is not None:
                return tool_hits

        for action in actions:
            step = self._extract_step(action)
            stage_hits = self.retriever_manager.execute(step, self.memory)
            if stage_hits:
                hits.extend(stage_hits)
        return hits

    # --- ToolHub 执行骨架 ---
    def _execute_plan_with_toolhub(
        self,
        actions: Sequence["PlannerAction"],
    ) -> Optional[List[RetrievalHit]]:
        if not actions:
            return []

        context: Dict[str, ToolResult] = {}
        hits: List[RetrievalHit] = []
        all_error = True

        for action in actions:
            step = self._extract_step(action)
            tool_id = self._resolve_tool_id(step.tool)
            call_args = self._materialize_args(step, context)
            call = ToolCall(tool_id=tool_id, args=call_args)
            result = self.tool_executor.run(call)
            key = step.step_id or step.tool
            context[key] = result
            if result.status != "error":
                all_error = False
            extracted = self._extract_hits_from_result(result)
            if extracted:
                hits.extend(extracted)
            if tool_id == "vlm.answer":
                self._record_vlm_observation(tool_id, step, call_args, result)
            if tool_id == "judger.verify":
                context["judger.verdict"] = result.data or {}

        if all_error:
            return None
        return hits

    def _should_run_stage(self, stage: StrategyStage, ctx: Dict[str, float]) -> bool:
        condition = (stage.run_if or "").strip() if stage.run_if else ""
        if not condition:
            return True
        tokens = condition.split()
        if len(tokens) != 3:
            return True
        lhs, op, rhs = tokens
        lhs_value = float(ctx.get(lhs, 0.0))
        try:
            rhs_value = float(rhs)
        except ValueError:
            rhs_value = float(ctx.get(rhs, 0.0))
        comparators: Dict[str, Callable[[float, float], bool]] = {
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        compare = comparators.get(op)
        if compare is None:
            return True
        return compare(lhs_value, rhs_value)

    def _actions_for_stage(
        self,
        actions: Sequence["PlannerAction"],
        stage: StrategyStage,
    ) -> List["PlannerAction"]:
        if not stage.step_ids:
            return list(actions)
        allowed = {step_id for step_id in stage.step_ids if step_id}
        if not allowed:
            return list(actions)
        filtered = [
            action
            for action in actions
            if action.source_step and action.source_step.step_id in allowed
        ]
        return filtered or list(actions)

    def _compute_quality_metrics(
        self,
        plan: StrategyPlan,
        stage: StrategyStage,
        hits: Sequence[RetrievalHit],
        answer: ReasonerAnswer,
        verdict: Optional[Dict[str, object]] = None,
    ) -> Dict[str, float]:
        coverage = self._estimate_coverage(hits, stage, answer)
        format_ok = self._format_score(
            plan.final,
            answer.answer if answer else "",
            verdict.get("scores", {}).get("format") if verdict else None,
        )
        source_priority = self._source_priority(answer.support_nodes if answer else [])
        model_conf = self._clamp(answer.confidence if answer else 0.0)
        return {
            "coverage": coverage,
            "format_ok": format_ok,
            "source_priority": source_priority,
            "model_conf": model_conf,
        }

    def _compute_quality_score(self, metrics: Dict[str, float], judger_pass: bool) -> float:
        return (
            0.25 * metrics.get("coverage", 0.0)
            + 0.15 * metrics.get("format_ok", 0.0)
            + 0.40 * (1.0 if judger_pass else 0.0)
            + 0.10 * metrics.get("source_priority", 0.0)
            + 0.10 * metrics.get("model_conf", 0.0)
        )

    def _estimate_coverage(
        self,
        hits: Sequence[RetrievalHit],
        stage: StrategyStage,
        answer: Optional[ReasonerAnswer] = None,
    ) -> float:
        unique_nodes = {hit.node_id for hit in hits if hit and hit.node_id}
        if (not unique_nodes) and answer and answer.support_nodes:
            unique_nodes = {node for node in answer.support_nodes if node}
        if not unique_nodes:
            return 0.0
        target = stage.k_nodes or 50
        target = target if target > 0 else 50
        return min(1.0, len(unique_nodes) / max(1, target))

    def _format_score(
        self,
        final_spec: Optional[FinalSpec],
        answer_text: str,
        override: Optional[object] = None,
    ) -> float:
        if isinstance(override, bool):
            return 1.0 if override else 0.0
        if not answer_text.strip():
            return 0.0
        if not final_spec or not final_spec.format:
            return 1.0
        options = [opt.strip() for opt in final_spec.format.split("|") if opt.strip()]
        if not options:
            return 1.0
        text = answer_text.strip()

        def matches(fmt: str) -> bool:
            cleaned = text.replace(",", "").strip()
            if fmt == "int":
                try:
                    int(cleaned)
                    return True
                except ValueError:
                    return False
            if fmt == "float":
                try:
                    float(cleaned.replace("%", ""))
                    return True
                except ValueError:
                    return False
            if fmt == "list":
                return any(sep in text for sep in ("\n", ";", ",")) and len(text) > 0
            if fmt == "string":
                return True
            return False

        for option in options:
            if matches(option):
                return 1.0
        return 0.0

    def _source_priority(self, support_nodes: Sequence[str]) -> float:
        resources = getattr(self.retriever_manager, "resources", None)
        if not resources or not support_nodes:
            return 0.0
        priority = 0.3
        for node_id in support_nodes:
            role = resources.node_roles.get(node_id)
            if role == "table":
                return 1.0
            if role in {"image", "figure", "chart"}:
                priority = max(priority, 0.6)
        return priority

    def _heuristic_judger(self, answer: ReasonerAnswer) -> bool:
        if not answer or not answer.answer.strip():
            return False
        has_support = bool(answer.support_nodes)
        return has_support and self._clamp(answer.confidence) >= 0.4

    def _judger_verify(
        self,
        answer: ReasonerAnswer,
        plan: StrategyPlan,
        stage: StrategyStage,
    ) -> Optional[Dict[str, object]]:
        if not answer:
            return None
        if not self.memory.router_history:
            return None
        decision = self.memory.router_history[-1]
        if not self.llm_judger:
            return None
        return self.llm_judger.verify(
            question=decision.query,
            answer=answer,
            observations=self.memory.observations,
            signals=decision.signals,
            constraints=decision.constraints,
        )

    @staticmethod
    def _merge_verdicts(
        rule_verdict: Dict[str, object],
        llm_verdict: Dict[str, object],
    ) -> Dict[str, object]:
        if not llm_verdict:
            return rule_verdict

        fatal_issues = {"unit_mismatch", "threshold_fail", "no_support"}
        rule_pass = bool(rule_verdict.get("pass", False))
        llm_pass = bool(llm_verdict.get("pass", False))
        rule_issues = rule_verdict.get("issues", []) or []
        has_fatal = any(issue in fatal_issues for issue in rule_issues)
        combined_pass = llm_pass and not has_fatal

        combined_scores: Dict[str, float] = {}
        for source in (rule_verdict, llm_verdict):
            scores = source.get("scores")
            if isinstance(scores, dict):
                for key, value in scores.items():
                    try:
                        combined_scores[key] = float(value) if isinstance(value, (int, float)) else value
                    except (TypeError, ValueError):
                        combined_scores[key] = value

        issues: List[str] = []
        for source in (rule_verdict, llm_verdict):
            for issue in source.get("issues", []) or []:
                if issue not in issues:
                    issues.append(issue)

        recommendation = AgentOrchestrator._select_recommendation(
            combined_pass,
            rule_verdict.get("recommendation"),
            llm_verdict.get("recommendation"),
        )

        details: Dict[str, object] = {}
        for source in (rule_verdict, llm_verdict):
            detail = source.get("details")
            if isinstance(detail, dict):
                details.update(detail)

        explanation = llm_verdict.get("details", {}).get("llm_explanation") if isinstance(llm_verdict.get("details"), dict) else None
        if not explanation:
            explanation = llm_verdict.get("explanation")

        final = {
            "pass": combined_pass,
            "scores": combined_scores,
            "issues": issues,
            "recommendation": {"action": recommendation},
            "details": details,
            "explanation": explanation,
        }
        return final

    @staticmethod
    def _select_recommendation(
        passed: bool,
        *recommendations,
    ) -> str:
        if passed:
            return "accept"
        priority = ["return_unanswerable", "redo_answer", "replan"]
        for action in priority:
            for rec in recommendations:
                if isinstance(rec, dict) and rec.get("action") == action:
                    return action
                if isinstance(rec, str) and rec == action:
                    return action
        return "replan"

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return low

    def _materialize_args(
        self,
        step: StrategyStep,
        context: Dict[str, ToolResult],
    ) -> Dict[str, object]:
        args: Dict[str, object] = {}
        if step.args:
            args.update(step.args)
        args["_step"] = step
        args["_memory"] = self.memory
        args["_retriever_manager"] = self.retriever_manager
        args["_context"] = context
        args["_source_tool"] = step.tool
        legacy_tool_map = {
            "bm25_node.search": "sparse_search",
            "dense_node.search": "dense_search",
            "chart_index.search": "dense_search",
            "table_index.search": "dense_search",
        }
        args["_legacy_tool"] = legacy_tool_map.get(step.tool)
        if step.tool == "vlm.answer":
            images, rois = self._gather_images_for_vlm(step, args, context)
            if images:
                args.setdefault("images", images)
            if rois:
                args.setdefault("rois", rois)
            base_dir = self.retriever_manager.resources.base_dir
            if base_dir:
                args.setdefault("base_dir", str(base_dir))
        return args

    def _resolve_tool_id(self, tool_name: str) -> str:
        alias_map = {
            "dense_search": "bm25_node.search",
            "sparse_search": "bm25_node.search",
            "hybrid_search": "bm25_node.search",
            "jump_to_page": "page_locator.locate",
            "jump_to_label": "figure_finder.find_regions",
        }
        return alias_map.get(tool_name, tool_name)

    @staticmethod
    def _extract_hits_from_result(result: ToolResult) -> List[RetrievalHit]:
        data = result.data or {}
        hits = data.get("hits")
        if isinstance(hits, list):
            return [hit for hit in hits if isinstance(hit, RetrievalHit)]
        return []

    def _gather_images_for_vlm(
        self,
        step: StrategyStep,
        args: Dict[str, object],
        context: Dict[str, ToolResult],
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        images: List[str] = []
        rois: List[Dict[str, object]] = []

        existing = args.get("images")
        if isinstance(existing, list):
            images.extend([str(item) for item in existing if isinstance(item, str)])
        elif isinstance(existing, str):
            images.append(existing)

        resources = self.retriever_manager.resources

        # gather images from page hints
        page_hint = args.get("page_hint") or []
        pages: List[int] = []
        if isinstance(page_hint, list):
            for item in page_hint:
                try:
                    pages.append(int(item))
                except (TypeError, ValueError):
                    continue
        allowed_pages = {p for p in pages if p > 0}
        figure_hint = args.get("figure_hint") or []
        target_nodes: set[str] = set()
        if isinstance(figure_hint, list):
            for label in figure_hint:
                if not isinstance(label, str):
                    continue
                mapped = resources.label_index.get(label) or resources.label_index.get(label.strip().lower())
                if mapped:
                    target_nodes.add(mapped)
        def _as_int(value) -> Optional[int]:
            try:
                if value is None:
                    return None
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        return None
                return int(value)
            except (TypeError, ValueError):
                return None
        seen_nodes: set[str] = set()

        def _node_matches(node_id: str) -> bool:
            if target_nodes:
                return node_id in target_nodes
            if not allowed_pages:
                return True
            logical_page = _as_int(resources.node_pages.get(node_id))
            if logical_page in allowed_pages:
                return True
            physical_page = _as_int(resources.node_physical_pages.get(node_id))
            if physical_page in allowed_pages:
                return True
            return False

        def _append_node(node_id: str) -> None:
            if node_id in seen_nodes:
                return
            if resources.node_roles.get(node_id) != "image":
                return
            path = resources.image_paths.get(node_id)
            if not path:
                return
            images.append(path)
            seen_nodes.add(node_id)

        if target_nodes:
            for node_id in target_nodes:
                _append_node(node_id)

        def _collect_candidates_for_page(page: int) -> set[str]:
            candidates: set[str] = set()
            logical_page_nodes = {
                node_id
                for node_id, value in resources.node_pages.items()
                if _as_int(value) == page
            }
            candidates.update(logical_page_nodes)

            physical_nodes = {
                node_id
                for node_id, value in resources.node_physical_pages.items()
                if _as_int(value) == page
            }
            candidates.update(physical_nodes)

            zero_based = page - 1
            if zero_based >= 0:
                candidates.update(resources.page_index.get(zero_based, []))
            candidates.update(resources.page_index.get(page, []))
            return candidates

        for page in sorted(allowed_pages):
            for node_id in _collect_candidates_for_page(page):
                if _node_matches(node_id):
                    _append_node(node_id)

        # gather from dependent steps (rois/hits)
        if not images:
            for use in step.uses or []:
                result = context.get(use)
                if not isinstance(result, ToolResult):
                    continue
                data = result.data if isinstance(result.data, dict) else {}
                rois_data = data.get("rois")
                if isinstance(rois_data, list):
                    rois.clear()
                    for roi in rois_data:
                        if not isinstance(roi, dict):
                            continue
                        path = roi.get("path") or resources.image_paths.get(str(roi.get("node_id")))
                        if path:
                            rois.append({**roi, "path": path})
                    continue
                hits = data.get("hits")
                if isinstance(hits, list):
                    for hit in hits:
                        if not isinstance(hit, RetrievalHit):
                            continue
                        node_id = hit.node_id
                        if not _node_matches(node_id):
                            continue
                        _append_node(node_id)

        # fallback: use last cached hits
        if not images:
            cache = self.memory.retrieval_cache.copy()
            for hits in cache.values():
                for hit in hits:
                    node_id = hit.node_id
                    if _node_matches(node_id):
                        _append_node(node_id)

        if rois:
            ordered = []
            seen = set()
            for roi in rois:
                path = roi.get("path")
                if path and path not in seen:
                    ordered.append(path)
                    seen.add(path)
            images = ordered
        return images, rois

    def _record_vlm_observation(
        self,
        tool_id: str,
        step: StrategyStep,
        call_args: Dict[str, object],
        result: ToolResult,
    ) -> None:
        data = result.data if isinstance(result.data, dict) else {}
        answer_text = data.get("answer")
        payload = {
            "question": call_args.get("question"),
            "images": call_args.get("images"),
            "vision": {
                "tool": tool_id,
                "answer": answer_text,
                "thinking": data.get("thinking"),
                "confidence": data.get("confidence"),
                "details": data.get("details"),
            },
        }
        if isinstance(answer_text, str) and answer_text.strip():
            payload["text"] = answer_text.strip()
        node_id = f"vlm.answer:{step.step_id or 'default'}"

        obs = Observation(node_id=node_id, modality="image", payload=payload)
        self.memory.remember_observation(obs)

    def _run_observers(self, hits) -> None:
        if not self.observer:
            raise MissingDependencyError("observer is missing")
        observe_actions = self.planner.observation_plan(hits=hits, memory=self.memory)
        for action in observe_actions:
            if action.type != "observe":
                continue
            nodes: Sequence[str] = action.payload.get("nodes", [])
            self.observer.observe(nodes, self.memory)

    @staticmethod
    def _extract_step(action: PlannerAction) -> StrategyStep:
        if not action.source_step:
            raise ValueError("PlannerAction requires source_step for execution")
        return action.source_step


__all__ = ["AgentConfig", "AgentOrchestrator", "MissingDependencyError"]
