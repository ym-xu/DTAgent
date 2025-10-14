"""
AgentOrchestrator
=================

协调问题处理、策略判定、规划、检索、观察与推理模块的主流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

from .memory import AgentMemory
from .schemas import (
    PlannerAction,
    ReasonerAnswer,
    RouterDecision,
    StrategyPlan,
    StrategyStep,
)
from .toolhub import ToolCall, ToolExecutor, ToolResult


class MissingDependencyError(RuntimeError):
    """缺少依赖时抛出的异常。"""


@dataclass
class AgentConfig:
    """Agent 运行时配置。"""

    max_iterations: int = 4


class AgentOrchestrator:
    """协调新一代 Agent 流程的顶层对象。"""

    def __init__(
        self,
        *,
        router,
        strategy_planner,
        planner,
        retriever_manager,
        observer,
        reasoner,
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
            hits = self._execute_plan(plan)
            self._run_observers(hits)
            answer = self.reasoner.run(normalized, self.memory.snapshot())
            if answer.action == "REPLAN":
                if iteration + 1 >= self.config.max_iterations:
                    final_answer = ReasonerAnswer(
                        answer="Insufficient information to reach a conclusion",
                        confidence=0.0,
                        support_nodes=[],
                        reasoning_trace=["Reached iteration limit"],
                    )
                    self.last_answer = final_answer
                    return final_answer
                plan = self._plan_from_router(decision, normalized)
                if on_plan and not plan.is_empty():
                    on_plan(plan)
                continue
            self.last_answer = answer
            return answer

        final_answer = ReasonerAnswer(
            answer="Insufficient information to reach a conclusion",
            confidence=0.0,
            support_nodes=[],
            reasoning_trace=["Exceeded iteration loop"],
        )
        self.last_answer = final_answer
        return final_answer

    # --- 内部辅助 ---
    def _plan_from_router(
        self,
        decision: RouterDecision,
        question: str,
    ) -> StrategyPlan:
        plan = self.planner.plan_from_router(decision)
        if not plan.is_empty():
            self.memory.push_strategy(plan)
            return plan
        return self.decide_strategy(question)

    def _execute_plan(self, plan: StrategyPlan):
        if not self.planner or not self.retriever_manager:
            raise MissingDependencyError("planner or retriever_manager is missing")

        if self.tool_executor:
            self._execute_plan_with_toolhub(plan)

        actions = self.planner.plan_from_strategy(plan)
        all_hits = []
        for action in actions:
            step = self._extract_step(action)
            hits = self.retriever_manager.execute(step, self.memory)
            if hits:
                all_hits.extend(hits)
        return all_hits

    # --- ToolHub 执行骨架 ---
    def _execute_plan_with_toolhub(self, plan: StrategyPlan) -> Dict[str, ToolResult]:
        context: Dict[str, ToolResult] = {}
        if not plan.steps:
            return context
        for step in plan.steps:
            call = ToolCall(tool_id=step.tool, args=self._materialize_args(step, context))
            result = self.tool_executor.run(call)
            key = step.step_id or step.tool
            context[key] = result
        return context

    def _materialize_args(
        self,
        step: StrategyStep,
        context: Dict[str, ToolResult],
    ) -> Dict[str, object]:
        # TODO: 根据上下文解析 from/uses
        return dict(step.args or {})

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
