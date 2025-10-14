"""
AgentMemory
===========

用于在一次问题求解过程中缓存检索与观察状态，
避免重复操作并为策略调整提供上下文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .schemas import Observation, RetrievalHit, RouterDecision, StrategyPlan


@dataclass
class AgentMemory:
    """简单的内存实现，后续可扩展为持久化或分层记忆。"""

    question_history: List[str] = field(default_factory=list)
    strategy_history: List[StrategyPlan] = field(default_factory=list)
    router_history: List[RouterDecision] = field(default_factory=list)
    retrieval_cache: Dict[str, List[RetrievalHit]] = field(default_factory=dict)
    observations: Dict[str, Observation] = field(default_factory=dict)
    visited_nodes: Set[str] = field(default_factory=set)
    iteration: int = 0

    def record_question(self, question: str) -> None:
        self.question_history.append(question)

    def push_router(self, decision: RouterDecision) -> None:
        self.router_history.append(decision)

    def push_strategy(self, plan: StrategyPlan) -> None:
        self.strategy_history.append(plan)

    def cache_hits(self, step_key: str, hits: List[RetrievalHit]) -> None:
        self.retrieval_cache[step_key] = hits

    def get_cached_hits(self, step_key: str) -> List[RetrievalHit]:
        return self.retrieval_cache.get(step_key, [])

    def remember_observation(self, obs: Observation) -> None:
        self.observations[obs.node_id] = obs
        self.visited_nodes.add(obs.node_id)

    def has_seen(self, node_id: str) -> bool:
        return node_id in self.visited_nodes

    def snapshot(self) -> Dict[str, object]:
        """返回当前状态快照，便于 Reasoner 使用。"""
        return {
            "iteration": self.iteration,
            "router": self.router_history,
            "strategies": self.strategy_history,
            "retrievals": self.retrieval_cache,
            "observations": self.observations,
            "visited_nodes": set(self.visited_nodes),
        }

    def next_iteration(self) -> int:
        self.iteration += 1
        return self.iteration


__all__ = ["AgentMemory"]
