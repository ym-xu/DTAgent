import json
import unittest

from agents_v2.observer import NodeEvidence, Observer
from agents_v2.orchestrator import AgentConfig, AgentOrchestrator
from agents_v2.planner import DocGraphNavigator
from agents_v2.reasoner import Reasoner
from agents_v2.retriever import RetrieverManager, build_stub_resources
from agents_v2.router import QuestionRouter
from agents_v2.schemas import (
    FinalSpec,
    PlannerAction,
    ReasonerAnswer,
    StrategyKind,
    StrategyPlan,
    StrategyStage,
    StrategyStep,
)


class StubPlanner:
    """Deterministic planner used for tests without LLM calls."""

    def __init__(self, *, graph: DocGraphNavigator, observation_limit: int = 5) -> None:
        self.graph = graph
        self.observation_limit = observation_limit

    def plan_from_router(self, decision) -> StrategyPlan:
        args = {"query": decision.query or "", "view": "section#gist"}
        step = StrategyStep(tool="dense_node.search", args=args, step_id="S1")
        stage = StrategyStage(
            stage="primary",
            methods=[step.tool],
            k_pages=0,
            k_nodes=0,
            page_window=0,
            params={},
            run_if=None,
            step_ids=[step.step_id],
            evidence_type="text",
        )
        final = FinalSpec(answer_var="reasoner", format="string")
        plan = StrategyPlan(
            strategy=StrategyKind.SINGLE,
            steps=[step],
            confidence=decision.confidence,
            retrieval_keys=[decision.query] if decision.query else None,
            stages=[stage],
            final=final,
        )
        return plan

    def plan_from_strategy(self, plan: StrategyPlan):
        actions = []
        for step in plan.steps:
            payload = {"tool": step.tool, **step.args}
            if step.save_as:
                payload["save_as"] = step.save_as
            if step.when:
                payload["when"] = step.when
            if step.uses:
                payload["uses"] = list(step.uses)
            actions.append(PlannerAction(type="retrieve", payload=payload, source_step=step))
        return actions

    def observation_plan(self, *, hits, memory, include_neighbors: bool = True):
        unseen = []
        for hit in hits:
            if hit.node_id and not memory.has_seen(hit.node_id):
                unseen.append(hit.node_id)
            if len(unseen) >= self.observation_limit:
                break
        if not unseen:
            return [PlannerAction(type="noop", payload={"reason": "no-new-targets"})]
        return [PlannerAction(type="observe", payload={"nodes": unseen})]


class OrchestratorIntegrationTests(unittest.TestCase):
    def _build_dependencies(self):
        resources = build_stub_resources(
            label_pairs=[("Figure 3", "img_3")],
            page_pairs=[(5, "sec_5"), (5, "tab_6")],
            text_pairs=[
                ("sec_5", "This section describes accuracy trends."),
                ("tab_6", "Table with latency numbers."),
                ("img_3", "Figure about accuracy increase."),
            ],
        )
        graph = DocGraphNavigator(
            children={"sec_5": ["img_3", "tab_6"]},
            parents={"img_3": "sec_5", "tab_6": "sec_5"},
            same_page={"sec_5": ["sec_6"]},
        )
        node_store = {
            "sec_5": NodeEvidence(node_id="sec_5", modality="text", content="章节描述准确率上升。"),
            "tab_6": NodeEvidence(node_id="tab_6", modality="table", content="表格列出延迟与准确率。"),
            "img_3": NodeEvidence(node_id="img_3", modality="image", content="图3展示准确率随时间提高。"),
        }
        router = QuestionRouter(llm_callable=self._stub_router_llm)
        planner = StubPlanner(graph=graph, observation_limit=5)
        retriever = RetrieverManager(resources)
        observer = Observer(store=node_store)
        def stub_llm(**kwargs):
            return json.dumps(
                {
                    "answer": "模型准确率持续上升，并伴随延迟改善。",
                    "confidence": 0.8,
                    "support_nodes": ["sec_5", "tab_6"],
                }
            )

        reasoner = Reasoner(min_confidence=0.65, use_llm=True, llm_callable=stub_llm)
        return router, planner, retriever, observer, reasoner

    def test_full_pipeline_returns_answer(self) -> None:
        deps = self._build_dependencies()
        orchestrator = AgentOrchestrator(
            router=deps[0],
            planner=deps[1],
            retriever_manager=deps[2],
            observer=deps[3],
            reasoner=deps[4],
            config=AgentConfig(max_iterations=2),
        )

        answer = orchestrator.run("请解释图3和第5页的趋势")
        self.assertIsInstance(answer, ReasonerAnswer)
        self.assertGreaterEqual(answer.confidence, 0.6)
        self.assertTrue(answer.support_nodes)
        self.assertIn("准确率", answer.answer)
        self.assertTrue(orchestrator.memory.strategy_history)
        first_plan = orchestrator.memory.strategy_history[0]
        self.assertTrue(first_plan.stages)

    def test_fallback_when_no_observations(self) -> None:
        resources = build_stub_resources(
            label_pairs=[],
            page_pairs=[],
            text_pairs=[],
        )
        graph = DocGraphNavigator()
        router = QuestionRouter(llm_callable=self._stub_router_llm)
        planner = StubPlanner(graph=graph)
        retriever = RetrieverManager(resources)
        observer = Observer(store={})
        reasoner = Reasoner(use_llm=True, llm_callable=lambda **_: "")

        orchestrator = AgentOrchestrator(
            router=router,
            planner=planner,
            retriever_manager=retriever,
            observer=observer,
            reasoner=reasoner,
            config=AgentConfig(max_iterations=1),
        )

        answer = orchestrator.run("请概述结果")

        self.assertEqual(answer.confidence, 0.0)
        self.assertIn("信息不足", answer.answer)

    # --- helpers ---

    @staticmethod
    def _stub_router_llm(*, question: str, config, toc_outline=None) -> str:  # noqa: D401
        return json.dumps(
            {
                "query": question,
                "query_type": "text",
                "signals": {
                    "page_hint": [],
                    "figure_hint": [],
                    "table_hint": [],
                    "objects": [],
                    "units": [],
                    "years": [],
                    "operations": [],
                    "expected_format": "string",
                    "section_cues": [],
                    "keywords": [],
                    "objects_scope": None,
                },
                "risk": {
                    "ambiguity": 0.1,
                    "need_visual": False,
                    "need_table": False,
                    "need_chart": False,
                },
                "constraints": {
                    "allow_unanswerable": True,
                    "must_cite": True,
                },
                "confidence": 0.9,
            }
        )

if __name__ == "__main__":
    unittest.main()
