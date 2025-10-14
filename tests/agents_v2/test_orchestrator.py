import json
import unittest

from agents_v2.observer import NodeEvidence, Observer
from agents_v2.orchestrator import AgentConfig, AgentOrchestrator
from agents_v2.planner import DocGraphNavigator, Planner
from agents_v2.reasoner import Reasoner
from agents_v2.retriever import RetrieverManager, build_stub_resources
from agents_v2.router import QuestionRouter
from agents_v2.schemas import ReasonerAnswer
from agents_v2.strategy_planner import RetrievalStrategyPlanner


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
        strategy_decider = RetrievalStrategyPlanner(llm_callable=self._stub_strategy_llm)
        planner = Planner(graph=graph, max_observation_neighbors=5)
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
        return router, strategy_decider, planner, retriever, observer, reasoner

    def test_full_pipeline_returns_answer(self) -> None:
        deps = self._build_dependencies()
        orchestrator = AgentOrchestrator(
            router=deps[0],
            strategy_planner=deps[1],
            planner=deps[2],
            retriever_manager=deps[3],
            observer=deps[4],
            reasoner=deps[5],
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
        self.assertGreater(first_plan.coverage_gate, 0)

    def test_fallback_when_no_observations(self) -> None:
        resources = build_stub_resources(
            label_pairs=[],
            page_pairs=[],
            text_pairs=[],
        )
        graph = DocGraphNavigator()
        router = QuestionRouter(llm_callable=self._stub_router_llm)
        strategy_decider = RetrievalStrategyPlanner(llm_callable=self._stub_strategy_llm)
        planner = Planner(graph=graph)
        retriever = RetrieverManager(resources)
        observer = Observer(store={})
        reasoner = Reasoner(use_llm=True, llm_callable=lambda **_: "")

        orchestrator = AgentOrchestrator(
            router=router,
            strategy_planner=strategy_decider,
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
    def _stub_router_llm(*, question: str, config) -> str:
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

    @staticmethod
    def _stub_strategy_llm(*, question: str, config) -> str:
        return json.dumps(
            {
                "thinking": "Use dense search to explore the relevant section.",
                "strategy": "SINGLE",
                "steps": [
                    {
                        "tool": "dense_search",
                        "args": {"query": question, "view": "section#gist"},
                    }
                ],
                "confidence": 0.8,
            }
        )


if __name__ == "__main__":
    unittest.main()
