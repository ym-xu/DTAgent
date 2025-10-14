import json
import unittest

from agents_v2.strategy_planner import RetrievalStrategyPlanner
from agents_v2.schemas import StrategyKind


class StrategyPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decider = RetrievalStrategyPlanner(llm_callable=self._stub_llm)

    def test_detects_label_and_page(self) -> None:
        question = "请直接查看图3和第5页的描述。"
        plan = self.decider.decide(question)

        tools = [step.tool for step in plan.steps]
        self.assertIn("jump_to_label", tools)
        self.assertIn("jump_to_page", tools)
        self.assertEqual(plan.strategy, StrategyKind.COMPOSITE)

    def test_sparse_trigger_via_keywords_and_quotes(self) -> None:
        question = '列出包含"accuracy"和"latency"的表格'
        plan = self.decider.decide(question)

        tools = [step.tool for step in plan.steps]
        self.assertIn("sparse_search", tools)
        self.assertIn("dense_search", tools)  # quotes + tokens -> hybrid
        self.assertTrue(plan.notes and "sparse" in plan.notes)

    def test_dense_trigger_for_open_question(self) -> None:
        question = "请解释模型在不同数据集上的趋势和影响"
        plan = self.decider.decide(question)

        tools = [step.tool for step in plan.steps]
        self.assertEqual(plan.strategy, StrategyKind.SINGLE)
        self.assertEqual(tools, ["dense_search"])
        self.assertGreater(plan.confidence, 0.7)

    def test_default_dense_when_no_signal(self) -> None:
        question = "Accuracy?"
        plan = self.decider.decide(question)

        tools = [step.tool for step in plan.steps]
        self.assertEqual(tools, ["dense_search"])
        self.assertEqual(plan.strategy, StrategyKind.SINGLE)

    @staticmethod
    def _stub_llm(*, question: str, config) -> str:
        question_lower = question.lower()
        if "图3" in question or "table" in question_lower:
            return json.dumps(
                {
                    "thinking": "Use label and page jumps along with dense search.",
                    "strategy": "COMPOSITE",
                    "steps": [
                        {"tool": "jump_to_label", "args": {"label": "Figure 3"}},
                        {"tool": "jump_to_page", "args": {"page": 5}},
                        {"tool": "dense_search", "args": {"query": question, "view": "section#gist"}},
                    ],
                    "confidence": 0.9,
                    "notes": "Label and page explicitly mentioned.",
                }
            )
        if "accuracy" in question_lower and "latency" in question_lower:
            return json.dumps(
                {
                    "thinking": "Use sparse for keywords and dense for context.",
                    "strategy": "COMPOSITE",
                    "steps": [
                        {
                            "tool": "sparse_search",
                            "args": {"keywords": ["accuracy", "latency"], "view": "section#child"},
                        },
                        {
                            "tool": "dense_search",
                            "args": {"query": "accuracy latency table", "view": "section#gist"},
                        },
                    ],
                    "confidence": 0.85,
                    "notes": "Combines sparse and dense search.",
                }
            )
        if "趋势" in question or "trend" in question_lower:
            return json.dumps(
                {
                    "thinking": "Open question best served by dense search.",
                    "strategy": "SINGLE",
                    "steps": [
                        {"tool": "dense_search", "args": {"query": question, "view": "section#gist"}},
                    ],
                    "confidence": 0.8,
                }
            )
        return json.dumps(
            {
                "thinking": "Fallback dense search.",
                "strategy": "SINGLE",
                "steps": [
                    {"tool": "dense_search", "args": {"query": question, "view": "section#gist"}},
                ],
                "confidence": 0.7,
            }
        )


if __name__ == "__main__":
    unittest.main()
