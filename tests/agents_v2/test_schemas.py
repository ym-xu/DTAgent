import unittest

from agents_v2.schemas import (
    Observation,
    PlannerAction,
    ReasonerAnswer,
    RetrievalHit,
    StrategyKind,
    StrategyPlan,
    StrategyStep,
)


class StrategyPlanTests(unittest.TestCase):
    def test_strategy_plan_describe_and_flags(self) -> None:
        step = StrategyStep(
            tool="dense_search",
            args={"query": "model accuracy", "view": "section#gist"},
            weight=0.6,
        )
        plan = StrategyPlan(
            strategy=StrategyKind.COMPOSITE,
            steps=[step],
            confidence=0.8,
            notes="composite",
        )

        self.assertFalse(plan.is_empty())
        self.assertIn("dense_search", step.describe())
        self.assertEqual(plan.strategy, StrategyKind.COMPOSITE)
        self.assertAlmostEqual(plan.confidence, 0.8)
        self.assertEqual(plan.notes, "composite")
        self.assertIsNone(plan.retrieval_keys)


class PlannerActionTests(unittest.TestCase):
    def test_planner_action_require_fetches_payload(self) -> None:
        action = PlannerAction(type="retrieve", payload={"tool": "dense_search"})
        self.assertEqual(action.require("tool"), "dense_search")

    def test_planner_action_require_missing_key(self) -> None:
        action = PlannerAction(type="noop")
        with self.assertRaises(KeyError):
            action.require("tool")


class RetrievalObservationTests(unittest.TestCase):
    def test_retrieval_hit_and_observation_dataclasses(self) -> None:
        hit = RetrievalHit(node_id="sec_1", score=0.9, tool="dense_search")
        obs = Observation(node_id="sec_1", modality="text", payload={"summary": "..."})

        self.assertEqual(hit.metadata, {})
        self.assertEqual(obs.payload["summary"], "...")


class ReasonerAnswerTests(unittest.TestCase):
    def test_reasoner_answer_default_fields(self) -> None:
        ans = ReasonerAnswer(answer="42", confidence=0.7, support_nodes=["sec_1"])
        self.assertEqual(ans.reasoning_trace, [])
        self.assertIsNone(ans.action)
        self.assertIsNone(ans.missing_intent)
        self.assertIsNone(ans.thinking)


if __name__ == "__main__":
    unittest.main()
