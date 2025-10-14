import unittest

from agents_v2.planner import DocGraphNavigator, Planner
from agents_v2.schemas import (
    RouterConstraints,
    RouterDecision,
    RouterRisk,
    RouterSignals,
    StrategyKind,
)


class PlannerRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        graph = DocGraphNavigator()
        self.planner = Planner(graph=graph)

    def test_router_text_query_generates_dense_and_sparse_steps(self) -> None:
        decision = RouterDecision(
            query="How many research questions does this paper answer?",
            query_type="text",
            signals=RouterSignals(
                page_hint=[],
                figure_hint=[],
                table_hint=[],
                objects=[],
                units=[],
                years=[],
                operations=["count"],
                expected_format="int",
                section_cues=["Introduction", "Research Questions"],
                keywords=["research questions", "RQ"],
                objects_scope=None,
            ),
            risk=RouterRisk(ambiguity=0.2, need_visual=False, need_table=False, need_chart=False),
            constraints=RouterConstraints(allow_unanswerable=True, must_cite=True),
            confidence=0.8,
            raw={},
        )

        plan = self.planner.plan_from_router(decision)

        self.assertEqual(plan.strategy, StrategyKind.COMPOSITE)
        self.assertGreaterEqual(len(plan.steps), 2)
        views = {step.args.get("view") for step in plan.steps if "view" in step.args}
        self.assertIn("section#gist", views)
        self.assertTrue(plan.rerank)
        self.assertTrue(plan.pack)
        self.assertGreater(plan.coverage_gate, 0)
        self.assertEqual(plan.final.format, "int")

    def test_router_visual_count_prefers_image_view(self) -> None:
        decision = RouterDecision(
            query="Count the number of cars on page 2.",
            query_type="visual_count",
            signals=RouterSignals(
                page_hint=[2],
                figure_hint=[],
                table_hint=[],
                objects=["car"],
                units=[],
                years=[],
                operations=["count"],
                expected_format="int",
                section_cues=[],
                keywords=["cars"],
                objects_scope="all_figures_on_page",
            ),
            risk=RouterRisk(ambiguity=0.1, need_visual=True, need_table=False, need_chart=False),
            constraints=RouterConstraints(allow_unanswerable=True, must_cite=True),
            confidence=0.9,
            raw={},
        )

        plan = self.planner.plan_from_router(decision)

        image_steps = [
            step for step in plan.steps if step.args.get("view") == "image"
        ]
        self.assertTrue(image_steps, "Expected at least one image-view retrieval step")
        jump_steps = [step for step in plan.steps if step.tool == "jump_to_page"]
        self.assertTrue(jump_steps, "Expected jump_to_page step for page hints")


if __name__ == "__main__":
    unittest.main()
