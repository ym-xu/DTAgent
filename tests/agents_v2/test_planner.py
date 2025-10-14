import unittest

from agents_v2.memory import AgentMemory
from agents_v2.planner import DocGraphNavigator, Planner
from agents_v2.schemas import RetrievalHit, StrategyKind, StrategyPlan, StrategyStep


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        graph = DocGraphNavigator(
            children={
                "sec_1": ["img_1", "tab_1"],
                "sec_2": ["txt_3"],
            },
            parents={
                "img_1": "sec_1",
                "tab_1": "sec_1",
                "txt_3": "sec_2",
            },
            same_page={
                "sec_1": ["sec_2"],
                "img_1": ["tab_1"],
            },
        )
        self.memory = AgentMemory()
        self.planner = Planner(graph=graph, max_observation_neighbors=4)

    def test_plan_from_strategy_converts_steps(self) -> None:
        plan = StrategyPlan(
            strategy=StrategyKind.COMPOSITE,
            steps=[
                StrategyStep(tool="jump_to_label", args={"label": "Figure 3"}),
                StrategyStep(tool="dense_search", args={"query": "summary"}),
            ],
            confidence=0.8,
        )
        actions = self.planner.plan_from_strategy(plan)

        self.assertEqual(actions[0].type, "move")
        self.assertEqual(actions[0].payload["tool"], "jump_to_label")
        self.assertEqual(actions[1].type, "retrieve")
        self.assertEqual(actions[1].payload["tool"], "dense_search")

    def test_observation_plan_includes_neighbors(self) -> None:
        hits = [
            RetrievalHit(node_id="sec_1", score=0.9, tool="dense_search"),
        ]

        actions = self.planner.observation_plan(hits=hits, memory=self.memory)
        self.assertEqual(actions[0].type, "observe")
        nodes = actions[0].payload["nodes"]
        # 应包含自身邻域（子节点、同页等）
        self.assertIn("img_1", nodes)
        self.assertIn("tab_1", nodes)
        self.assertIn("sec_2", nodes)
        # 原节点本身需要也在列表中
        self.assertIn("sec_1", nodes)

    def test_observation_plan_noop_when_all_seen(self) -> None:
        self.memory.visited_nodes.update({"sec_1", "img_1", "tab_1", "sec_2"})
        hits = [RetrievalHit(node_id="sec_1", score=0.9, tool="dense_search")]
        actions = self.planner.observation_plan(hits=hits, memory=self.memory)

        self.assertEqual(actions[0].type, "noop")
        self.assertEqual(actions[0].payload["reason"], "no-new-targets")


if __name__ == "__main__":
    unittest.main()

