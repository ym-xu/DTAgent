import json
import tempfile
import unittest
from pathlib import Path

from agents_v2.loaders import build_resources_from_index
from agents_v2.memory import AgentMemory
from agents_v2.observer import NodeEvidence, Observer
from agents_v2.planner import Planner
from agents_v2.retriever import RetrieverManager
from agents_v2.schemas import StrategyKind, StrategyPlan, StrategyStep


class RetrieverManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        summary = {
            "doc_id": "doc1",
            "nodes": [
                {
                    "node_id": "sec_5",
                    "role": "section",
                    "page_idx": 5,
                    "dense_text": "This section discusses accuracy and latency trends.",
                },
                {
                    "node_id": "tab_6",
                    "role": "table",
                    "page_idx": 5,
                    "label": "Table 1",
                    "dense_text": "Table of hyper-parameters and latency.",
                },
                {
                    "node_id": "img_3",
                    "role": "image",
                    "label": "Figure 3",
                    "page_idx": 5,
                    "dense_text": "Figure showing accuracy curve.",
                },
                {
                    "node_id": "sec_2",
                    "role": "section",
                    "page_idx": 4,
                    "dense_text": "Background information.",
                },
            ],
        }
        (base / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        edges = [
            {"src": "sec_5", "dst": "tab_6", "type": "child"},
            {"src": "sec_5", "dst": "img_3", "type": "child"},
            {"src": "sec_5", "dst": "sec_2", "type": "same_page"},
        ]
        with (base / "graph_edges.jsonl").open("w", encoding="utf-8") as f:
            for edge in edges:
                f.write(json.dumps(edge) + "\n")

        dense_records = [
            {
                "node_id": "sec_5#h",
                "variant": "heading",
                "dense_text": "Accuracy trends heading.",
            },
            {
                "node_id": "sec_5#g",
                "variant": "gist",
                "dense_text": "Discuss accuracy improvements and latency trade-offs.",
            },
            {
                "node_id": "tab_6",
                "variant": "table",
                "dense_text": "Latency numbers per configuration.",
            },
        ]
        with (base / "dense_coarse.jsonl").open("w", encoding="utf-8") as f:
            for rec in dense_records:
                f.write(json.dumps(rec) + "\n")

        sparse_records = [
            {
                "id": "sec_5",
                "title": "Accuracy analysis",
                "caption": "Chart accuracy latency",
                "body": "Accuracy improves while latency decreases.",
            },
            {
                "id": "tab_6",
                "title": "Hyper-parameter table",
                "table_schema": "accuracy (%); latency (ms)",
                "body": "Model latency values.",
            },
        ]
        with (base / "sparse_coarse.jsonl").open("w", encoding="utf-8") as f:
            for rec in sparse_records:
                f.write(json.dumps(rec) + "\n")

        resources, graph = build_resources_from_index(base)

        def retriever_stub(*, tool: str, query: str, candidates, config):
            ordered = []
            for idx, cand in enumerate(candidates[:5]):
                ordered.append(
                    {
                        "node_id": cand["node_id"],
                        "base_id": cand.get("base_id", cand["node_id"]),
                        "score": max(0.1, 1.0 - 0.1 * idx),
                    }
                )
            return json.dumps({
                "thinking": f"{tool} stub ranks {len(ordered)} candidates for query '{query}'.",
                "results": ordered,
            })

        self.manager = RetrieverManager(resources, llm_callable=retriever_stub)
        self.planner = Planner(graph=graph, observation_limit=5)
        self.memory = AgentMemory()
        self.observer = Observer(
            store={
                "sec_5": NodeEvidence(node_id="sec_5", content="Section evidence"),
                "tab_6": NodeEvidence(node_id="tab_6", content="Table evidence"),
            }
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_jump_to_label(self) -> None:
        step = StrategyStep(tool="jump_to_label", args={"label": "Figure 3"})
        hits = self.manager.execute(step, self.memory)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].node_id, "img_3")
        cached = self.memory.get_cached_hits(step.describe())
        self.assertEqual(cached, hits)

    def test_dense_search_with_variant(self) -> None:
        step = StrategyStep(tool="dense_search", args={"query": "accuracy improvements", "view": "section#gist"})
        hits = self.manager.execute(step, self.memory)
        self.assertTrue(hits)
        self.assertEqual(hits[0].node_id, "sec_5")

    def test_sparse_search_uses_weighted_fields(self) -> None:
        step = StrategyStep(tool="sparse_search", args={"query": "latency"})
        hits = self.manager.execute(step, self.memory)
        nodes = [h.node_id for h in hits]
        self.assertIn("sec_5", nodes)
        self.assertIn("tab_6", nodes)

    def test_sparse_search_with_keywords(self) -> None:
        step = StrategyStep(
            tool="sparse_search",
            args={"keywords": ["latency % change", "table"], "view": "section#child"},
        )
        hits = self.manager.execute(step, self.memory)
        self.assertTrue(hits)

    def test_hybrid_combines_dense_and_sparse(self) -> None:
        step = StrategyStep(tool="hybrid_search", args={"query": "latency accuracy"})
        hits = self.manager.execute(step, self.memory)
        self.assertGreaterEqual(len(hits), 2)
        self.assertTrue(all(h.tool == "hybrid_search" for h in hits))

    def test_integration_with_planner_and_observer(self) -> None:
        plan = StrategyPlan(
            strategy=StrategyKind.SINGLE,
            steps=[StrategyStep(tool="dense_search", args={"query": "latency", "view": "section#gist"})],
            confidence=0.8,
        )
        retrieve_actions = self.planner.plan_from_strategy(plan)
        self.assertEqual(len(retrieve_actions), 1)

        hits = self.manager.execute(plan.steps[0], self.memory)
        obs_actions = self.planner.observation_plan(hits=hits, memory=self.memory)

        nodes = obs_actions[0].payload["nodes"]
        observations = self.observer.observe(nodes, self.memory)
        self.assertTrue(observations)
        self.assertIn("sec_5", [obs.node_id for obs in observations])


if __name__ == "__main__":
    unittest.main()
