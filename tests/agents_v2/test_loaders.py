import json
import tempfile
import unittest
from pathlib import Path

from agents_v2.loaders import build_resources_from_index


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        summary = {
            "doc_id": "doc1",
            "nodes": [
                {
                    "node_id": "sec_1",
                    "role": "section",
                    "page_idx": 5,
                    "dense_text": "Section summary about accuracy.",
                },
                {
                    "node_id": "img_1",
                    "role": "image",
                    "label": "Figure 1",
                    "page_idx": 5,
                    "dense_text": "Figure showing accuracy.",
                },
                {
                    "node_id": "sec_2",
                    "role": "section",
                    "page_idx": 6,
                    "dense_text": "Another section on latency.",
                },
            ],
        }
        (base / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        edges = [
            {"src": "sec_1", "dst": "img_1", "type": "child"},
            {"src": "sec_1", "dst": "sec_2", "type": "same_page"},
        ]
        with (base / "graph_edges.jsonl").open("w", encoding="utf-8") as f:
            for edge in edges:
                f.write(json.dumps(edge) + "\n")

        dense_records = [
            {
                "node_id": "sec_1#h",
                "variant": "heading",
                "dense_text": "Accuracy overview heading.",
            },
            {
                "node_id": "sec_1#g",
                "variant": "gist",
                "dense_text": "Discuss accuracy improvements in detail.",
            },
            {
                "node_id": "img_1",
                "role": "image",
                "dense_text": "Figure explaining accuracy rise.",
            },
        ]
        with (base / "dense_coarse.jsonl").open("w", encoding="utf-8") as f:
            for rec in dense_records:
                f.write(json.dumps(rec) + "\n")

        sparse_records = [
            {
                "id": "sec_1",
                "title": "Accuracy analysis section",
                "caption": "Line chart accuracy",
                "body": "Accuracy improves over time",
            }
        ]
        with (base / "sparse_coarse.jsonl").open("w", encoding="utf-8") as f:
            for rec in sparse_records:
                f.write(json.dumps(rec) + "\n")

        self.base = base

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_build_resources_from_index(self) -> None:
        resources, graph = build_resources_from_index(self.base)

        self.assertIn("Figure 1", resources.label_index)
        self.assertEqual(resources.label_index["Figure 1"], "img_1")
        self.assertIn(5, resources.page_index)
        self.assertIn("sec_1", resources.text_index)
        self.assertIn("gist", resources.dense_views)
        self.assertIn("sec_1#g", resources.dense_views["gist"])
        self.assertIn("sec_1", resources.sparse_docs)

        neighbors = graph.expand("sec_1", include_self=True)
        self.assertIn("img_1", neighbors)
        self.assertIn("sec_2", neighbors)


if __name__ == "__main__":
    unittest.main()
