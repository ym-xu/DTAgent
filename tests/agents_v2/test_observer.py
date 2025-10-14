import json
import tempfile
import unittest
from pathlib import Path

from agents_v2.loaders import build_observer_store
from agents_v2.memory import AgentMemory
from agents_v2.observer import Observer


class ObserverLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        summary = {
            "doc_id": "doc1",
            "nodes": [
                {
                    "node_id": "sec_1",
                    "role": "section",
                    "title": "Introduction",
                    "page_idx": 1,
                    "dense_text": "Intro section overview.",
                    "hints": ["intro"],
                },
                {
                    "node_id": "img_1",
                    "role": "image",
                    "label": "Figure 1",
                    "page_idx": 1,
                    "dense_text": "Figure shows workflow.",
                },
                {
                    "node_id": "tab_1",
                    "role": "table",
                    "label": "Table 1",
                    "page_idx": 1,
                    "dense_text": "Table of land area percentages.",
                },
                {
                    "node_id": "tab_html",
                    "role": "table",
                    "label": "Table HTML",
                    "page_idx": 2,
                    "dense_text": "HTML table sample.",
                },
            ],
        }
        (base / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        edges = [
            {"src": "sec_1", "dst": "img_1", "type": "child"},
            {"src": "sec_1", "dst": "tab_1", "type": "child"},
            {"src": "sec_1", "dst": "tab_html", "type": "child"},
        ]
        with (base / "graph_edges.jsonl").open("w", encoding="utf-8") as f:
            for edge in edges:
                f.write(json.dumps(edge) + "\n")

        doctree = {
            "node_id": "root",
            "children": [
                {
                    "node_id": "sec_1",
                    "role": "section",
                    "children": [
                        {
                            "node_id": "img_1",
                            "role": "image",
                            "children": [
                                {"role": "caption", "text": "Workflow overview"},
                            ],
                        },
                        {
                            "node_id": "tab_1",
                            "role": "table",
                            "data": [
                                {"Borough": "Bronx", "% Land Area Rezoned": "1.5%"},
                                {"Borough": "Queens", "% Land Area Rezoned": "2.0%"},
                            ],
                            "children": [
                                {"role": "caption", "text": "Rezoned land area by borough"},
                            ],
                        },
                        {
                            "node_id": "tab_html",
                            "role": "table",
                            "data": "<table><tr><td></td><td>% Land Area Rezoned</td></tr><tr><td>The Bronx</td><td>18.4%</td></tr><tr><td>Brooklyn</td><td>13.9%</td></tr></table>",
                            "children": [
                                {"role": "caption", "text": "HTML table"},
                            ],
                        },
                    ],
                }
            ],
        }
        (base / "doctree.mm.json").write_text(json.dumps(doctree), encoding="utf-8")

        dense_leaf = [
            {
                "node_id": "sec_1#c0",
                "raw_text": "Detailed introduction paragraph.",
                "filters": {"parent_section": "sec_1", "page_idx": 1},
            }
        ]
        with (base / "dense_leaf.jsonl").open("w", encoding="utf-8") as f:
            for rec in dense_leaf:
                f.write(json.dumps(rec) + "\n")

        self.base = base

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_build_observer_store(self) -> None:
        store = build_observer_store(self.base)
        self.assertIn("sec_1", store)
        self.assertIn("img_1", store)
        self.assertIn("tab_1", store)
        self.assertIn("sec_1#c0", store)
        self.assertEqual(store["img_1"].modality, "image")
        self.assertIn("table", store["tab_1"].extra)
        table_info = store["tab_1"].extra["table"]
        self.assertIn("% Land Area Rezoned", table_info["columns"])
        self.assertTrue(any(cell.lower().startswith("bronx") for cell in (row[0] for row in table_info["rows"])))
        self.assertEqual(store["img_1"].extra["image"]["caption"], "Workflow overview")
        # image_path resolved only if存在，在测试环境为空即可

    def test_observer_from_doc_dir(self) -> None:
        observer = Observer.from_doc_dir(self.base)
        memory = AgentMemory()
        obs = observer.observe(["sec_1", "img_1", "sec_1#c0"], memory)
        self.assertEqual(len(obs), 3)
        self.assertIn("sec_1", memory.observations)

    def test_observer_image_analyzer(self) -> None:
        store = build_observer_store(self.base)

        calls = {}

        def analyzer(evidence):
            calls[evidence.node_id] = True
            return {"summary": f"vision summary for {evidence.node_id}", "labels": ["workflow"]}

        observer = Observer(store=store, image_analyzer=analyzer)
        memory = AgentMemory()
        obs_list = observer.observe(["img_1"], memory)
        self.assertTrue(calls.get("img_1"))
        payload = obs_list[0].payload
        self.assertIn("vision", payload)
        self.assertEqual(payload["vision"]["labels"], ["workflow"])
        self.assertEqual(payload["vision"]["summary"], "vision summary for img_1")

    def test_html_table_parsing(self) -> None:
        # reuse store from setUp (already has HTML table)
        store = build_observer_store(self.base)
        table = store["tab_html"].extra["table"]
        self.assertIn("% Land Area Rezoned", table["columns"])
        idx = table["columns"].index("% Land Area Rezoned") if "% Land Area Rezoned" in table["columns"] else -1
        self.assertGreaterEqual(idx, 0)
        bronx_row = next(row for row in table["rows"] if row[0].lower().startswith("the bronx") or row[0].lower().startswith("bronx"))
        self.assertEqual(bronx_row[idx], "18.4%")


if __name__ == "__main__":
    unittest.main()
