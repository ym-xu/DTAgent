import json
import tempfile
import unittest
from pathlib import Path

from agents_v2.cli import run_question
from agents_v2.loaders import build_observer_store


class CLIRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        summary = {
            "doc_id": "doccli",
            "nodes": [
                {
                    "node_id": "sec_10",
                    "role": "section",
                    "title": "Results",
                    "page_idx": 2,
                    "dense_text": "Results section summarises accuracy and latency trends.",
                    "hints": ["accuracy", "latency"],
                },
                {
                    "node_id": "img_5",
                    "role": "image",
                    "label": "Figure 5",
                    "page_idx": 2,
                    "dense_text": "Figure shows latency decreasing.",
                },
                {
                    "node_id": "tab_7",
                    "role": "table",
                    "label": "Table 2",
                    "page_idx": 2,
                    "dense_text": "Table lists latency percentages by borough.",
                },
                {
                    "node_id": "lst_7",
                    "role": "list",
                    "page_idx": 2,
                    "dense_text": "List: First question; Second question; Third question",
                },
            ],
        }
        (base / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        edges = [
            {"src": "sec_10", "dst": "img_5", "type": "child"},
            {"src": "sec_10", "dst": "tab_7", "type": "child"},
            {"src": "sec_10", "dst": "lst_7", "type": "child"},
        ]
        with (base / "graph_edges.jsonl").open("w", encoding="utf-8") as f:
            for edge in edges:
                f.write(json.dumps(edge) + "\n")

        doctree = {
            "node_id": "root",
            "children": [
                {
                    "node_id": "sec_10",
                    "role": "section",
                    "children": [
                        {
                            "node_id": "img_5",
                            "role": "image",
                            "children": [
                                {"role": "caption", "text": "Latency trend chart"},
                            ],
                        },
                        {
                            "node_id": "tab_7",
                            "role": "table",
                            "data": [
                                {"Borough": "Bronx", "% Change": "-5%"},
                                {"Borough": "Queens", "% Change": "-3%"},
                            ],
                            "children": [
                                {"role": "caption", "text": "Latency changes by borough"},
                            ],
                        },
                        {
                            "node_id": "lst_7",
                            "role": "list",
                            "items": [
                                {"text": "First synthetic question"},
                                {"text": "Second synthetic question"},
                                {"text": "Third synthetic question"},
                            ],
                        },
                    ],
                }
            ],
        }
        (base / "doctree.mm.json").write_text(json.dumps(doctree), encoding="utf-8")

        dense_records = [
            {
                "node_id": "sec_10#h",
                "variant": "heading",
                "dense_text": "Results heading latency.",
            },
            {
                "node_id": "sec_10#g",
                "variant": "gist",
                "dense_text": "Latency improves significantly over time.",
            },
            {
                "node_id": "tab_7",
                "variant": "table",
                "dense_text": "Latency percentages table.",
            },
        ]
        with (base / "dense_coarse.jsonl").open("w", encoding="utf-8") as f:
            for rec in dense_records:
                f.write(json.dumps(rec) + "\n")

        sparse_records = [
            {
                "id": "sec_10",
                "title": "Results accuracy latency",
                "body": "Accuracy increases while latency decreases.",
            },
            {
                "id": "tab_7",
                "title": "Latency table",
                "table_schema": "Borough; % Change",
                "body": "Bronx -5%, Queens -3%",
            }
        ]
        with (base / "sparse_coarse.jsonl").open("w", encoding="utf-8") as f:
            for rec in sparse_records:
                f.write(json.dumps(rec) + "\n")

        self.base = base

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_run_question_returns_answer_and_history(self) -> None:
        def reasoner_stub(**_):
            return json.dumps({
                "thinking": "Focus on section 10 for latency trend.",
                "answer": "Latency trend improved",
                "confidence": 0.8,
                "support_nodes": ["sec_10"],
                "reasoning": "Section 10 discusses latency improvements."
            })

        def strategy_stub(*, question: str, config):
            return json.dumps({
                "thinking": "Use dense search on section gist to capture latency discussion.",
                "strategy": "SINGLE",
                "steps": [
                    {
                        "tool": "dense_search",
                        "args": {
                            "query": "latency results",
                            "queries": ["latency trend", "results latency"],
                            "view": "section#gist"
                        }
                    },
                    {
                        "tool": "sparse_search",
                        "args": {
                            "keywords": ["latency trend", "results section"],
                            "view": "section#heading"
                        }
                    }
                ],
                "confidence": 0.8,
                "retrieval_keys": ["latency trend", "latency results"]
            })

        def retriever_stub(*, tool: str, query: str, candidates, config):
            top = candidates[0]
            return json.dumps({
                "thinking": f"{top['node_id']} mentions latency; returning it first.",
                "results": [
                    {"node_id": top["node_id"], "base_id": top.get("base_id", top["node_id"]), "score": 0.95}
                ]
            })

        image_stub = lambda evidence: {"summary": f"vision summary {evidence.node_id}", "labels": ["line chart"]}
        answer, orchestrator, plans = run_question(
            self.base,
            "请解释结果中的 latency 趋势",
            max_iterations=2,
            use_llm=True,
            llm_callable=reasoner_stub,
            image_analyzer=image_stub,
            strategy_llm_callable=strategy_stub,
            retriever_llm_callable=retriever_stub,
        )

        self.assertIn("latency", answer.answer.lower())
        self.assertTrue(orchestrator.memory.strategy_history)
        self.assertTrue(plans)
        self.assertTrue(orchestrator.memory.retrieval_cache)
        self.assertTrue(orchestrator.memory.observations)
        any_table = any("structured_table" in obs.payload for obs in orchestrator.memory.observations.values())
        self.assertTrue(any_table)
        store = build_observer_store(self.base)
        self.assertIn("lst_7", store)
        self.assertIn("structured_list", store["lst_7"].extra)
        self.assertTrue(orchestrator.memory.strategy_history[0].retrieval_keys)

    def test_run_question_with_llm_stub(self) -> None:
        def reasoner_stub(**kwargs):
            return json.dumps(
                {
                    "thinking": "Section 10 summarises latency decrease.",
                    "answer": "Latency decreased by 5%",
                    "confidence": 0.95,
                    "support_nodes": ["sec_10"],
                    "reasoning": "Section 10 states latency is down by 5%."
                }
            )

        def strategy_stub(*, question: str, config):
            return json.dumps({
                "thinking": "Jump to results heading then dense search the gist.",
                "strategy": "COMPOSITE",
                "steps": [
                    {
                        "tool": "dense_search",
                        "args": {
                            "query": "latency results",
                            "queries": ["latency decreased", "results latency"],
                            "view": "section#gist"
                        }
                    },
                    {
                        "tool": "sparse_search",
                        "args": {
                            "keywords": ["latency % change", "latency table"],
                            "view": "section#child"
                        }
                    }
                ],
                "confidence": 0.85,
                "notes": "Combines semantic and field-oriented retrieval",
                "retrieval_keys": ["latency results", "latency % change"]
            })

        def retriever_stub(*, tool: str, query: str, candidates, config):
            ordered = []
            for idx, cand in enumerate(candidates[:2]):
                ordered.append({
                    "node_id": cand["node_id"],
                    "base_id": cand.get("base_id", cand["node_id"]),
                    "score": 0.9 - idx * 0.1,
                })
            return json.dumps({
                "thinking": "Top candidates mention latency changes; ranking accordingly.",
                "results": ordered,
            })

        image_stub = lambda evidence: {"summary": f"vision summary {evidence.node_id}", "labels": ["line chart"]}
        answer, orchestrator, plans = run_question(
            self.base,
            "Explain latency",
            max_iterations=1,
            use_llm=True,
            llm_callable=reasoner_stub,
            image_analyzer=image_stub,
            strategy_llm_callable=strategy_stub,
            retriever_llm_callable=retriever_stub,
        )

        self.assertIn("decreased", answer.answer.lower())
        self.assertTrue(orchestrator.memory.strategy_history)
        self.assertTrue(plans)
        store = build_observer_store(self.base)
        self.assertIn("structured_list", store["lst_7"].extra)
        self.assertTrue(orchestrator.memory.strategy_history[0].retrieval_keys)


if __name__ == "__main__":
    unittest.main()
