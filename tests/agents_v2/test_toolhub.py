import unittest

from agents_v2.retriever import build_stub_resources
from agents_v2.toolhub import ToolCall, ToolExecutor, ToolRegistry, ToolResult
from agents_v2.toolhub.adapters import compute as compute_adapter
from agents_v2.toolhub.adapters import pack_mmr
from agents_v2.toolhub.adapters import figure_finder, chart_screener
from agents_v2.toolhub.adapters import chart_index
from agents_v2.toolhub.types import CommonHit


class ToolHubTests(unittest.TestCase):
    def test_registry_executes_registered_tool(self) -> None:
        registry = ToolRegistry()

        def sample_tool(call: ToolCall) -> ToolResult:
            return ToolResult(
                status="ok",
                info={"echo": call.args},
                metrics={"custom": 1},
            )

        registry.register("echo.tool", sample_tool)
        executor = ToolExecutor(registry)
        result = executor.run(ToolCall(tool_id="echo.tool", args={"msg": "hi"}))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["echo"]["msg"], "hi")
        self.assertIn("latency_ms", result.metrics)
        self.assertEqual(result.metrics["attempt"], 1)

    def test_missing_tool_raises(self) -> None:
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        with self.assertRaises(KeyError):
            executor.run(ToolCall(tool_id="missing.tool", args={}))

    def test_pack_mmr_knapsack_limits_hits(self) -> None:
        context_result = ToolResult(
            status="ok",
            hits=[
                CommonHit(
                    node_id="sec_1",
                    evidence_type="text",
                    score=0.9,
                    provenance={"page_idx": 1},
                    modality="text",
                ),
                CommonHit(
                    node_id="sec_2",
                    evidence_type="text",
                    score=0.8,
                    provenance={"page_idx": 1},
                    modality="text",
                ),
            ],
        )
        call = ToolCall(
            tool_id="pack.mmr_knapsack",
            args={
                "source": "S1",
                "limit": 1,
                "per_page_limit": 1,
                "_context": {"S1": context_result},
            },
        )
        result = pack_mmr.mmr_knapsack(call)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0].node_id, "sec_1")
        self.assertAlmostEqual(result.info.get("coverage", 0.0), 1.0)

    def test_compute_eval_sum(self) -> None:
        matches = [
            {"node_id": "tab_1", "value": 10, "unit": "%"},
            {"node_id": "tab_2", "value": 5, "unit": "%"},
        ]
        context = {
            "F": ToolResult(
                status="ok",
                info={"matches": matches},
            )
        }
        call = ToolCall(
            tool_id="compute.eval",
            args={
                "source": "F",
                "operation": "sum",
                "_context": context,
            },
        )
        result = compute_adapter.eval(call)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.info.get("result"), 15)
        self.assertEqual(len(result.hits), 2)

    def test_figure_finder_uses_resources(self) -> None:
        resources = build_stub_resources(
            label_pairs=[("Figure 1", "img_1")],
            page_pairs=[(1, "img_1")],
            text_pairs=[],
            node_roles={"img_1": "image"},
            image_paths={"img_1": "/tmp/fig1.png"},
            image_meta={"img_1": {"caption": "Line chart sales", "description": "statistical chart", "page_idx": 1}},
            node_pages={"img_1": 1},
            figure_spans={
                "img_1": [
                    {
                        "node_id": "img_1",
                        "role": "axis_label",
                        "dense_text": "Year",
                        "chart_type": "statistical",
                    }
                ]
            },
        )
        call = ToolCall(
            tool_id="figure_finder.find_regions",
            args={"pages": [1], "_resources": resources},
        )
        result = figure_finder.find_regions(call)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.hits)
        roi = result.info["rois"][0]
        self.assertEqual(roi["node_id"], "img_1")
        self.assertTrue(roi["spans"])

        # chart screener should detect chart based on spans/caption
        context = {"FIG": result}
        chart_call = ToolCall(
            tool_id="chart_screener.screen",
            args={"source": "FIG", "_context": context, "_resources": resources},
        )
        chart_result = chart_screener.screen(chart_call)
        self.assertEqual(chart_result.status, "ok")
        self.assertTrue(chart_result.hits)
        detection = chart_result.info["results"][0]
        self.assertTrue(detection["has_chart"])
        self.assertIn("axis_label", detection["span_roles"])

    def test_table_index_search_without_manager(self) -> None:
        resources = build_stub_resources(
            label_pairs=[],
            page_pairs=[],
            text_pairs=[],
            tables={
                "tab_a": {
                    "columns": ["Operator", "Subscribers 2014"],
                    "rows": [
                        ["Telkomsel", "132"],
                        ["XL", "68"],
                        ["Indosat", "63"],
                    ],
                    "caption": "Telecom operators 2014",
                    "preview": "Operators with subscriber counts",
                }
            },
        )
        call = ToolCall(
            tool_id="table_index.search",
            args={
                "_resources": resources,
                "keys": {"columns": ["Operator", "Subscribers"], "keywords": ["Telecom"]},
            },
        )
        from agents_v2.toolhub.adapters import table_index  # local import to avoid circular during test collection

        result = table_index.search(call)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.hits), 1)
        hit = result.hits[0]
        self.assertEqual(hit.node_id, "tab_a")

    def test_chart_index_bm25_with_page_hint(self) -> None:
        resources = build_stub_resources(
            label_pairs=[],
            page_pairs=[(9, "img_42"), (3, "img_2")],
            text_pairs=[],
            node_roles={"img_42": "image", "img_2": "image"},
            figure_spans={
                "img_42": [
                    {
                        "dense_text": "Telecom operators 2013-2014 subscriber share Telkomsel XL Indosat",
                        "role": "figure_description",
                    }
                ],
                "img_2": [
                    {
                        "dense_text": "Mascot illustration",
                        "role": "figure_caption",
                    }
                ],
            },
            image_meta={"img_42": {"caption": "Telecom Operators – 2013-2014"}},
            node_pages={"img_42": 9, "img_2": 3},
        )
        call = ToolCall(
            tool_id="chart_index.search",
            args={
                "_resources": resources,
                "keys": {
                    "keywords": ["Telecom Operator", "Number of Subscribers"],
                    "years": [2013, 2014],
                },
                "filters": {"page_idx": 9},
            },
        )
        result = chart_index.search(call)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.hits)
        self.assertEqual(result.hits[0].node_id, "img_42")


if __name__ == "__main__":
    unittest.main()
