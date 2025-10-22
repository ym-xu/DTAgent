import json
from pathlib import Path

import pytest

from src.index.node_summarizer import build_summary


RET_KEYS = [
    "summary",
    "coarse_legacy",
    "leaf_legacy",
    "dense_coarse",
    "dense_leaf",
    "sparse_coarse",
    "sparse_leaf",
    "table_cells",
    "figure_spans",
    "graph_edges",
    "heading_index",
    "metadata",
    "id_maps",
]


@pytest.fixture
def summary_outputs(tmp_path: Path) -> dict:
    table_html = """
    <table>
      <thead>
        <tr><th>Region</th><th>Latency (ms)</th><th>Year</th></tr>
      </thead>
      <tbody>
        <tr><td>Staten Island</td><td>120</td><td>2021</td></tr>
        <tr><td>Queens</td><td>130</td><td>2020</td></tr>
      </tbody>
    </table>
    """
    doctree = {
        "doc_id": "doc-test",
        "node_id": "root",
        "children": [
            {
                "type": "section",
                "node_id": "sec_1",
                "title": "Introduction",
                "level": 1,
                "page_idx": 1,
                "children": [
                    {
                        "type": "text",
                        "role": "paragraph",
                        "node_id": "p_1",
                        "page_idx": 1,
                        "text": "This study introduces the dataset and outlines goals.",
                        "read_order_idx": 1,
                    }
                ],
            },
            {
                "type": "section",
                "node_id": "sec_2",
                "title": "Experimental Results",
                "level": 1,
                "page_idx": 2,
                "children": [
                    {
                        "type": "table",
                        "node_id": "tab_1",
                        "title": "Table 1: Latency breakdown",
                        "page_idx": 2,
                        "data": table_html,
                        "read_order_idx": 3,
                        "children": [
                            {
                                "role": "caption",
                                "node_id": "cap_tab_1",
                                "page_idx": 2,
                                "text": "Table 1 compares latency across boroughs.",
                            }
                        ],
                    },
                    {
                        "type": "image",
                        "node_id": "img_1",
                        "title": "Figure 1: Accuracy trend",
                        "kind": "statistical",
                        "page_idx": 3,
                        "description": "A line chart showing accuracy (%) over years.",
                        "legend": ["Model A", "Model B"],
                        "axis_labels": {"x": "Year", "y": "Accuracy (%)"},
                        "ocr_text": "2019 2020 2021 Accuracy 90%",
                        "read_order_idx": 4,
                        "children": [
                            {
                                "role": "caption",
                                "node_id": "cap_img_1",
                                "page_idx": 3,
                                "text": "Figure 1 illustrates accuracy improvement over time.",
                            }
                        ],
                    },
                ],
            },
        ],
    }
    doc_path = tmp_path / "doctree.mm.json"
    doc_path.write_text(json.dumps(doctree, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = build_summary(
        doctree_path=str(doc_path),
        out_dir=str(tmp_path),
        doc_id="doc-test",
        model="heuristic-test",
        max_tokens=128,
        include_leaves=True,
        use_llm=False,
        llm_backend="auto",
        llm_model="test",
    )
    return {key: value for key, value in zip(RET_KEYS, outputs)}


def test_dense_coarse_variants(summary_outputs: dict) -> None:
    dense_coarse = summary_outputs["dense_coarse"]
    sec2_variants = {rec.get("variant") for rec in dense_coarse if rec["node_id"].startswith("sec_2#")}
    assert {"heading", "gist", "child", "path"}.issubset(sec2_variants)
    for rec in dense_coarse:
        if rec.get("variant") == "child":
            assert rec["affordances"]["supports_ROUTE_TO_table"] is True


def test_dense_leaf_contains_paragraph(summary_outputs: dict) -> None:
    dense_leaf = summary_outputs["dense_leaf"]
    paragraphs = [rec for rec in dense_leaf if rec["role"] == "paragraph"]
    assert paragraphs, "paragraph leaf chunks should be generated"
    assert "dataset" in paragraphs[0]["raw_text"].lower()


def test_sparse_views_capture_schema(summary_outputs: dict) -> None:
    sparse_coarse = summary_outputs["sparse_coarse"]
    table_doc = next(rec for rec in sparse_coarse if rec["id"] == "tab_1")
    assert "Region" in table_doc["table_schema"]
    sparse_leaf = summary_outputs["sparse_leaf"]
    assert any("dataset" in (rec.get("body") or "").lower() for rec in sparse_leaf)


def test_table_cells_records_from_html(summary_outputs: dict) -> None:
    rows = [rec for rec in summary_outputs["table_cells"] if rec["node_id"] == "tab_1"]
    assert len(rows) == 2
    first = rows[0]
    assert first["role"] == "table_row"
    assert first["columns"][0]["name"] == "Region"
    assert any(col.get("unit") == "ms" for col in first["columns"])
    assert first["filters"]["label"].startswith("Table 1")


def test_figure_spans_include_caption_and_axes(summary_outputs: dict) -> None:
    span_roles = {span["role"] for span in summary_outputs["figure_spans"]}
    assert {"figure_caption", "figure_legend", "axis_label"}.issubset(span_roles)
    span_with_years = next(span for span in summary_outputs["figure_spans"] if span["role"] == "figure_ocr")
    assert "2020" in span_with_years["dense_text"]


def test_heading_index_contains_children(summary_outputs: dict) -> None:
    heading_index = summary_outputs["heading_index"]
    assert heading_index["heading_titles"]["sec_1"] == "Introduction"
    assert "tab_1" in heading_index["heading_children"]["sec_2"]
    assert any("sec_2" in ids for ids in heading_index["keyword_map"].values())


def test_metadata_counts_align(summary_outputs: dict) -> None:
    metadata = summary_outputs["metadata"]
    counts = metadata["counts"]
    assert metadata["doc_id"] == "doc-test"
    assert counts["dense_coarse"] == len(summary_outputs["dense_coarse"])
    assert counts["sparse_coarse"] == len(summary_outputs["sparse_coarse"])
    assert counts["table_cells"] == len(summary_outputs["table_cells"])
    assert counts["figure_spans"] == len(summary_outputs["figure_spans"])
