from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from src.utils.llm_clients import gpt_llm_call, qwen_llm_call

from ..router import RouterDecision
from .plan_ir import PlanIR, PlanNode, PlanStage


TOOL_WHITELIST = [
    "toc_anchor.locate",
    "structure.children",
    "structure.expand",
    "bm25_node.search",
    "dense_node.search",
    "table_index.search",
    "chart_index.search",
    "page_locator.locate",
    "figure_finder.find_regions",
    "chart_screener.screen",
    "extract.table",
    "extract.column",
    "extract.chart_read_axis",
    "extract.regex",
    "compute.filter",
    "compute.eval",
    "pack.mmr_knapsack",
    "vlm.answer",
]

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "toc_anchor.locate": "Locate DocTree sections whose headings or ancestor path match the provided keywords.",
    "structure.children": "Return child nodes (e.g., subheadings, tables, images) of a given section; use for hierarchical navigation.",
    "structure.expand": "Expand a node's neighbourhood (parents/siblings/children) to gather related context nodes.",
    "bm25_node.search": "Sparse lexical search over DocTree nodes using keyword queries; best for exact term matches.",
    "dense_node.search": "Semantic dense retrieval over DocTree summaries; useful for topical similarity.",
    "table_index.search": "Retrieve table nodes matching column/value hints, units, or filters.",
    "chart_index.search": "Retrieve chart/figure nodes based on entities, metrics, units, or years.",
    "page_locator.locate": "List nodes/images that appear on specific logical/physical pages.",
    "figure_finder.find_regions": "Return figure/image regions (with paths/metadata) for given pages or labels.",
    "chart_screener.screen": "Inspect figure regions to flag whether they contain charts and expose detection signals (e.g., has_chart).",
    "extract.table": "Parse structured tables into rows/columns (use when table text needs normalization).",
    "extract.column": "Extract columns/rows from table hits using value/unit/year hints (returns structured rows).",
    "extract.chart_read_axis": "Read quantitative series from chart descriptions/captions produced by retrieval steps.",
    "extract.regex": "Apply regular-expression extraction over text hits (e.g., pulling numbers, dates, tokens).",
    "compute.filter": "Filter structured rows by comparator/threshold/unit (supports inequality conditions).",
    "compute.eval": "Aggregate numeric rows (sum/avg/max/min/count) from earlier extraction/filter results.",
    "pack.mmr_knapsack": "Select a budgeted subset of hits using MMR to balance relevance and diversity before reasoning.",
    "vlm.answer": "Call the visual-language model to reason over image regions returned by figure tools.",
}

PLAN_SCHEMA = """Plan object schema (JSON):\n{\n  \"plan_id\": string,\n  \"stages\": [\n    {\n      \"name\": string,\n      \"evidence_type\": \"text|layout|table|graphics\",\n      \"run_if\": string|null,\n      \"graph\": [\n        {\n          \"id\": string,\n          \"op\": string,\n          \"args\": object,\n          \"save_as\": string|null,\n          \"uses\": [string]|null,\n          \"when\": string|null,\n          \"meta\": object|null\n        }\n      ] (<= 6 items)\n    }\n  ] (1-2 items),\n  \"final\": string,\n  \"constraints\": {\"format\": \"int|float|string|list\"},\n  \"meta\": object (optional)\n}\n"""

FEW_SHOT_EXAMPLES = [
    {
        "task": "Heading lookup",
        "plan": {
            "plan_id": "example-layout",
            "stages": [
                {
                    "name": "layout_stage",
                    "evidence_type": "layout",
                    "run_if": None,
                    "graph": [
                        {
                            "id": "L",
                            "op": "toc_anchor.locate",
                            "args": {"keywords": ["Areas for future research"]},
                            "save_as": "SEC",
                        },
                        {
                            "id": "C",
                            "op": "structure.children",
                            "args": {"from": "L", "level": "subheading"},
                            "save_as": "SUBS",
                            "uses": ["L"],
                        },
                    ],
                }
            ],
            "final": "reasoner",
            "constraints": {"format": "string"},
        },
    },
    {
        "task": "Table percentage",
        "plan": {
            "plan_id": "example-table",
            "stages": [
                {
                    "name": "table_stage",
                    "evidence_type": "table",
                    "run_if": None,
                    "params": {
                        "pack": {"limit": 6, "per_page_limit": 2},
                        "k_nodes": 6,
                    },
                    "graph": [
                        {
                            "id": "T",
                            "op": "table_index.search",
                            "args": {
                                "keys": {"columns": ["% land area rezoned", "Bronx"]},
                                "filters": {"unit": "%", "years": [2003, 2004, 2005, 2006, 2007]},
                            },
                            "save_as": "TAB",
                        },
                        {
                            "id": "E",
                            "op": "extract.column",
                            "args": {
                                "from": "T",
                                "unit": "%",
                                "years": [2003, 2004, 2005, 2006, 2007],
                                "value_hints": ["%", "rezoned"],
                            },
                            "save_as": "VALS",
                            "uses": ["T"],
                        },
                        {
                            "id": "PK",
                            "op": "pack.mmr_knapsack",
                            "args": {"source": ["T", "E"], "limit": 6, "per_page_limit": 2},
                            "save_as": "PACKED",
                            "uses": ["T", "E"],
                        },
                        {
                            "id": "AGG",
                            "op": "compute.eval",
                            "args": {"source": "E", "operation": "max", "round": 1},
                            "save_as": "MAX_VAL",
                            "uses": ["E"],
                        },
                    ],
                },
                {
                    "name": "chart_fallback",
                    "evidence_type": "graphics",
                    "run_if": "prev_quality < 0.75",
                    "graph": [
                        {
                            "id": "CF",
                            "op": "chart_index.search",
                            "args": {
                                "keys": {
                                    "entities": ["Bronx"],
                                    "metric": "% land area rezoned",
                                    "years": [2003, 2007],
                                }
                            },
                            "save_as": "HFIG",
                        },
                        {
                            "id": "EF",
                            "op": "extract.chart_read_axis",
                            "args": {
                                "from": "CF",
                                "entities": ["Bronx"],
                                "years": [2003, 2007],
                                "unit": "%",
                            },
                            "save_as": "VALS",
                            "uses": ["CF"],
                        },
                    ],
                },
            ],
            "final": "reasoner",
            "constraints": {"format": "float"},
        },
    },
    {
        "task": "Visual sum",
        "plan": {
            "plan_id": "example-graphics",
            "stages": [
                {
                    "name": "graphics_stage",
                    "evidence_type": "graphics",
                    "run_if": None,
                    "graph": [
                        {
                            "id": "P2",
                            "op": "page_locator.locate",
                            "args": {"pages": [2]},
                            "save_as": "P2P",
                        },
                        {
                            "id": "F2",
                            "op": "figure_finder.find_regions",
                            "args": {"pages": [2], "want": ["image", "figure", "chart"]},
                            "save_as": "ROIS2",
                            "uses": ["P2"],
                        },
                        {
                            "id": "V2",
                            "op": "vlm.answer",
                            "args": {
                                "images": "ROIS2",
                                "question": "How many cars are on page 2? Return an integer.",
                            },
                            "save_as": "CARS",
                            "uses": ["F2"],
                        },
                        {
                            "id": "P4",
                            "op": "page_locator.locate",
                            "args": {"pages": [4]},
                            "save_as": "P4P",
                        },
                        {
                            "id": "F4",
                            "op": "figure_finder.find_regions",
                            "args": {"pages": [4], "want": ["image", "figure", "chart"]},
                            "save_as": "ROIS4",
                            "uses": ["P4"],
                        },
                        {
                            "id": "V4",
                            "op": "vlm.answer",
                            "args": {
                                "images": "ROIS4",
                                "question": "How many bars are on page 4? Return an integer.",
                            },
                            "save_as": "BARS",
                            "uses": ["F4"],
                        },
                    ],
                }
            ],
            "final": "reasoner",
            "constraints": {"format": "int"},
        },
    },
]


def sample_plans(decision: RouterDecision, stage_order: List[str], k: int = 2) -> List[PlanIR]:
    if k <= 0:
        raise ValueError("k must be >= 1")
    system_prompt = _build_system_prompt()
    user_payload = _build_user_payload(decision, stage_order, k)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    response = _call_llm(messages, backend=user_payload.get("backend"))
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Planner LLM returned non-JSON response: {response[:200]}...") from exc

    if isinstance(data, dict) and "plans" in data:
        plans_data = data.get("plans", [])
    elif isinstance(data, dict) and "plan" in data:
        plans_data = [data.get("plan")]
    elif isinstance(data, dict):
        plans_data = [data]
    elif isinstance(data, list):
        plans_data = data
    else:
        snippet = response[:200].replace("\n", " ")
        raise RuntimeError(
            f"Planner LLM response must be a JSON array or object with 'plans'; got: {snippet}"
        )

    if not isinstance(plans_data, list) or not plans_data:
        raise RuntimeError("Planner LLM returned an empty plan list.")

    parsed: List[PlanIR] = []
    for item in plans_data[:k]:
        plan = _parse_plan_ir(item)
        parsed.append(plan)
    if not parsed:
        raise RuntimeError("Planner LLM produced no parsable plans.")
    return parsed


def _build_system_prompt() -> str:
    ops_lines = []
    for op in TOOL_WHITELIST:
        desc = TOOL_DESCRIPTIONS.get(op)
        if desc:
            ops_lines.append(f"- {op}: {desc}")
        else:
            ops_lines.append(f"- {op}")
    ops = "\n".join(ops_lines)
    examples = []
    for ex in FEW_SHOT_EXAMPLES:
        examples.append(
            f"Example ({ex['task']}):\n" + json.dumps(ex["plan"], ensure_ascii=False, indent=2)
        )
    instruction = (
        "You are the planning module of a multimodal DocTree agent.\n"
        "Generate executable Plan-IR JSON objects describing tool execution graphs.\n"
        "Return a JSON array containing 1 to k plans. Each plan must follow the schema.\n"
        "Do NOT output explanations or prose. Only JSON.\n"
        "Respect all constraints and keep each stage graph within 6 nodes and 2 stages per plan.\n"
        "The tools should focus on retrieving candidate nodes or evidence; Reasoner will synthesize the final answer."
    )
    return (
        f"{instruction}\n\nAllowed tool ops:\n{ops}\n\n{PLAN_SCHEMA}\n"
        + "\n".join(examples)
    )


def _build_user_payload(decision: RouterDecision, stage_order: List[str], k: int) -> Dict[str, Any]:
    signals = decision.signals
    payload: Dict[str, Any] = {
        "question": decision.query,
        "router_query_type": decision.query_type,
        "expected_format": signals.expected_format or "string",
        "evidence_hint": signals.evidence_hint or "unknown",
        "stage_order_hint": stage_order,
        "candidate_count": k,
        "signals": {
            "page_hint": signals.page_hint,
            "figure_hint": signals.figure_hint,
            "table_hint": signals.table_hint,
            "objects": signals.objects,
            "units": signals.units,
            "years": signals.years,
            "operations": signals.operations,
            "section_cues": signals.section_cues,
            "keywords": signals.keywords,
            "mentions": signals.mentions,
        },
    }
    return payload


def _call_llm(messages: List[Dict[str, Any]], backend: Optional[str] = None, model: Optional[str] = None) -> str:
    backend = (backend or "gpt").lower()
    if backend == "qwen":
        return qwen_llm_call(messages, model=model or "qwen-plus", json_mode=True)
    return gpt_llm_call(messages, model=model or "gpt-4o", json_mode=True)


def _parse_plan_ir(obj: Any) -> PlanIR:
    if not isinstance(obj, dict):
        raise RuntimeError("Plan entry must be a JSON object")
    plan_id = str(obj.get("plan_id") or f"plan-{uuid.uuid4().hex[:8]}")
    final = obj.get("final")
    if not isinstance(final, str) or not final.strip():
        raise RuntimeError("Plan final field must be a non-empty string")
    constraints = obj.get("constraints") or {}
    if not isinstance(constraints, dict):
        raise RuntimeError("Plan constraints must be an object")
    stages_data = obj.get("stages")
    if not isinstance(stages_data, list) or not stages_data:
        raise RuntimeError("Plan must contain at least one stage")
    if len(stages_data) > 2:
        raise RuntimeError("Plan may not contain more than 2 stages")

    stages: List[PlanStage] = []
    for stage_obj in stages_data:
        if not isinstance(stage_obj, dict):
            raise RuntimeError("Stage definition must be an object")
        name = stage_obj.get("name") or f"stage-{len(stages)+1}"
        evidence_type = stage_obj.get("evidence_type")
        if not isinstance(evidence_type, str) or evidence_type.lower() not in {"text", "layout", "table", "graphics"}:
            raise RuntimeError("Stage evidence_type must be one of text|layout|table|graphics")
        run_if = stage_obj.get("run_if")
        if run_if is not None and not isinstance(run_if, str):
            run_if = None
        params = stage_obj.get("params")
        if not isinstance(params, dict):
            params = {}
        graph_data = stage_obj.get("graph")
        if not isinstance(graph_data, list) or not graph_data:
            raise RuntimeError("Stage graph must contain at least one node")
        if len(graph_data) > 6:
            raise RuntimeError("Stage graph exceeds 6 nodes")
        nodes: List[PlanNode] = []
        for node_obj in graph_data:
            nodes.append(_parse_plan_node(node_obj))
        stages.append(PlanStage(name=str(name), evidence_type=evidence_type.lower(), run_if=run_if, params=params, graph=nodes))

    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    return PlanIR(
        plan_id=plan_id,
        stages=stages,
        final=final,
        constraints=constraints,
        meta=meta,
    )


def _parse_plan_node(node_obj: Any) -> PlanNode:
    if not isinstance(node_obj, dict):
        raise RuntimeError("Plan node must be an object")
    node_id = node_obj.get("id")
    op = node_obj.get("op")
    args = node_obj.get("args")
    if not isinstance(node_id, str) or not node_id.strip():
        raise RuntimeError("Plan node missing id")
    if not isinstance(op, str) or op not in TOOL_WHITELIST:
        raise RuntimeError(f"Plan node op {op!r} is not allowed")
    if not isinstance(args, dict):
        raise RuntimeError("Plan node args must be an object")
    save_as = node_obj.get("save_as")
    if save_as is not None and not isinstance(save_as, str):
        save_as = None
    uses = node_obj.get("uses")
    if uses is None:
        uses_list: List[str] = []
    elif isinstance(uses, list):
        uses_list = [str(u) for u in uses if isinstance(u, str)]
    elif isinstance(uses, str):
        uses_list = [uses]
    else:
        raise RuntimeError("Plan node uses must be string or list of strings")
    when = node_obj.get("when")
    if when is not None and not isinstance(when, str):
        when = None
    meta = node_obj.get("meta") if isinstance(node_obj.get("meta"), dict) else {}
    return PlanNode(
        id=node_id,
        op=op,
        args=args,
        save_as=save_as,
        uses=uses_list or None,
        when=when,
        meta=meta,
    )


__all__ = ["sample_plans"]
