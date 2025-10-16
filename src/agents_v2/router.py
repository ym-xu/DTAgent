"""
QuestionRouter
==============

使用 LLM 对用户问题进行类型识别与信号抽取，
为后续 Planner/Reader 提供结构化的查询卡片。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Protocol, Tuple

from .schemas import (
    RouterConstraints,
    RouterDecision,
    RouterRisk,
    RouterSignals,
)

logger = logging.getLogger(__name__)


class RouterLLMCallable(Protocol):
    def __call__(self, *, question: str, config: "RouterLLMConfig") -> str: ...


@dataclass
class RouterLLMConfig:
    backend: str = "gpt"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_response_tokens: int = 512


ALLOWED_QUERY_TYPES = {
    "text",
    "definition",
    "table",
    "chart",
    "visual_count",
    "visual_presence",
    "visual_compare",
    "visual_trend",
    "metadata",
    "metadata_visual",
    "list",
    "cross_page",
    "numeric_compute",
    "price_total",
    "multi_modal",
    "hybrid",
    "map_spatial",
}

VISUAL_QUERY_TYPES = {
    "visual_count",
    "visual_presence",
    "visual_compare",
    "visual_trend",
    "metadata_visual",
}


ROUTER_PROMPT = """You are the Router of a DocTree multimodal QA agent.
Your job is to classify the user question, extract structured signals, and decide the toolchains that will be required.

Allowed query_type values (pick exactly one, do NOT invent new types):
- text : answer found in narrative text or headings.
- definition : acronym/term definition lookup.
- table : read structured values from a table.
- chart : interpret chart or figure captions/axes (non-visual).
- visual_count : count objects in an image/figure (requires VLM).
- visual_presence : detect whether an object/logo/map feature exists on a page.
- visual_compare : compare counts/attributes between visuals.
- visual_trend : describe trend/pattern visible in figures.
- metadata : document-level metadata (author, publication date, etc.).
- metadata_visual : visual metadata such as photos/logos per page.
- list : enumerate items (e.g., list chapters, bullet points).
- cross_page : combine information across distinct pages/sections.
- numeric_compute : perform arithmetic (sum/diff/max/min/avg) on retrieved numbers.
- price_total : multiply quantities and unit prices, compute totals.
- multi_modal : reasoning needs text + table/chart together.
- hybrid : fallback when both dense & sparse textual cues are crucial.
- map_spatial : questions about map regions or spatial color coding.

Return ONLY JSON following this schema exactly (include up to 3 candidate query types ranked by confidence):
{
  "query": string,                     # original question text (trimmed)
  "query_type": "...",                 # one of the predefined task types
  "signals": {
    "page_hint": [int, ...],           # optional; pages start at 1
    "figure_hint": [string, ...],      # figure labels or aliases
    "table_hint": [string, ...],
    "objects": [string, ...],          # visual targets
    "units": [string, ...],
    "years": [int, ...],
    "operations": [string, ...],       # filter/lookup/sum/diff/max/min/avg/count etc.
    "threshold": number|null,          # numeric cutoff when using filter comparisons
    "comparator": string|null,         # >, >=, <, <=, ==, !=
    "expected_format": "int|float|string|list",
    "section_cues": [string, ...],
    "keywords": [string, ...],
    "objects_scope": "figure|page|all_figures_on_page"
  },
  "candidates": [
    {"query_type": "table", "score": 0.8},
    {"query_type": "visual_presence", "score": 0.2}
  ],
  "risk": {
    "ambiguity": 0.0-1.0,
    "need_visual": boolean,
    "need_table": boolean,
    "need_chart": boolean
  },
  "constraints": {
    "allow_unanswerable": boolean,
    "must_cite": boolean
  },
  "confidence": 0.0-1.0
}

Rules:
- Populate every field; use empty arrays when no data.
- query_type must reflect the downstream reader/toolchain that best fits the question.
- ambiguity reflects how uncertain you are (0=fully certain, 1=very ambiguous).
- If the question explicitly references pages/figures/tables, fill the hints.
- expected_format should align with how the final answer should be delivered.
- Do NOT add extra fields or text outside JSON.
- Unless the user explicitly guarantees the answer exists, set constraints.allow_unanswerable to true.
- Provide a "candidates" array listing up to three likely query types with confidence scores (float in [0,1]).
- Disambiguation rubric (VERY IMPORTANT):
  * If the question asks for a percentage/value with explicit units or years (e.g., %, percent, rate, \"from 2003 to 2007\"), prioritize query_type=\"table\". If the value is explicitly tied to a figure caption, use query_type=\"chart\".
  * Use \"numeric_compute\" only when the answer requires arithmetic on multiple retrieved values (e.g., sum/diff/avg across different cells or pages).
  * If the question requests counting objects in images/figures or mentions pages with figures/photos/icons, set query_type=\"visual_count\" (also set risk.need_visual=true).
  * If risk.need_visual=true and signals.objects is non-empty, prefer a visual_* query type over textual ones.
  * signals.objects_scope is ONLY for visual tasks; keep it null otherwise.
  * If the question specifies inequalities (\"more than\", \"at least\", etc.), include \"filter\" in operations, set comparator to the mathematical symbol (\">\", \">=\", \"<\", \"<=\", \"==\", \"!=\"), and threshold to the numeric value. Add \"lookup\" alongside \"filter\" when the task is to identify which entity satisfies the condition.
- Few-shot expectations:
  * Question: \"From 2003 to 2007, what share (%) of the Bronx’s total land area was rezoned?\" → query_type=\"table\", units=[\"%\"], years=[2003,2004,2005,2006,2007], risk.need_table=true, risk.need_chart=true (fallback).
  * Question: \"Count how many research questions this paper addresses.\" → query_type=\"text\", operations=[\"count\"], expected_format=\"int\".
  * Question: \"On page 2, count the cars shown in the figures; on page 4, count the bars; what is the total when you add them together?\" → query_type=\"visual_count\", page_hint=[2,4], objects=[\"car\",\"bar\"], operations=[\"sum\"], risk.need_visual=true.
"""


def _default_router_llm_callable(*, question: str, config: RouterLLMConfig) -> str:
    from src.utils.llm_clients import gpt_llm_call, qwen_llm_call  # type: ignore

    payload = json.dumps({"question": question}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": payload},
    ]
    if config.backend == "qwen":
        return qwen_llm_call(messages, model=config.model, json_mode=True)
    result = gpt_llm_call(messages, model=config.model, json_mode=True)
    # print("router raw:", result)
    return result


@dataclass
class QuestionRouter:
    """LLM Router."""

    llm_config: RouterLLMConfig = field(default_factory=RouterLLMConfig)
    llm_callable: Optional[RouterLLMCallable] = None

    def route(self, question: str) -> RouterDecision:
        normalized = question.strip()
        if not normalized:
            raise ValueError("Question must be non-empty for routing")

        llm_call = self.llm_callable or _default_router_llm_callable
        raw = llm_call(question=normalized, config=self.llm_config)
        if not raw:
            raise RuntimeError("Router LLM returned empty response")

        data = self._parse_json(raw)
        decision = self._parse_decision(normalized, data)
        return decision

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, object]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Router LLM response is not valid JSON: {exc}") from exc

    def _parse_decision(self, question: str, data: Dict[str, object]) -> RouterDecision:
        query = self._parse_query(data.get("query"), fallback=question)
        query_type = self._parse_query_type(data.get("query_type"))
        signals = self._parse_signals(data.get("signals") or {})
        risk = self._parse_risk(data.get("risk") or {})
        constraints = self._parse_constraints(data.get("constraints") or {})
        confidence = self._parse_confidence(data.get("confidence"))
        candidates = self._parse_candidates(data.get("candidates"))

        decision = RouterDecision(
            query=query,
            query_type=query_type,
            signals=signals,
            risk=risk,
            constraints=constraints,
            confidence=confidence,
            raw=data,
            candidates=candidates,
        )

        decision = normalize_router_decision(decision)

        logger.info(
            "[Router Decision] %s",
            json.dumps(
                {
                    "query": decision.query,
                    "query_type": decision.query_type,
                    "signals": decision.signals.__dict__,
                    "risk": decision.risk.__dict__,
                    "constraints": decision.constraints.__dict__,
                    "confidence": decision.confidence,
                },
                ensure_ascii=False,
            ),
        )

        return decision

    @staticmethod
    def _parse_query(value, *, fallback: str) -> str:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        return fallback

    @staticmethod
    def _parse_query_type(value) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise RuntimeError("Router LLM must provide a non-empty query_type")

    def _parse_signals(self, payload) -> RouterSignals:
        if not isinstance(payload, dict):
            payload = {}

        def _as_int_list(key: str) -> list[int]:
            values = payload.get(key, [])
            if not isinstance(values, list):
                return []
            result: list[int] = []
            for item in values:
                try:
                    result.append(int(item))
                except (TypeError, ValueError):
                    continue
            return result

        def _as_str_list(key: str) -> list[str]:
            values = payload.get(key, [])
            if not isinstance(values, list):
                return []
            result: list[str] = []
            for item in values:
                if isinstance(item, str):
                    trimmed = item.strip()
                    if trimmed:
                        result.append(trimmed)
            return result

        def _parse_threshold_value(raw) -> Optional[float]:
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                try:
                    return float(raw.strip().replace("%", ""))
                except ValueError:
                    return None
            return None

        def _parse_comparator_value(raw) -> Optional[str]:
            if isinstance(raw, str):
                trimmed = raw.strip()
                if trimmed in {">", ">=", "<", "<=", "==", "!=", "="}:
                    return "==" if trimmed == "=" else trimmed
            return None

        expected_format = payload.get("expected_format")
        if not isinstance(expected_format, str) or not expected_format.strip():
            expected_format = None
        else:
            expected_format = expected_format.strip()

        objects_scope = payload.get("objects_scope")
        if not isinstance(objects_scope, str) or not objects_scope.strip():
            objects_scope = None
        else:
            objects_scope = objects_scope.strip()

        return RouterSignals(
            page_hint=_as_int_list("page_hint"),
            figure_hint=_as_str_list("figure_hint"),
            table_hint=_as_str_list("table_hint"),
            objects=_as_str_list("objects"),
            units=_as_str_list("units"),
            years=_as_int_list("years"),
            operations=_as_str_list("operations"),
            expected_format=expected_format,
            section_cues=_as_str_list("section_cues"),
            keywords=_as_str_list("keywords"),
            objects_scope=objects_scope,
            threshold=_parse_threshold_value(payload.get("threshold")),
            comparator=_parse_comparator_value(payload.get("comparator")),
        )

    def _parse_risk(self, payload) -> RouterRisk:
        if not isinstance(payload, dict):
            payload = {}

        def _as_bool(key: str) -> bool:
            return bool(payload.get(key))

        ambiguity = payload.get("ambiguity")
        try:
            ambiguity_value = float(ambiguity)
        except (TypeError, ValueError):
            ambiguity_value = 0.0
        ambiguity_value = max(0.0, min(1.0, ambiguity_value))

        return RouterRisk(
            ambiguity=ambiguity_value,
            need_visual=_as_bool("need_visual"),
            need_table=_as_bool("need_table"),
            need_chart=_as_bool("need_chart"),
        )

    def _parse_constraints(self, payload) -> RouterConstraints:
        if not isinstance(payload, dict):
            payload = {}
        allow_unanswerable = bool(payload.get("allow_unanswerable"))
        must_cite = bool(payload.get("must_cite"))
        return RouterConstraints(
            allow_unanswerable=allow_unanswerable,
            must_cite=must_cite,
        )

    @staticmethod
    def _parse_candidates(payload) -> List[Tuple[str, float]]:
        if not isinstance(payload, list):
            return []
        out: List[Tuple[str, float]] = []
        for item in payload[:3]:
            if isinstance(item, dict):
                qt = item.get("query_type") or item.get("type")
                score = item.get("score")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                qt, score = item[0], item[1]
            else:
                continue
            if not isinstance(qt, str):
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = 0.0
            score_f = max(0.0, min(1.0, score_f))
            out.append((qt.strip().lower(), score_f))
        return out

    @staticmethod
    def _parse_confidence(value) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))


def normalize_router_decision(decision: RouterDecision) -> RouterDecision:
    """Post-process router decision to keep query_type within supported set."""

    signals = decision.signals
    risk = decision.risk
    query_type = decision.query_type.strip().lower() if decision.query_type else "text"

    ops: List[str] = []
    seen_ops: set[str] = set()
    for raw_op in signals.operations or []:
        if not isinstance(raw_op, str):
            continue
        lowered = raw_op.strip().lower()
        if lowered and lowered not in seen_ops:
            seen_ops.add(lowered)
            ops.append(lowered)
    keywords = [kw.lower() for kw in (signals.keywords or [])]
    units = [unit.lower() for unit in (signals.units or [])]
    objects = [obj.strip() for obj in (signals.objects or []) if obj.strip()]

    textual_percentage = any(unit in {"%", "percent", "percentage"} for unit in units)
    mentions_table = any("table" in kw for kw in keywords)
    mentions_figure = any(kw in {"figure", "chart", "graph", "trend", "legend"} for kw in keywords)

    need_visual = bool(risk.need_visual or objects)
    need_table = bool(risk.need_table or textual_percentage or mentions_table)

    comparator = (signals.comparator or "").strip() if signals.comparator else None
    if comparator == "=":
        comparator = "=="
    threshold_value = signals.threshold
    if comparator and threshold_value is not None:
        for auto_op in ("filter", "lookup"):
            if auto_op not in seen_ops:
                ops.append(auto_op)
                seen_ops.add(auto_op)

    # Preference rules
    if need_visual:
        if any(op in {"count", "sum"} for op in ops) or signals.page_hint:
            query_type = "visual_count"
        elif query_type not in VISUAL_QUERY_TYPES:
            query_type = "visual_presence"
    else:
        if query_type in {"visual_count", "visual_presence", "visual_compare", "visual_trend"}:
            query_type = "text"

    if query_type in VISUAL_QUERY_TYPES and need_table and not need_visual:
        query_type = "table"
    if query_type == "table" and need_visual and not objects:
        need_visual = False
        risk = replace(risk, need_visual=False)

    if query_type in {"numeric_compute", "text"} and need_table:
        query_type = "table"

    if query_type in {"numeric_compute", "text"} and mentions_figure and not need_table:
        query_type = "chart"

    if query_type not in ALLOWED_QUERY_TYPES:
        query_type = "text"

    if query_type not in VISUAL_QUERY_TYPES and signals.objects_scope is not None:
        signals = replace(signals, objects_scope=None)

    if comparator:
        comparator = comparator if comparator in {">", ">=", "<", "<=", "==", "!="} else None
    updated_signals = replace(
        signals,
        operations=ops,
        comparator=comparator if comparator else None,
        threshold=threshold_value if threshold_value is not None else None,
    )

    return replace(decision, query_type=query_type, signals=updated_signals, risk=risk)


__all__ = ["QuestionRouter", "RouterLLMConfig", "RouterLLMCallable", "normalize_router_decision"]
