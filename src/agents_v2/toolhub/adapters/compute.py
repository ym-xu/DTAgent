"""Compute adapters."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ...schemas import RetrievalHit
from ..types import ToolCall, ToolResult


def eval(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        data={},
        metrics={},
        error="NOT_IMPLEMENTED",
    )


def filter(call: ToolCall) -> ToolResult:
    context = call.args.get("_context") or {}
    source_key = call.args.get("source") or call.args.get("source_step") or call.args.get("source_result")
    if not isinstance(source_key, str):
        return ToolResult(status="error", data={}, metrics={}, error="source step_id required")

    source_result = context.get(source_key)
    if not isinstance(source_result, ToolResult):
        return ToolResult(status="error", data={}, metrics={}, error=f"missing context for {source_key}")

    data = source_result.data if isinstance(source_result.data, dict) else {}
    rows: List[Dict[str, object]] = data.get("rows") if isinstance(data.get("rows"), list) else []
    if not rows:
        return ToolResult(status="empty", data={"matches": []}, metrics={"n_rows": 0, "source": source_key})

    comparator = _clean_comparator(call.args.get("comparator")) or ">"
    threshold = _to_number(call.args.get("threshold"))
    if threshold is None:
        return ToolResult(status="error", data={}, metrics={}, error="threshold required for compute.filter")

    comparator_fn = _COMPARATORS.get(comparator)
    if comparator_fn is None:
        return ToolResult(status="error", data={}, metrics={}, error=f"unsupported comparator {comparator}")

    expected_unit = _clean_unit(call.args.get("unit") or data.get("unit"))
    expected_year = _to_int(call.args.get("year") or data.get("year"))

    matches: List[Dict[str, object]] = []
    for row in rows:
        value = _to_number(row.get("value"))
        if value is None:
            continue
        unit = _clean_unit(row.get("unit"))
        if expected_unit and unit and expected_unit != unit:
            continue
        row_year = _to_int(row.get("year"))
        if expected_year is not None and row_year is not None and expected_year != row_year:
            continue
        if comparator_fn(value, threshold):
            matches.append(row)

    status = "ok" if matches else "empty"
    metrics = {
        "n_rows": len(rows),
        "n_match": len(matches),
        "comparator": comparator,
        "threshold": threshold,
        "suggest_chart": status == "empty",
    }
    hits: List[RetrievalHit] = []
    for match in matches:
        node_id = match.get("node_id")
        if isinstance(node_id, str):
            hits.append(
                RetrievalHit(
                    node_id=node_id,
                    score=1.0,
                    tool="compute.filter",
                    metadata={
                        "label": match.get("label"),
                        "value": match.get("value"),
                        "unit": match.get("unit"),
                        "year": match.get("year"),
                    },
                )
            )

    payload = {
        "matches": matches,
        "threshold": threshold,
        "comparator": comparator,
        "unit": expected_unit,
        "year": expected_year,
        "source_step": source_key,
        "hits": hits,
    }
    return ToolResult(status=status, data=payload, metrics=metrics)


def _to_number(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _clean_comparator(value) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        if text == "=":
            text = "=="
        if text in _COMPARATORS:
            return text
    return None


def _clean_unit(value) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _to_int(value) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return int(text)
            except ValueError:
                return None
    return None


_COMPARATORS: Dict[str, Callable[[float, float], bool]] = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


__all__ = ["eval", "filter"]
