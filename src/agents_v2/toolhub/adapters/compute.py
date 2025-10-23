"""Compute adapters."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..types import CommonError, CommonHit, ToolCall, ToolResult


def eval(call: ToolCall) -> ToolResult:
    context = call.args.get("_context") or {}
    source_key = call.args.get("source") or call.args.get("from") or call.args.get("source_step")
    if not isinstance(source_key, str):
        return ToolResult(
            status="error",
            errors=[CommonError(error="INVALID_ARGS", message="source step_id required", retryable=False)],
            error="source step_id required",
        )

    source_result = context.get(source_key)
    if not isinstance(source_result, ToolResult):
        return ToolResult(
            status="error",
            errors=[CommonError(error="MISSING_CONTEXT", message=f"missing context for {source_key}", retryable=False)],
            error="missing context",
        )

    operation = str(call.args.get("operation") or "sum").lower()
    field = str(call.args.get("field") or "value")
    round_digits = call.args.get("round")
    scale = call.args.get("scale")

    records = _collect_numeric_records(source_result, field=field)
    if not records and operation != "count":
        return ToolResult(
            status="empty",
            hits=[],
            metrics={"n_values": 0, "operation": operation, "source": source_key},
            info={"result": None, "source_step": source_key, "operation": operation},
        )

    values = [record[0] for record in records]
    result_value = _apply_operation(values, operation)
    if scale not in (None, ""):
        factor = _to_number(scale)
        if factor is not None:
            result_value *= factor

    if result_value is not None and isinstance(round_digits, (int, float)):
        try:
            round_int = int(round_digits)
            result_value = round(result_value, round_int)
        except Exception:
            pass

    hits: List[CommonHit] = []
    for value, row in records:
        node_id = row.get("node_id")
        if not isinstance(node_id, str):
            continue
        provenance = {
            "tool": "compute.eval",
            "source_step": source_key,
            "operation": operation,
        }
        cell_id = row.get("cell_id")
        if isinstance(cell_id, str):
            provenance["cell_id"] = cell_id
        hits.append(
            CommonHit(
                node_id=node_id,
                evidence_type=row.get("evidence_type") or "table",
                score=1.0,
                provenance=provenance,
                modality=row.get("modality") or "table",
                affordances=["numeric"],
                meta={
                    "value": value,
                    "unit": row.get("unit"),
                    "label": row.get("label"),
                    "year": row.get("year"),
                    "field": field,
                },
            )
        )

    metrics = {
        "n_values": len(values),
        "operation": operation,
        "field": field,
    }
    info = {
        "result": result_value,
        "operation": operation,
        "field": field,
        "source_step": source_key,
        "count": len(values),
        "values": values,
    }
    status = "ok" if (result_value is not None or operation == "count") else "empty"
    return ToolResult(status=status, hits=hits, metrics=metrics, info=info)


def filter(call: ToolCall) -> ToolResult:
    context = call.args.get("_context") or {}
    source_key = call.args.get("source") or call.args.get("source_step") or call.args.get("source_result")
    if not isinstance(source_key, str):
        return ToolResult(
            status="error",
            errors=[CommonError(error="INVALID_ARGS", message="source step_id required", retryable=False)],
            error="source step_id required",
        )

    source_result = context.get(source_key)
    if not isinstance(source_result, ToolResult):
        return ToolResult(
            status="error",
            errors=[CommonError(error="MISSING_CONTEXT", message=f"missing context for {source_key}", retryable=False)],
            error="missing context",
        )

    data = source_result.info if isinstance(source_result.info, dict) else {}
    if isinstance(source_result.data, dict):
        data = {**source_result.data, **data}
    rows: List[Dict[str, Any]] = data.get("rows") if isinstance(data.get("rows"), list) else []
    if not rows:
        return ToolResult(status="empty", hits=[], metrics={"n_rows": 0, "source": source_key})

    expected_unit = _clean_unit(call.args.get("unit") or data.get("unit"))
    expected_year = _to_int(call.args.get("year") or data.get("year"))

    comparator_raw = _clean_comparator(call.args.get("comparator"))
    operation = _clean_operation(call.args.get("operation"))

    # Rank-based selection
    if comparator_raw in {"rank"} or operation == "rank":
        rank = _to_int(call.args.get("rank"))
        if rank is None or rank <= 0:
            return ToolResult(status="error", data={}, metrics={}, error="rank required for comparator 'rank'")
        ranked_rows = _apply_rank(rows, rank, expected_unit=expected_unit, expected_year=expected_year)
        status = "ok" if ranked_rows else "empty"
        metrics = {
            "n_rows": len(rows),
            "n_match": len(ranked_rows),
            "rank": rank,
        }
        hits = [
            hit
            for hit in (
                _row_to_hit(match, source_key, comparator="rank", extra_meta={"rank": rank})
                for match in ranked_rows
            )
            if hit is not None
        ]
        info = {
            "matches": ranked_rows,
            "comparator": "rank",
            "rank": rank,
            "unit": expected_unit,
            "year": expected_year,
            "source_step": source_key,
        }
        return ToolResult(status=status, hits=hits, metrics=metrics, info=info)

    comparator = comparator_raw or ">"
    threshold = _to_number(call.args.get("threshold"))
    if threshold is None:
        return ToolResult(status="error", data={}, metrics={}, error="threshold required for compute.filter")

    comparator_fn = _COMPARATORS.get(comparator)
    if comparator_fn is None:
        return ToolResult(status="error", data={}, metrics={}, error=f"unsupported comparator {comparator}")

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
    hits = [
        hit
        for hit in (
            _row_to_hit(match, source_key, comparator=comparator, extra_meta={"threshold": threshold})
            for match in matches
        )
        if hit is not None
    ]

    info = {
        "matches": matches,
        "threshold": threshold,
        "comparator": comparator,
        "unit": expected_unit,
        "year": expected_year,
        "source_step": source_key,
    }
    return ToolResult(status=status, hits=hits, metrics=metrics, info=info)


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


def _clean_operation(value) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip().lower()
        return text or None
    return None


def _apply_rank(
    rows: List[Dict[str, object]],
    rank: int,
    *,
    expected_unit: Optional[str],
    expected_year: Optional[int],
) -> List[Dict[str, object]]:
    scored: List[Tuple[float, Dict[str, object]]] = []
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
        scored.append((value, dict(row)))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    for idx, (_, row) in enumerate(scored, start=1):
        row["rank"] = idx
    if rank > len(scored):
        return []
    target = scored[rank - 1][0]
    return [row for value, row in scored if value == target]


def _row_to_hit(row: Dict[str, object], source_key: str, *, comparator: str, extra_meta: Dict[str, object]) -> Optional[CommonHit]:
    node_id = row.get("node_id")
    if not isinstance(node_id, str):
        return None
    provenance = {
        "tool": "compute.filter",
        "source_step": source_key,
        "comparator": comparator,
    }
    cell_id = row.get("cell_id")
    if isinstance(cell_id, str):
        provenance["cell_id"] = cell_id
    provenance.update({k: v for k, v in extra_meta.items() if v is not None})
    meta = {
        "label": row.get("label"),
        "value": row.get("value"),
        "unit": row.get("unit"),
        "year": row.get("year"),
        "rank": row.get("rank"),
    }
    return CommonHit(
        node_id=node_id,
        evidence_type="table",
        score=1.0,
        provenance=provenance,
        modality="table",
        affordances=["numeric"],
        meta={k: v for k, v in meta.items() if v is not None},
    )


def _collect_numeric_records(result: ToolResult, *, field: str) -> List[Tuple[float, Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    seen: set[Tuple[str, float, Optional[str], Optional[str]]] = set()
    for container in (getattr(result, "info", None), result.data if hasattr(result, "data") else None):
        if not isinstance(container, dict):
            continue
        for key in ("matches", "rows", "values", "series"):
            items = container.get(key)
            if isinstance(items, Sequence):
                records.extend(_ensure_record_dict(item, default_evidence=container.get("evidence_type")) for item in items if isinstance(item, dict))
    cleaned: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        value = record.get(field)
        if field != "value" and value is None:
            value = record.get("value")
        number = _to_number(value)
        if number is None:
            continue
        dedup_key = (
            str(record.get("node_id", "")),
            float(number),
            record.get("label"),
            record.get("year"),
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        cleaned.append((number, record))
    return cleaned


def _ensure_record_dict(item: Dict[str, Any], *, default_evidence: Optional[str] = None) -> Dict[str, Any]:
    record = dict(item)
    if "evidence_type" not in record and default_evidence:
        record["evidence_type"] = default_evidence
    if "modality" not in record:
        record["modality"] = "table"
    return record


def _apply_operation(values: Sequence[float], operation: str) -> Optional[float]:
    if operation in {"sum", "total"}:
        return float(sum(values))
    if operation in {"avg", "average", "mean"}:
        return float(fmean(values)) if values else None
    if operation in {"max", "maximum"}:
        return float(max(values)) if values else None
    if operation in {"min", "minimum"}:
        return float(min(values)) if values else None
    if operation in {"count", "len"}:
        return float(len(values))
    if operation in {"range", "spread"}:
        return float(max(values) - min(values)) if values else None
    if operation in {"first"}:
        return float(values[0]) if values else None
    if operation in {"last"}:
        return float(values[-1]) if values else None
    return float(sum(values))


__all__ = ["eval", "filter"]
