"""Extraction adapters."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from ...retriever.manager import RetrieverManager
from ...schemas import RetrievalHit, StrategyStep
from ..types import ToolCall, ToolResult


def extract(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        data={},
        metrics={},
        error="NOT_IMPLEMENTED",
    )


def regex(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        data={},
        metrics={},
        error="NOT_IMPLEMENTED",
    )


def chart_read_axis(call: ToolCall) -> ToolResult:
    context = call.args.get("_context") or {}
    manager = _require_manager(call.args.get("_retriever_manager"))

    source_key = call.args.get("source")
    if not isinstance(source_key, str):
        return ToolResult(status="error", data={}, metrics={}, error="source step_id required")
    source_result = context.get(source_key)
    if not isinstance(source_result, ToolResult):
        return ToolResult(status="error", data={}, metrics={}, error=f"missing context for {source_key}")

    hits: Sequence[RetrievalHit] = source_result.data.get("hits") if isinstance(source_result.data, dict) else []
    if not hits:
        return ToolResult(status="empty", data={"series": []}, metrics={"n_series": 0})

    series: List[Dict[str, object]] = []
    for hit in hits:
        text = _get_table_text(manager, hit.node_id) or ""
        if not text:
            continue
        parsed = _parse_chart_text(node_id=hit.node_id, text=text)
        series.extend(parsed)

    metrics = {"n_series": len(series), "source": source_key}
    status = "ok" if series else "empty"
    hits: List[RetrievalHit] = []
    for item in series:
        node_id = item.get("node_id")
        if isinstance(node_id, str):
            hits.append(
                RetrievalHit(
                    node_id=node_id,
                    score=1.0,
                    tool="extract.chart_read_axis",
                    metadata={"label": item.get("label"), "value": item.get("value"), "unit": item.get("unit")},
                )
            )
    return ToolResult(
        status=status,
        data={"series": series, "source_step": source_key, "hits": hits},
        metrics=metrics,
    )


def column(call: ToolCall) -> ToolResult:
    context = call.args.get("_context") or {}
    manager = _require_manager(call.args.get("_retriever_manager"))
    step = _require_step(call.args.get("_step"))

    source_key = call.args.get("source")
    if not isinstance(source_key, str):
        return ToolResult(status="error", data={}, metrics={}, error="source step_id required")
    source_result = context.get(source_key)
    if not isinstance(source_result, ToolResult):
        return ToolResult(status="error", data={}, metrics={}, error=f"missing context for {source_key}")

    hits: Sequence[RetrievalHit] = source_result.data.get("hits") if isinstance(source_result.data, dict) else []
    if not hits:
        return ToolResult(status="empty", data={"rows": []}, metrics={"n_rows": 0, "source": source_key})

    value_hints = _to_str_list(call.args.get("value_hints"))
    label_hints = _to_str_list(call.args.get("label_hints"))
    unit_hint = _clean_str(call.args.get("unit"))
    years = [int(y) for y in call.args.get("years") or [] if _is_int_like(y)]

    rows = []
    for hit in hits:
        structured = manager.resources.tables.get(hit.node_id)
        if structured:
            parsed_struct = _rows_from_structured(
                node_id=hit.node_id,
                table=structured,
                value_hints=value_hints,
                label_hints=label_hints,
                unit_hint=unit_hint,
                years=years,
            )
            rows.extend(parsed_struct)
            if parsed_struct:
                continue
        table_text = _get_table_text(manager, hit.node_id)
        if not table_text:
            continue
        parsed = _parse_table_text(
            node_id=hit.node_id,
            text=table_text,
            value_hints=value_hints,
            label_hints=label_hints,
            unit_hint=unit_hint,
            years=years,
        )
        rows.extend(parsed)

    metrics = {"n_rows": len(rows), "n_hits": len(hits)}
    status = "ok" if rows else "empty"
    hit_list: List[RetrievalHit] = []
    for row in rows:
        node_id = row.get("node_id")
        if isinstance(node_id, str):
            hit_list.append(
                RetrievalHit(
                    node_id=node_id,
                    score=1.0,
                    tool="extract.column",
                    metadata={
                        "label": row.get("label"),
                        "value": row.get("value"),
                        "unit": row.get("unit"),
                        "year": row.get("year"),
                    },
                )
            )

    data = {
        "rows": rows,
        "unit": unit_hint,
        "years": years,
        "source_step": source_key,
        "hits": hit_list,
    }
    return ToolResult(status=status, data=data, metrics=metrics)


def _require_manager(manager: Optional[RetrieverManager]) -> RetrieverManager:
    if not isinstance(manager, RetrieverManager):
        raise RuntimeError("extract.column requires RetrieverManager instance")
    return manager


def _require_step(step: Optional[StrategyStep]) -> StrategyStep:
    if not isinstance(step, StrategyStep):
        raise RuntimeError("extract.column requires StrategyStep context")
    return step


def _clean_str(value) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _to_str_list(value) -> List[str]:
    items: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    items.append(cleaned)
    return items


def _is_int_like(value) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False


def _get_table_text(manager: RetrieverManager, node_id: str) -> Optional[str]:
    resources = manager.resources
    if node_id in resources.text_index:
        return resources.text_index[node_id]

    for variant in ("table", "table_dense", "table#dense"):
        corpus = resources.dense_views.get(variant)
        if corpus and node_id in corpus:
            return corpus[node_id]

    # fallback: look for entries whose base id matches
    for corpus in resources.dense_views.values():
        if node_id in corpus:
            return corpus[node_id]
    return None


def _parse_table_text(
    *,
    node_id: str,
    text: str,
    value_hints: List[str],
    label_hints: List[str],
    unit_hint: Optional[str],
    years: List[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not text:
        return rows

    value_hints_lower = [vh.lower() for vh in value_hints]
    label_hints_lower = [lh.lower() for lh in label_hints]
    year_strings = {str(y) for y in years}

    segments = re.split(r"[\n;\u2022]+", text)
    for segment in segments:
        line = segment.strip()
        if not line:
            continue
        lower = line.lower()
        if value_hints_lower and not any(h in lower for h in value_hints_lower):
            # allow fallback if it looks like a percentage/value
            if "%" not in lower and re.search(r"\d", lower) is None:
                continue

        match = re.search(r"(-?\d[\d,]*(?:\.\d+)?)\s*(%|percent|pct)?", line)
        if not match:
            continue
        number_raw = match.group(1)
        unit_token = match.group(2)
        try:
            value = float(number_raw.replace(",", ""))
        except ValueError:
            continue

        unit = unit_hint
        if unit is None and unit_token:
            unit = "%"
        if unit is None and "%" in line:
            unit = "%"

        label = line[: match.start()].strip(" :•-")
        if not label:
            label = line.strip()
        if label_hints_lower and not any(h in label.lower() for h in label_hints_lower):
            # keep if number hint matched strongly even without label hint
            if not value_hints_lower:
                continue

        year = None
        for y in year_strings:
            if y in line:
                year = int(y)
                break

        rows.append(
            {
                "node_id": node_id,
                "label": label,
                "value": value,
                "unit": unit,
                "year": year,
                "text": line,
            }
        )
    return rows


def _parse_chart_text(*, node_id: str, text: str) -> List[Dict[str, object]]:
    segments = re.split(r"[\n;\u2022]+", text)
    series: List[Dict[str, object]] = []
    for segment in segments:
        line = segment.strip()
        if not line:
            continue
        match = re.search(r"([A-Za-z0-9\s\-]+):?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(%|percent|pct)?", line)
        if not match:
            continue
        label = match.group(1).strip()
        number_raw = match.group(2)
        unit_token = match.group(3)
        try:
            value = float(number_raw.replace(",", ""))
        except ValueError:
            continue
        unit = "%"
        if not unit_token and "%" not in line:
            unit = None
        series.append(
            {
                "node_id": node_id,
                "label": label,
                "value": value,
                "unit": unit,
                "text": line,
            }
        )
    return series


def _rows_from_structured(
    *,
    node_id: str,
    table: Dict[str, object],
    value_hints: List[str],
    label_hints: List[str],
    unit_hint: Optional[str],
    years: List[int],
) -> List[Dict[str, object]]:
    columns = table.get("columns") or []
    rows_data = table.get("rows") or []
    if not isinstance(columns, list) or not isinstance(rows_data, list):
        return []
    if not columns or not rows_data:
        return []

    value_hints_lower = [vh.lower() for vh in value_hints]
    label_hints_lower = [lh.lower() for lh in label_hints]
    numeric_pattern = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

    label_idx = 0
    value_idxs: List[int] = []
    for idx, column in enumerate(columns):
        if idx == label_idx:
            continue
        col_lower = str(column).lower()
        if value_hints_lower and any(h in col_lower for h in value_hints_lower):
            value_idxs.append(idx)
        elif "%" in col_lower or "percent" in col_lower:
            value_idxs.append(idx)
    if not value_idxs:
        # fallback: any column that appears numeric in data
        for idx in range(1, len(columns)):
            for row in rows_data:
                if idx < len(row) and numeric_pattern.search(str(row[idx])):
                    value_idxs.append(idx)
                    break
    if not value_idxs:
        return []

    year_strings = {str(y) for y in years}
    visual_hint = table.get("caption") or table.get("preview")
    extracted: List[Dict[str, object]] = []
    for row in rows_data:
        if not isinstance(row, (list, tuple)):
            continue
        if label_idx >= len(row):
            continue
        label = str(row[label_idx]).strip()
        if label_hints_lower and label and not any(h in label.lower() for h in label_hints_lower):
            continue

        for idx in value_idxs:
            if idx >= len(row):
                continue
            cell = str(row[idx]).strip()
            value = _to_number(cell)
            if value is None:
                continue
            unit = unit_hint or ("%"
                                 if "%" in cell or any("%" in str(r[idx]) for r in rows_data if isinstance(r, (list, tuple)) and idx < len(r))
                                 else None)
            year = None
            for y in year_strings:
                if y in cell or y in label:
                    year = int(y)
                    break
            extracted.append(
                {
                    "node_id": node_id,
                    "label": label,
                    "value": value,
                    "unit": unit,
                    "year": year,
                    "text": cell,
                    "column": columns[idx],
                    "visual": visual_hint,
                }
            )
    return extracted


def _to_number(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


__all__ = ["extract", "regex", "chart_read_axis", "column"]
