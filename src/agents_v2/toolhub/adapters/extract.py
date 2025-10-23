"""Extraction adapters."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from ...retriever.manager import RetrieverManager, RetrieverResources
from ...schemas import StrategyStep
from ..types import CommonError, CommonHit, ToolCall, ToolResult


def extract(call: ToolCall) -> ToolResult:
    return ToolResult(
        status="error",
        errors=[CommonError(error="NOT_IMPLEMENTED", message="extract.* not implemented", retryable=False)],
        error="NOT_IMPLEMENTED",
    )


def regex(call: ToolCall) -> ToolResult:
    context = call.args.get("_context") or {}
    resources = _resolve_resources(call)
    if resources is None:
        raise RuntimeError("extract.regex requires RetrieverResources")
    pattern_raw = call.args.get("pattern")
    if not isinstance(pattern_raw, str) or not pattern_raw.strip():
        return ToolResult(
            status="error",
            errors=[CommonError(error="INVALID_ARGS", message="pattern is required", retryable=False)],
            error="pattern is required",
        )
    try:
        pattern = re.compile(pattern_raw, flags=re.IGNORECASE)
    except re.error as exc:
        return ToolResult(
            status="error",
            errors=[CommonError(error="INVALID_PATTERN", message=str(exc), retryable=False)],
            error=str(exc),
        )

    source_key = call.args.get("source") or call.args.get("from") or call.args.get("source_step")
    hits_source = []
    if isinstance(source_key, str):
        source_result = context.get(source_key)
        if isinstance(source_result, ToolResult):
            hits_source = _collect_hits(source_result)

    nodes: List[str] = []
    for hit in hits_source:
        node_id = _hit_node_id(hit)
        if node_id:
            nodes.append(node_id)
    if not nodes:
        extra_nodes = call.args.get("nodes")
        if isinstance(extra_nodes, str):
            nodes.append(extra_nodes)
        elif isinstance(extra_nodes, (list, tuple, set)):
            nodes.extend(str(n) for n in extra_nodes if isinstance(n, str))

    if not nodes:
        return ToolResult(status="empty", hits=[], metrics={"reason": "no-nodes"})

    matches_hits: List[CommonHit] = []
    matches_info: List[Dict[str, Any]] = []
    for node_id in nodes:
        text = _get_node_text(resources, node_id)
        if not text:
            continue
        for match in pattern.finditer(text):
            start, end = match.span()
            snippet = text[max(0, start - 40): min(len(text), end + 40)]
            matches_info.append({
                "node_id": node_id,
                "match": match.group(0),
                "span": [start, end],
                "snippet": snippet,
            })
            matches_hits.append(
                CommonHit(
                    node_id=node_id,
                    evidence_type="text",
                    score=1.0,
                    provenance={
                        "tool": "extract.regex",
                        "pattern": pattern_raw,
                        "span": [start, end],
                    },
                    modality="text",
                    affordances=["regex"],
                    meta={"match": match.group(0)},
                )
            )

    status = "ok" if matches_hits else "empty"
    metrics = {"n_matches": len(matches_hits), "pattern": pattern_raw}
    return ToolResult(status=status, hits=matches_hits, metrics=metrics, info={"matches": matches_info})


def chart_read_axis(call: ToolCall) -> ToolResult:
    context = call.args.get("_context") or {}
    resources = _resolve_resources_with_step(call)

    source_key = call.args.get("source") or call.args.get("from")
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

    source_hits = _collect_hits(source_result)
    if not source_hits:
        return ToolResult(status="empty", hits=[], metrics={"n_series": 0, "source": source_key})

    series: List[Dict[str, object]] = []
    for raw_hit in source_hits:
        node_id = _hit_node_id(raw_hit)
        if not node_id:
            continue
        text = _get_table_text(manager, node_id) or ""
        if not text:
            continue
        parsed = _parse_chart_text(node_id=node_id, text=text)
        series.extend(parsed)

    metrics = {"n_series": len(series), "source": source_key}
    status = "ok" if series else "empty"
    common_hits: List[CommonHit] = []
    for item in series:
        node_id = item.get("node_id")
        if isinstance(node_id, str):
            meta_payload = {
                "label": item.get("label"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "series": item.get("series"),
            }
            common_hits.append(
                CommonHit(
                    node_id=node_id,
                    evidence_type="graphics",
                    score=1.0,
                    provenance={
                        "tool": "extract.chart_read_axis",
                        "source_step": source_key,
                    },
                    modality="chart",
                    affordances=["axis_read"],
                    meta={k: v for k, v in meta_payload.items() if v is not None},
                )
            )
    return ToolResult(
        status=status,
        hits=common_hits,
        metrics=metrics,
        info={"series": series, "source_step": source_key},
    )


def column(call: ToolCall) -> ToolResult:
    context = call.args.get("_context") or {}
    resources = _resolve_resources_with_step(call)
    step = _require_step(call.args.get("_step"))

    source_key = call.args.get("source") or call.args.get("from")
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

    source_hits = _collect_hits(source_result)
    if not source_hits:
        return ToolResult(status="empty", hits=[], metrics={"n_rows": 0, "source": source_key})

    value_hints = _to_str_list(call.args.get("value_hints"))
    label_hints_base = _to_str_list(call.args.get("label_hints"))
    unit_hint = _clean_str(call.args.get("unit"))
    years = [int(y) for y in call.args.get("years") or [] if _is_int_like(y)]

    rows = []
    for raw_hit in source_hits:
        label_hints = list(label_hints_base)
        allowed_rows: Optional[List[int]] = None
        metadata = _hit_metadata(raw_hit)
        meta_rows = metadata.get("rows")
        if isinstance(meta_rows, list):
            allowed_rows = []
            for item in meta_rows:
                if not isinstance(item, dict):
                    continue
                idx = item.get("row_index")
                if isinstance(idx, int):
                    allowed_rows.append(idx)
                label = item.get("row_label")
                if isinstance(label, str) and label.strip():
                    low = label.strip()
                    if low not in label_hints:
                        label_hints.append(low)
            if not allowed_rows:
                allowed_rows = None

        node_id = _hit_node_id(raw_hit)
        if not node_id:
            continue
        structured = resources.tables.get(node_id)
        if structured:
            parsed_struct = _rows_from_structured(
                node_id=node_id,
                table=structured,
                value_hints=value_hints,
                label_hints=label_hints,
                unit_hint=unit_hint,
                years=years,
                allowed_rows=allowed_rows,
            )
            rows.extend(parsed_struct)
            if parsed_struct:
                continue
        table_text = _get_table_text(resources, node_id)
        if not table_text:
            continue
        parsed = _parse_table_text(
            node_id=node_id,
            text=table_text,
            value_hints=value_hints,
            label_hints=label_hints,
            unit_hint=unit_hint,
            years=years,
        )
        rows.extend(parsed)

    metrics = {"n_rows": len(rows), "n_hits": len(source_hits)}
    status = "ok" if rows else "empty"
    common_hits: List[CommonHit] = []
    for row in rows:
        node_id = row.get("node_id")
        if not isinstance(node_id, str):
            continue
        cell_id = row.get("cell_id")
        meta_payload = {
            "label": row.get("label"),
            "value": row.get("value"),
            "unit": row.get("unit"),
            "year": row.get("year"),
            "text": row.get("text"),
            "column": row.get("column"),
        }
        provenance = {
            "tool": "extract.column",
            "source_step": source_key,
            "table_id": node_id,
        }
        if isinstance(cell_id, str):
            provenance["cell_id"] = cell_id
        common_hits.append(
            CommonHit(
                node_id=node_id,
                evidence_type="table",
                score=1.0,
                provenance=provenance,
                modality="table",
                affordances=["cell"],
                meta={k: v for k, v in meta_payload.items() if v is not None},
            )
        )

    info = {
        "rows": rows,
        "unit": unit_hint,
        "years": years,
        "source_step": source_key,
    }
    return ToolResult(status=status, hits=common_hits, metrics=metrics, info=info)


def _require_step(step: Optional[StrategyStep]) -> StrategyStep:
    if not isinstance(step, StrategyStep):
        raise RuntimeError("extract.column requires StrategyStep context")
    return step


def _resolve_resources(call: ToolCall) -> Optional[RetrieverResources]:
    resources = call.args.get("_resources")
    if isinstance(resources, RetrieverResources):
        return resources
    manager = call.args.get("_retriever_manager")
    if isinstance(manager, RetrieverManager):
        return manager.resources
    return None


def _resolve_resources_with_step(call: ToolCall) -> RetrieverResources:
    step = _require_step(call.args.get("_step"))
    resources = _resolve_resources(call)
    if resources is None:
        raise RuntimeError(f"{step.tool} requires RetrieverResources")
    return resources


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


def _get_table_text(resources: RetrieverResources, node_id: str) -> Optional[str]:
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


def _get_node_text(resources: RetrieverResources, node_id: str) -> Optional[str]:
    text = _get_table_text(resources, node_id)
    if text:
        return text
    resources = manager.resources
    dense_views = resources.dense_views or {}
    for corpus in dense_views.values():
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


def _collect_hits(result: ToolResult) -> List[object]:
    hits = list(getattr(result, "hits", []) or [])
    if hits:
        return hits
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        legacy_hits = data.get("hits")
        if isinstance(legacy_hits, list):
            return legacy_hits
    return []


def _hit_node_id(hit: object) -> Optional[str]:
    if hasattr(hit, "node_id"):
        node_id = getattr(hit, "node_id")
        if isinstance(node_id, str):
            return node_id
    if isinstance(hit, dict):
        node_id = hit.get("node_id")
        if isinstance(node_id, str):
            return node_id
    return None


def _hit_metadata(hit: object) -> Dict[str, Any]:
    if hasattr(hit, "meta"):
        meta = getattr(hit, "meta")
        if isinstance(meta, dict):
            return dict(meta)
    if hasattr(hit, "metadata"):
        meta = getattr(hit, "metadata")
        if isinstance(meta, dict):
            return dict(meta)
    if isinstance(hit, dict):
        meta = hit.get("meta") or hit.get("metadata")
        if isinstance(meta, dict):
            return dict(meta)
    return {}


def _rows_from_structured(
    *,
    node_id: str,
    table: Dict[str, object],
    value_hints: List[str],
    label_hints: List[str],
    unit_hint: Optional[str],
    years: List[int],
    allowed_rows: Optional[Sequence[int]] = None,
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
    allowed_set = set(allowed_rows) if allowed_rows is not None else None

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
    for row_idx, row in enumerate(rows_data):
        if not isinstance(row, (list, tuple)):
            continue
        if label_idx >= len(row):
            continue
        if allowed_set is not None and row_idx not in allowed_set:
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
                    "cell_id": f"{node_id}:r{row_idx+1}c{idx+1}",
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
