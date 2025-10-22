"""Table index adapter.

复用 RetrieverManager 的 dense/sparse 检索，为表格查询提供统一入口。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ...retriever.manager import RetrieverManager
from ...schemas import RetrievalHit, StrategyStep
from ..types import ToolCall, ToolResult


def search(call: ToolCall) -> ToolResult:
    manager = _require_manager(call.args.get("_retriever_manager"))
    _require_step(call.args.get("_step"))

    resources = manager.resources
    tables = getattr(resources, "tables", {}) or {}

    keys_raw = call.args.get("keys")
    keys = keys_raw if isinstance(keys_raw, dict) else {}
    filters = call.args.get("filters") if isinstance(call.args.get("filters"), dict) else {}

    column_terms = _normalize_terms(keys.get("columns"))
    row_terms = _normalize_terms(keys.get("rows"))
    keyword_terms = _normalize_terms(call.args.get("keywords"))
    query_term = _clean_str(call.args.get("query"))
    if query_term:
        keyword_terms.append(query_term.lower())

    if not row_terms:
        inferred = [term for term in column_terms if term and len(term.split()) <= 3]
        row_terms = _unique_extend([], inferred)
    keyword_terms = _unique_extend(keyword_terms, column_terms)
    keyword_terms = _unique_extend(keyword_terms, row_terms)

    units: List[str] = []
    unit = _clean_str(filters.get("unit"))
    if unit:
        units.append(unit.lower())
    years = _normalize_years(filters.get("years"))

    annotated_hits: List[RetrievalHit] = []

    for table_id, table in tables.items():
        columns = _ensure_list(table.get("columns"))
        caption = _clean_str(table.get("caption")) or ""
        rows = table.get("rows")
        if not isinstance(rows, Sequence):
            continue

        matched_rows: List[Dict[str, object]] = []
        best_row_score = 0.0

        for row_idx, row in enumerate(rows):
            cells = _ensure_list(row)
            row_label = cells[0] if cells else ""
            row_score = _score_row(
                row_label=row_label,
                cells=cells,
                columns=columns,
                column_terms=column_terms,
                row_terms=row_terms,
                keyword_terms=keyword_terms,
                units=units,
                years=years,
            )
            if row_score <= 0:
                continue

            best_row_score = max(best_row_score, row_score)
            matched_rows.append(
                {
                    "row_index": row_idx,
                    "row_label": row_label or None,
                    "score": row_score,
                    "cells": [
                        {
                            "column": columns[i] if i < len(columns) else f"col{i+1}",
                            "value": cells[i],
                        }
                        for i in range(len(cells))
                    ],
                }
            )

        if not matched_rows and not row_terms:
            # If explicit filters exist, allow table-level match via caption/columns
            table_level_score = _score_table_level(
                caption=caption,
                columns=columns,
                column_terms=column_terms,
                keyword_terms=keyword_terms,
                units=units,
                years=years,
            )
            if table_level_score > 0:
                best_row_score = max(best_row_score, table_level_score)
                matched_rows.append(
                    {
                        "row_index": None,
                        "row_label": None,
                        "score": table_level_score,
                        "cells": [],
                    }
                )

        if not matched_rows:
            continue

        score = min(1.0, best_row_score / 5.0)
        metadata = {
            "table_id": table_id,
            "caption": caption,
            "columns": columns,
            "rows": matched_rows,
            "filters": filters,
            "keys": keys,
        }
        annotated_hits.append(
            RetrievalHit(
                node_id=table_id,
                score=score,
                tool="table_index.search",
                metadata=metadata,
            )
        )

    status = "ok" if annotated_hits else "empty"
    metrics = {
        "n_hits": len(annotated_hits),
        "n_tables": len(tables),
    }
    return ToolResult(status=status, data={"hits": annotated_hits, "filters": filters, "keys": keys}, metrics=metrics)


def _require_manager(manager: Optional[RetrieverManager]) -> RetrieverManager:
    if not isinstance(manager, RetrieverManager):
        raise RuntimeError("table_index.search requires RetrieverManager instance")
    return manager


def _require_step(step: Optional[StrategyStep]) -> StrategyStep:
    if not isinstance(step, StrategyStep):
        raise RuntimeError("table_index.search requires StrategyStep context")
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


def _normalize_terms(value) -> List[str]:
    terms = _to_str_list(value)
    normalized: List[str] = []
    for term in terms:
        low = term.lower()
        if low not in normalized:
            normalized.append(low)
    return normalized


def _normalize_years(value) -> List[str]:
    years: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (int, float)):
                years.append(str(int(item)))
            elif isinstance(item, str):
                cleaned = item.strip()
                if cleaned.isdigit():
                    years.append(cleaned)
    return years


def _ensure_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v) if v is not None else "" for v in value]
    return []


def _unique_extend(dst: List[str], items: Sequence[str]) -> List[str]:
    seen = set(dst)
    for item in items:
        if item and item not in seen:
            dst.append(item)
            seen.add(item)
    return dst


def _score_row(
    *,
    row_label: str,
    cells: List[str],
    columns: List[str],
    column_terms: List[str],
    row_terms: List[str],
    keyword_terms: List[str],
    units: List[str],
    years: List[str],
) -> float:
    row_label_low = (row_label or "").lower()
    cell_lows = [str(cell or "").lower() for cell in cells]
    column_lows = [str(col or "").lower() for col in columns]

    score = 0.0
    row_term_hit = False

    for term in row_terms:
        if term and term in row_label_low:
            score += 3.0
            row_term_hit = True
        elif term and any(term in cell for cell in cell_lows):
            score += 2.0
            row_term_hit = True

    for term in column_terms:
        if term and any(term in col for col in column_lows):
            score += 2.0

    for term in keyword_terms:
        if term and (term in row_label_low or any(term in cell for cell in cell_lows)):
            score += 1.5
        elif term and any(term in col for col in column_lows):
            score += 1.0

    if units:
        if any(unit in cell for unit in units for cell in cell_lows) or any(unit in col for unit in units for col in column_lows):
            score += 1.0
        else:
            return 0.0

    if years:
        if any(year in cell for year in years for cell in cell_lows) or any(year in col for year in years for col in column_lows):
            score += 1.0
        else:
            return 0.0

    if row_terms and not row_term_hit:
        return 0.0

    return score


def _score_table_level(
    *,
    caption: str,
    columns: List[str],
    column_terms: List[str],
    keyword_terms: List[str],
    units: List[str],
    years: List[str],
) -> float:
    caption_low = caption.lower()
    column_lows = [str(col or "").lower() for col in columns]

    score = 0.0

    for term in column_terms:
        if term and any(term in col for col in column_lows):
            score += 2.0
    for term in keyword_terms:
        if term and (term in caption_low or any(term in col for col in column_lows)):
            score += 1.5
    if units and any(unit in col for unit in units for col in column_lows):
        score += 1.0
    if years and any(year in col for year in years for col in column_lows):
        score += 1.0
    return score


__all__ = ["search"]
