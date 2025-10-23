"""MMR-style evidence packing adapter."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from ..types import CommonHit, ToolCall, ToolResult

DEFAULT_LIMIT = 40
DEFAULT_PER_PAGE = 2
DEFAULT_LAMBDA = 0.72


def mmr_knapsack(call: ToolCall) -> ToolResult:
    context = call.args.get("_context") or {}
    sources = _normalize_sources(call.args.get("source") or call.args.get("sources") or call.args.get("from"))
    limit = _to_int(call.args.get("limit") or call.args.get("k") or call.args.get("max_hits"), DEFAULT_LIMIT)
    per_page_limit = _to_int(call.args.get("per_page_limit"), DEFAULT_PER_PAGE)
    mmr_lambda = _to_float(call.args.get("mmr_lambda"), DEFAULT_LAMBDA)
    coverage_target = _to_float(
        call.args.get("coverage_target") or call.args.get("target") or call.args.get("k_nodes"),
        float(limit),
    )
    ctx_tokens = _to_int(call.args.get("ctx_tokens"), 1500)

    raw_hits: List[CommonHit] = []
    for key in sources:
        result = context.get(key)
        if isinstance(result, ToolResult):
            raw_hits.extend(result.hits)
    extra_hits = call.args.get("hits")
    if isinstance(extra_hits, Iterable):
        for item in extra_hits:
            if isinstance(item, CommonHit):
                raw_hits.append(item)

    if not raw_hits:
        return ToolResult(
            status="empty",
            hits=[],
            metrics={
                "n_candidates": 0,
                "selected": 0,
                "limit": limit,
                "per_page_limit": per_page_limit,
                "ctx_tokens": ctx_tokens,
            },
            info={
                "sources": sources,
                "coverage": 0.0,
            },
        )

    deduped = _deduplicate_hits(raw_hits)
    selected = _mmr_select(deduped, limit=limit, per_page_limit=per_page_limit, mmr_lambda=mmr_lambda)
    coverage = 0.0
    if coverage_target > 0:
        coverage = min(1.0, len(selected) / coverage_target)

    status = "ok" if selected else "empty"
    return ToolResult(
        status=status,
        hits=selected,
        metrics={
            "n_candidates": len(deduped),
            "selected": len(selected),
            "limit": limit,
            "per_page_limit": per_page_limit,
            "mmr_lambda": mmr_lambda,
            "ctx_tokens": ctx_tokens,
        },
        info={
            "coverage": coverage,
            "sources": sources,
            "budget_hits": limit,
            "ctx_tokens": ctx_tokens,
        },
    )


def _normalize_sources(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _deduplicate_hits(hits: Iterable[CommonHit]) -> List[CommonHit]:
    best_by_node: Dict[str, Tuple[float, CommonHit]] = {}
    for hit in hits:
        node_id = getattr(hit, "node_id", None)
        if not isinstance(node_id, str) or not node_id:
            continue
        score = _score(hit)
        stored = best_by_node.get(node_id)
        if stored is None or score > stored[0]:
            best_by_node[node_id] = (score, hit)
    deduped = [item[1] for item in best_by_node.values()]
    deduped.sort(key=_score, reverse=True)
    return deduped


def _mmr_select(
    hits: Sequence[CommonHit],
    *,
    limit: int,
    per_page_limit: int,
    mmr_lambda: float,
) -> List[CommonHit]:
    if limit <= 0:
        return []
    remaining = list(hits)
    selected: List[CommonHit] = []
    page_counts: Dict[int, int] = defaultdict(int)

    while remaining and len(selected) < limit:
        best_hit = None
        best_score = float("-inf")

        for candidate in remaining:
            page_idx = _page_idx(candidate)
            if per_page_limit > 0 and page_idx is not None and page_counts[page_idx] >= per_page_limit:
                continue
            diversity_penalty = 0.0
            for chosen in selected:
                if chosen.node_id == candidate.node_id:
                    diversity_penalty = 1.0
                    break
                chosen_page = _page_idx(chosen)
                if page_idx is not None and chosen_page == page_idx:
                    diversity_penalty = max(diversity_penalty, 0.35)
            score = mmr_lambda * _score(candidate) - (1.0 - mmr_lambda) * diversity_penalty
            if score > best_score:
                best_score = score
                best_hit = candidate

        if best_hit is None:
            break

        selected.append(best_hit)
        page = _page_idx(best_hit)
        if page is not None:
            page_counts[page] += 1
        remaining.remove(best_hit)

    return selected


def _page_idx(hit: CommonHit) -> int | None:
    provenance = getattr(hit, "provenance", {}) or {}
    meta = getattr(hit, "meta", {}) or {}
    candidates = [
        provenance.get("page_idx"),
        meta.get("page_idx"),
        meta.get("page"),
    ]
    for value in candidates:
        num = _to_int(value)
        if num is not None:
            return num
    return None


def _score(hit: CommonHit) -> float:
    try:
        return float(getattr(hit, "score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _to_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["mmr_knapsack"]
