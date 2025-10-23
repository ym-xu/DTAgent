"""Chart index adapter leveraging figure spans & image metadata."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

from collections import Counter
from math import log
from typing import Dict, Iterable, List, Optional, Sequence, Set

from ...retriever.manager import RetrieverManager, RetrieverResources
from ..common import clamp_score
from ..types import CommonHit, ToolCall, ToolResult


def search(call: ToolCall) -> ToolResult:
    resources = _resolve_resources(call)
    if resources is None:
        raise RuntimeError("chart_index.search requires RetrieverResources")

    figures = resources.figure_spans or {}
    image_meta = resources.image_meta or {}
    tokens_map = resources.figure_tokens or {}

    keys_raw = call.args.get("keys")
    keys = keys_raw if isinstance(keys_raw, dict) else {}
    filters = call.args.get("filters") if isinstance(call.args.get("filters"), dict) else {}

    keyword_terms = _normalize_terms(keys.get("keywords"))
    entity_terms = _normalize_terms(keys.get("entities"))
    metric_terms = _normalize_terms(keys.get("metric"))
    caption_terms = _normalize_terms(keys.get("caption"))
    all_terms = _unique_extend(keyword_terms, entity_terms)
    all_terms = _unique_extend(all_terms, metric_terms)
    all_terms = _unique_extend(all_terms, caption_terms)

    units = _normalize_terms(keys.get("units") or call.args.get("units"))
    years = _normalize_years(keys.get("years") or call.args.get("years"))
    chart_types = _normalize_terms(keys.get("chart_types") or call.args.get("chart_types"))
    page_hint = filters.get("page_idx") or filters.get("page")
    parent_hint = filters.get("parent_section")

    query_tokens = _gather_query_tokens(all_terms, units, years, chart_types)

    candidate_nodes = list(tokens_map.keys())
    if page_hint:
        hint_values = _normalize_page_hint(page_hint)
        filtered = [nid for nid in candidate_nodes if resources.node_pages.get(nid) in hint_values]
        if filtered:
            candidate_nodes = filtered

    if not candidate_nodes:
        return ToolResult(status="empty", hits=[], metrics={"reason": "no-candidates"}, info={})

    cache = _build_bm25_cache(resources, candidate_nodes)

    scored_nodes: List[tuple[str, float, Dict[str, object]]] = []
    for node_id in candidate_nodes:
        doc_tokens = tokens_map.get(node_id) or []
        if not doc_tokens:
            continue
        score = _bm25_score(doc_tokens, query_tokens, cache)
        if score <= 0:
            continue
        analytics = _collect_metadata(node_id, figures, image_meta, resources)
        if parent_hint and analytics.get("parent_section") not in _ensure_list(parent_hint):
            continue
        scored_nodes.append((node_id, score, analytics))

    scored_nodes.sort(key=lambda item: item[1], reverse=True)
    top_n = int(call.args.get("top_k") or 6)
    scored_nodes = scored_nodes[:top_n]

    hits: List[CommonHit] = []
    max_idf = max(cache["idf"].values()) if cache.get("idf") else 1.0
    for node_id, score, analytics in scored_nodes:
        provenance = {
            "tool": "chart_index.search",
            "page_idx": analytics.get("page_idx"),
            "chart_type": analytics.get("chart_type"),
        }
        meta_payload = {
            "caption": analytics.get("caption"),
            "description": analytics.get("description"),
            "span_roles": analytics.get("span_roles"),
            "chart_type": analytics.get("chart_type"),
            "score_bm25": score,
        }
        hits.append(
            CommonHit(
                node_id=node_id,
                evidence_type="graphics",
                score=clamp_score(score / (max_idf + 1e-9)),
                provenance={k: v for k, v in provenance.items() if v is not None},
                modality="image",
                affordances=["chart"],
                meta={k: v for k, v in meta_payload.items() if v is not None},
            )
        )

    status = "ok" if hits else "empty"
    metrics = {
        "n_hits": len(hits),
        "n_figures": len(candidate_nodes),
        "n_query_terms": len(query_tokens),
    }
    return ToolResult(
        status=status,
        hits=hits,
        metrics=metrics,
        info={
            "filters": filters,
            "keys": keys,
            "query_tokens": query_tokens,
            "view": "image",
            "evidence_type": "graphics",
            "modality": "image",
        },
    )


def _resolve_resources(call: ToolCall) -> Optional[RetrieverResources]:
    resources = call.args.get("_resources")
    if isinstance(resources, RetrieverResources):
        return resources
    manager = call.args.get("_retriever_manager")
    if isinstance(manager, RetrieverManager):
        return manager.resources
    return None


def _unique_extend(base: List[str], extra: Sequence[str]) -> List[str]:
    seen = set(base)
    for item in extra:
        if item not in seen:
            base.append(item)
            seen.add(item)
    return base


def _normalize_terms(value) -> List[str]:
    base_terms: List[str] = []
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            base_terms.append(cleaned)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    base_terms.append(cleaned)

    normalized: List[str] = []
    for term in base_terms:
        tokens = _expanded_query_tokens(term)
        for tok in tokens:
            if tok and tok not in normalized:
                normalized.append(tok)
    return normalized


def _normalize_years(value) -> List[str]:
    years: List[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, int):
                years.append(str(item))
            elif isinstance(item, str):
                cleaned = item.strip()
                if cleaned.isdigit():
                    years.append(cleaned)
    elif isinstance(value, int):
        years.append(str(value))
    elif isinstance(value, str) and value.strip().isdigit():
        years.append(value.strip())
    return years


def _clean_str(value) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _ensure_list(value) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


__all__ = ["search"]


def _expanded_query_tokens(text: str) -> List[str]:
    low = text.lower()
    words = re.findall(r"[a-z0-9]+", low)
    tokens = list(words)
    tokens.extend(f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1))
    tokens.extend(_simple_stem(word) for word in words)
    return [tok for tok in tokens if tok]


def _simple_stem(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _gather_query_tokens(keywords: List[str], units: List[str], years: List[str], chart_types: List[str]) -> List[str]:
    tokens: List[str] = []
    for collection in (keywords, units, years, chart_types):
        for term in collection:
            tokens.extend(_expanded_query_tokens(term))
    return list(dict.fromkeys(tokens))


def _normalize_page_hint(value) -> Set[int]:
    hints: Set[int] = set()
    if isinstance(value, (list, tuple, set)):
        iterable = value
    else:
        iterable = [value]
    for item in iterable:
        try:
            num = int(item)
        except (TypeError, ValueError):
            continue
        hints.add(num)
    return hints


def _build_bm25_cache(resources: RetrieverResources, nodes: List[str]) -> Dict[str, object]:
    cache = getattr(resources, "_figure_bm25_cache", None)
    tokens_map = resources.figure_tokens or {}
    subset = {nid: tokens_map.get(nid, []) for nid in nodes}
    key = (len(subset), tuple(sorted(subset.keys())))
    if cache and cache.get("key") == key:
        return cache

    doc_freq: Dict[str, int] = {}
    doc_len: Dict[str, int] = {}
    total_len = 0
    for node_id, tokens in subset.items():
        doc_len[node_id] = len(tokens)
        total_len += len(tokens)
        unique_terms = set(tokens)
        for term in unique_terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    N = len(subset)
    avgdl = total_len / N if N > 0 else 0.0
    idf = {}
    for term, df in doc_freq.items():
        idf[term] = log((N - df + 0.5) / (df + 0.5) + 1)

    cache = {
        "key": key,
        "doc_freq": doc_freq,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "idf": idf,
        "tokens": subset,
    }
    setattr(resources, "_figure_bm25_cache", cache)
    return cache


def _bm25_score(doc_tokens: List[str], query_tokens: List[str], cache: Dict[str, object]) -> float:
    if not query_tokens:
        return 0.0
    freq = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    avgdl = cache.get("avgdl") or 0.0
    if doc_len == 0 or avgdl == 0:
        return 0.0
    doc_freq = cache.get("doc_freq", {})
    idf = cache.get("idf", {})
    k1 = 1.5
    b = 0.75
    score = 0.0
    for term in query_tokens:
        f = freq.get(term)
        if not f:
            continue
        df = doc_freq.get(term)
        if not df:
            continue
        term_idf = idf.get(term, 0.0)
        denom = f + k1 * (1 - b + b * doc_len / avgdl)
        score += term_idf * (f * (k1 + 1)) / denom
    return score


def _collect_metadata(
    node_id: str,
    figures: Dict[str, List[Dict[str, object]]],
    image_meta: Dict[str, Dict[str, object]],
    resources: RetrieverResources,
) -> Dict[str, object]:
    spans = figures.get(node_id, [])
    meta = image_meta.get(node_id, {}) or {}
    caption = meta.get("caption")
    description = meta.get("description")
    span_roles: Set[str] = set()
    chart_type = meta.get("chart_type")
    for span in spans:
        if isinstance(span, dict):
            role = span.get("role")
            if isinstance(role, str):
                span_roles.add(role)
            if not chart_type:
                st = span.get("chart_type")
                if isinstance(st, str):
                    chart_type = st
            if not caption:
                dense = span.get("dense_text")
                if isinstance(dense, str) and span.get("role") == "figure_caption":
                    caption = dense
    page_idx = resources.node_pages.get(node_id) or meta.get("page_idx")
    parent_section = meta.get("parent_section")
    return {
        "caption": caption,
        "description": description,
        "span_roles": sorted(span_roles),
        "chart_type": chart_type,
        "page_idx": page_idx,
        "parent_section": parent_section,
    }
