"""
RetrieverManager
================

提供简化版检索工具，实现标签跳转、页码定位、以及基于文本内容的
稠密/稀疏/混合检索。主要用于单元测试与模块集成验证。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from ..memory import AgentMemory
from ..schemas import RetrievalHit, StrategyStep

logger = logging.getLogger(__name__)


class RetrieverLLMCallable(Protocol):
    def __call__(
        self,
        *,
        tool: str,
        query: str,
        candidates: List[Dict[str, object]],
        config: "RetrieverLLMConfig",
    ) -> str: ...


@dataclass
class RetrieverLLMConfig:
    backend: str = "gpt"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_candidates: int = 20
    max_candidate_chars: int = 320


def _default_llm_callable(
    *,
    tool: str,
    query: str,
    candidates: List[Dict[str, object]],
    config: RetrieverLLMConfig,
) -> str:
    from src.utils.llm_clients import gpt_llm_call, qwen_llm_call  # type: ignore

    system_prompt = (
        "You rank DocTree nodes for a QA agent. Use only the supplied candidates. "
        "Produce a JSON object with keys 'thinking' (string, <= 25 words) and 'results' (array sorted by relevance). "
        "Each array item must be {node_id: string, base_id?: string, score: float between 0 and 1}. "
        "Use base_id exactly as provided when present. If nothing is relevant, return {\"thinking\": \"no match\", \"results\": []}.\n"
        "\n"
        "Guidelines:\n"
        "- Consider the question intent and the candidate metadata (view, text, fields).\n"
        "- Reward nodes whose text or fields contain the entities, years, or metrics implied by the question.\n"
        "- Penalize generic nodes that do not mention key concepts.\n"
        "- Keep scores within [0,1]; use higher scores for the best matches and lower (but non-zero) scores for weaker matches.\n"
        "\n"
        "Example (fictional dossier \"Renewable Cities Atlas 2042\"):\n"
        "Question: \"Which chart compares solar rooftops across districts?\"\n"
        "Candidates:\n"
        "  - node_id: img_12 (view=image, text: \"Bar chart of rooftop solar adoption by district\")\n"
        "  - node_id: sec_30 (view=section#gist, text: \"Policy incentives for wind zones\")\n"
        "Expected JSON:\n"
        "{\n"
        "  \"thinking\": \"Image node mentions solar rooftops; section is tangential.\",\n"
        "  \"results\": [\n"
        "    {\"node_id\": \"img_12\", \"base_id\": \"img_12\", \"score\": 0.92},\n"
        "    {\"node_id\": \"sec_30\", \"base_id\": \"sec_30\", \"score\": 0.25}\n"
        "  ]\n"
        "}\n"
    )
    trimmed_candidates = [
        {
            **cand,
            "text": (cand.get("text") or "")[: config.max_candidate_chars] if cand.get("text") else None,
        }
        for cand in candidates
    ]
    payload = json.dumps(
        {
            "tool": tool,
            "question": query,
            "candidates": trimmed_candidates[: config.max_candidates],
        },
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]
    if config.backend == "qwen":
        return qwen_llm_call(messages, model=config.model, json_mode=True)
    return gpt_llm_call(messages, model=config.model, json_mode=True)


def _normalize_variant(view_name: str) -> str | None:
    if not view_name:
        return None
    if "#" in view_name:
        return view_name.split("#", 1)[1] or None
    return view_name


@dataclass
class RetrieverResources:
    """
    检索器依赖的数据索引。

    - label_index:   图表编号到 node_id
    - page_index:    页码到 node_id 列表
    - text_index:    node_id 到文本内容（用于 LLM 选择）
    - dense_views:   variant -> {node_id_variant: text}
    - dense_base_ids: variant -> {node_id_variant: base_node_id}
    - sparse_docs:   node_id -> 字段字典
    - tables:        node_id -> 结构化表格信息
    - node_roles:    node_id -> role（table/image/text …）
    - image_paths:   image node_id -> 本地路径
    - image_meta:    image node_id -> 描述/摘要等
    - node_pages:    node_id -> 逻辑页码（1-based，若存在）
    - node_physical_pages: node_id -> 物理页码（1-based，若存在）
    - base_dir:      文档索引所在目录
    """

    label_index: Dict[str, str] = field(default_factory=dict)
    page_index: Dict[int, List[str]] = field(default_factory=dict)
    text_index: Dict[str, str] = field(default_factory=dict)
    dense_views: Dict[str, Dict[str, str]] = field(default_factory=dict)
    dense_base_ids: Dict[str, Dict[str, str]] = field(default_factory=dict)
    sparse_docs: Dict[str, Dict[str, str]] = field(default_factory=dict)
    tables: Dict[str, Dict[str, object]] = field(default_factory=dict)
    node_roles: Dict[str, str] = field(default_factory=dict)
    image_paths: Dict[str, str] = field(default_factory=dict)
    image_meta: Dict[str, Dict[str, object]] = field(default_factory=dict)
    node_pages: Dict[str, int] = field(default_factory=dict)
    node_physical_pages: Dict[str, int] = field(default_factory=dict)
    figure_spans: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)
    figure_tokens: Dict[str, List[str]] = field(default_factory=dict)
    base_dir: Optional[Path] = None
    toc_outline: List[str] = field(default_factory=list)
    heading_index: Dict[str, List[str]] = field(default_factory=dict)
    heading_titles: Dict[str, str] = field(default_factory=dict)
    heading_children: Dict[str, List[str]] = field(default_factory=dict)


def build_stub_resources(
    *,
    label_pairs: Sequence[Tuple[str, str]],
    page_pairs: Sequence[Tuple[int, str]],
    text_pairs: Sequence[Tuple[str, str]],
    dense_views: Dict[str, Dict[str, str]] | None = None,
    sparse_docs: Dict[str, Dict[str, str]] | None = None,
    tables: Dict[str, Dict[str, object]] | None = None,
    node_roles: Dict[str, str] | None = None,
    image_paths: Dict[str, str] | None = None,
    image_meta: Dict[str, Dict[str, object]] | None = None,
    node_pages: Dict[str, int] | None = None,
    node_physical_pages: Dict[str, int] | None = None,
    figure_spans: Dict[str, List[Dict[str, object]]] | None = None,
    figure_tokens: Dict[str, List[str]] | None = None,
    base_dir: Optional[Path] = None,
    toc_outline: Optional[List[str]] = None,
    heading_index: Optional[Dict[str, List[str]]] = None,
    heading_titles: Optional[Dict[str, str]] = None,
    heading_children: Optional[Dict[str, List[str]]] = None,
) -> RetrieverResources:
    """便捷构建器，供测试使用。"""
    label_index = {label: nid for label, nid in label_pairs}
    page_index: Dict[int, List[str]] = {}
    for page, nid in page_pairs:
        page_index.setdefault(page, []).append(nid)
    text_index = {nid: text for nid, text in text_pairs}
    dense_views = dense_views or {}
    dense_base_ids = {
        view: {nid: nid.split("#")[0] for nid in corpus.keys()}
        for view, corpus in dense_views.items()
    }
    if figure_tokens is None and figure_spans:
        figure_tokens = {
            node_id: _tokens_from_spans(spans)
            for node_id, spans in figure_spans.items()
        }

    return RetrieverResources(
        label_index=label_index,
        page_index=page_index,
        text_index=text_index,
        dense_views=dense_views,
        dense_base_ids=dense_base_ids,
        sparse_docs=sparse_docs or {},
        tables=tables or {},
        node_roles=node_roles or {},
        image_paths=image_paths or {},
        image_meta=image_meta or {},
        node_pages=node_pages or {},
        node_physical_pages=node_physical_pages or {},
        figure_spans=figure_spans or {},
        figure_tokens=figure_tokens or {},
        base_dir=base_dir,
        toc_outline=toc_outline or [],
        heading_index=heading_index or {},
        heading_titles=heading_titles or {},
        heading_children=heading_children or {},
    )


def _tokens_from_spans(spans: List[Dict[str, object]]) -> List[str]:
    tokens: List[str] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        dense = span.get("dense_text")
        if isinstance(dense, str):
            tokens.extend(_expanded_tokens(dense))
        sparse = span.get("sparse_tokens")
        if isinstance(sparse, list):
            for item in sparse:
                if isinstance(item, str):
                    tokens.append(item.lower())
        chart_type = span.get("chart_type")
        if isinstance(chart_type, str):
            tokens.extend(_expanded_tokens(chart_type))
    return tokens


def _expanded_tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = list(words)
    tokens.extend(f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1))
    return tokens


class RetrieverManager:
    """选择合适的检索工具并执行，结果写入 Memory 缓存。"""

    def __init__(
        self,
        resources: RetrieverResources,
        *,
        llm_config: Optional[RetrieverLLMConfig] = None,
        llm_callable: Optional[RetrieverLLMCallable] = None,
    ) -> None:
        self.resources = resources
        self.llm_config = llm_config or RetrieverLLMConfig()
        self.llm_callable = llm_callable or _default_llm_callable

    def execute(self, step: StrategyStep, memory: AgentMemory) -> List[RetrievalHit]:
        tool = step.tool
        if tool == "jump_to_label":
            hits = self._jump_to_label(step.args)
        elif tool == "jump_to_page":
            hits = self._jump_to_page(step.args)
        elif tool in {"dense_search", "dense_node.search"}:
            hits = self._dense_search(step.args, tool_name=tool)
        elif tool in {"sparse_search", "bm25_node.search"}:
            hits = self._sparse_search(step.args, tool_name=tool)
        elif tool in {"hybrid_search"}:
            hits = self._hybrid(step.args, tool_name=tool)
        elif tool == "table_index.search":
            hits = self._table_index_search(step.args)
        elif tool == "chart_index.search":
            hits = self._chart_index_search(step.args)
        else:
            raise ValueError(f"Unknown retrieval tool: {tool}")

        memory.cache_hits(step.describe(), hits)
        return hits

    # --- 工具实现 ---

    def _jump_to_label(self, args: Dict[str, object]) -> List[RetrievalHit]:
        label = str(args.get("label", "")).strip()
        node_id = self.resources.label_index.get(label)
        if not node_id:
            node_id = self.resources.label_index.get(label.lower())
        if not node_id:
            return []
        return [
            RetrievalHit(node_id=node_id, score=1.0, tool="jump_to_label", metadata={"label": label})
        ]

    def _jump_to_page(self, args: Dict[str, object]) -> List[RetrievalHit]:
        try:
            page = int(args.get("page", -1))
        except (TypeError, ValueError):
            page = -1
        node_ids = self.resources.page_index.get(page, [])
        if not node_ids:
            # fallback：找到离目标页最近的键
            candidates = sorted(self.resources.page_index.keys())
            closest = None
            min_gap = None
            for key in candidates:
                if not isinstance(key, int):
                    continue
                gap = abs(key - page)
                if min_gap is None or gap < min_gap:
                    min_gap = gap
                    closest = key
            if closest is not None:
                node_ids = self.resources.page_index.get(closest, [])
        if not node_ids:
            return []
        hits: List[RetrievalHit] = []
        for idx, nid in enumerate(node_ids):
            hits.append(
                RetrievalHit(
                    node_id=nid,
                    score=max(0.5, 1.0 - 0.1 * idx),
                    tool="jump_to_page",
                    metadata={"page": page},
                )
            )
        return hits

    def _dense_search(self, args: Dict[str, object], *, tool_name: str = "dense_search") -> List[RetrievalHit]:
        view = str(args.get("view") or "")
        variant = _normalize_variant(view)
        corpus = self.resources.text_index
        base_map: Dict[str, str] = {}
        if variant and variant in self.resources.dense_views:
            corpus = self.resources.dense_views[variant]
            base_map = self.resources.dense_base_ids.get(variant, {})
        candidates = self._prepare_dense_candidates(corpus, base_map, variant)
        queries = self._collect_queries(args)
        return self._aggregate_rank(tool=tool_name, queries=queries, candidates=candidates)

    def _sparse_search(self, args: Dict[str, object], *, tool_name: str = "sparse_search") -> List[RetrievalHit]:
        if self.resources.sparse_docs:
            candidates = self._prepare_sparse_candidates(self.resources.sparse_docs)
        else:
            candidates = self._prepare_dense_candidates(self.resources.text_index, {}, None)
        queries = self._collect_queries(args)
        return self._aggregate_rank(tool=tool_name, queries=queries, candidates=candidates)

    def _hybrid(self, args: Dict[str, object], *, tool_name: str = "hybrid_search") -> List[RetrievalHit]:
        dense_variant = _normalize_variant(str(args.get("view") or ""))
        dense_corpus = self.resources.text_index
        dense_base: Dict[str, str] = {}
        if dense_variant and dense_variant in self.resources.dense_views:
            dense_corpus = self.resources.dense_views[dense_variant]
            dense_base = self.resources.dense_base_ids.get(dense_variant, {})
        dense_candidates = self._prepare_dense_candidates(dense_corpus, dense_base, dense_variant)
        sparse_candidates = (
            self._prepare_sparse_candidates(self.resources.sparse_docs)
            if self.resources.sparse_docs
            else []
        )
        combined = dense_candidates + sparse_candidates
        queries = self._collect_queries(args)
        return self._aggregate_rank(tool=tool_name, queries=queries, candidates=combined)

    def _table_index_search(self, args: Dict[str, object]) -> List[RetrievalHit]:
        query = self._clean_str_arg(args.get("query"))
        column_hints = self._clean_str_list(args.get("column_hints"))
        keywords = self._clean_str_list(args.get("keywords"))
        filters = args.get("filters") if isinstance(args.get("filters"), dict) else {}

        queries: List[str] = []
        if query:
            queries.append(query)
        for hint in column_hints:
            if hint not in queries:
                queries.append(hint)
        for kw in keywords:
            if kw not in queries:
                queries.append(kw)

        dense_args = {
            "query": queries[0] if queries else query,
            "queries": queries[1:] if len(queries) > 1 else None,
            "view": "table",
        }
        dense_args = {k: v for k, v in dense_args.items() if v}
        raw_hits = self._dense_search(dense_args, tool_name="table_index.search")

        annotated: List[RetrievalHit] = []
        for hit in raw_hits:
            metadata = dict(hit.metadata)
            if column_hints:
                metadata["column_hints"] = column_hints
            if filters:
                metadata["filters"] = filters
            annotated.append(
                RetrievalHit(
                    node_id=hit.node_id,
                    score=hit.score,
                    tool="table_index.search",
                    metadata=metadata,
                )
            )
        return annotated

    def _chart_index_search(self, args: Dict[str, object]) -> List[RetrievalHit]:
        query = self._clean_str_arg(args.get("query"))
        keywords = self._clean_str_list(args.get("keywords"))
        units = self._clean_str_list(args.get("units"))
        years = [str(item).strip() for item in args.get("years") or [] if str(item).strip()]

        queries: List[str] = []
        if query:
            queries.append(query)
        for item in keywords + units + years:
            if item not in queries:
                queries.append(item)

        dense_args = {
            "query": queries[0] if queries else query,
            "queries": queries[1:] if len(queries) > 1 else None,
            "view": "image",
        }
        dense_args = {k: v for k, v in dense_args.items() if v}
        raw_hits = self._dense_search(dense_args, tool_name="chart_index.search")

        annotated: List[RetrievalHit] = []
        for hit in raw_hits:
            metadata = dict(hit.metadata)
            if keywords:
                metadata["keywords"] = keywords
            if units:
                metadata["units"] = units
            if years:
                metadata["years"] = years
            annotated.append(
                RetrievalHit(
                    node_id=hit.node_id,
                    score=hit.score,
                    tool="chart_index.search",
                    metadata=metadata,
                )
            )
        return annotated

    def _collect_queries(self, args: Dict[str, object]) -> List[str]:
        queries: List[str] = []

        def _append(text: Optional[str]) -> None:
            if not text:
                return
            cleaned = " ".join(text.strip().split())
            if cleaned and cleaned not in queries:
                queries.append(cleaned)

        primary = args.get("query")
        if isinstance(primary, str):
            _append(primary)

        extra_queries = args.get("queries")
        if isinstance(extra_queries, list):
            for item in extra_queries:
                if isinstance(item, str):
                    _append(item)

        keywords = args.get("keywords")
        keyword_strings: List[str] = []
        if isinstance(keywords, list):
            for kw in keywords:
                if isinstance(kw, str):
                    cleaned = " ".join(kw.strip().split())
                    if cleaned:
                        keyword_strings.append(cleaned)
        if keyword_strings:
            _append(" ".join(keyword_strings))
            for kw in keyword_strings:
                _append(kw)

        return queries

    def _clean_str_arg(self, value) -> Optional[str]:
        if isinstance(value, str):
            cleaned = " ".join(value.strip().split())
            return cleaned or None
        return None

    def _clean_str_list(self, value) -> List[str]:
        items: List[str] = []
        if isinstance(value, list):
            for element in value:
                if isinstance(element, str):
                    cleaned = " ".join(element.strip().split())
                    if cleaned:
                        items.append(cleaned)
        return items

    def _aggregate_rank(
        self,
        *,
        tool: str,
        queries: List[str],
        candidates: List[Dict[str, object]],
    ) -> List[RetrievalHit]:
        combined: Dict[str, RetrievalHit] = {}
        if not queries:
            return []
        for query in queries[:5]:
            cleaned = query.strip()
            if not cleaned:
                continue
            hits = self._rank_candidates(tool=tool, query=cleaned, candidates=candidates)
            for hit in hits:
                current = combined.get(hit.node_id)
                if not current or hit.score > current.score:
                    metadata = dict(hit.metadata)
                    metadata.setdefault("query", cleaned)
                    combined[hit.node_id] = RetrievalHit(
                        node_id=hit.node_id,
                        score=hit.score,
                        tool=tool,
                        metadata=metadata,
                    )
        ordered = sorted(combined.values(), key=lambda h: h.score, reverse=True)
        return ordered

    def _prepare_dense_candidates(
        self,
        corpus: Dict[str, str],
        base_map: Dict[str, str],
        variant: Optional[str],
    ) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        for node_id, text in list(corpus.items())[: self.llm_config.max_candidates * 2]:
            candidates.append(
                {
                    "node_id": node_id,
                    "base_id": base_map.get(node_id, node_id),
                    "view": variant or "default",
                    "text": text,
                }
            )
        return candidates

    def _prepare_sparse_candidates(self, docs: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        for node_id, fields in list(docs.items())[: self.llm_config.max_candidates * 2]:
            candidates.append(
                {
                    "node_id": node_id,
                    "base_id": node_id,
                    "view": "sparse",
                    "fields": fields,
                    "text": self._collapse_fields(fields),
                }
            )
        return candidates

    @staticmethod
    def _collapse_fields(fields: Dict[str, str]) -> str:
        ordered = [f"{key}: {value}" for key, value in fields.items()]
        return "; ".join(ordered)

    def _rank_candidates(
        self,
        *,
        tool: str,
        query: str,
        candidates: List[Dict[str, object]],
    ) -> List[RetrievalHit]:
        query = query.strip()
        if not query or not candidates:
            return []
        raw = self.llm_callable(
            tool=tool,
            query=query,
            candidates=candidates,
            config=self.llm_config,
        )
        results, thinking = self._parse_rank_response(raw)
        hits: List[RetrievalHit] = []
        for item in results:
            node_id = item.get("base_id") or item.get("node_id")
            detail_id = item.get("node_id")
            score = item.get("score")
            if not isinstance(detail_id, str) or not isinstance(node_id, str):
                continue
            try:
                score_val = float(score)
            except (TypeError, ValueError):
                score_val = 0.0
            score_val = max(0.0, min(1.0, score_val))
            hits.append(
                RetrievalHit(
                    node_id=node_id,
                    score=score_val,
                    tool=tool,
                    metadata={"source_node": detail_id},
                )
            )
        logger.debug("Retriever raw response: %s", raw)
        if thinking:
            logger.debug("Retriever thinking: %s", thinking)
        return hits

    @staticmethod
    def _parse_rank_response(raw: str) -> Tuple[List[Dict[str, object]], Optional[str]]:
        thinking: Optional[str] = None
        if not raw:
            return [], thinking
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [], thinking
        if isinstance(data, dict):
            raw_thinking = data.get("thinking")
            if isinstance(raw_thinking, str):
                thinking = raw_thinking.strip() or None
            items = data.get("results") or data.get("hits") or data.get("result")
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)], thinking
            return [], thinking
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)], thinking
        return [], thinking
    

__all__ = [
    "RetrieverManager",
    "RetrieverResources",
    "build_stub_resources",
    "RetrieverLLMConfig",
    "RetrieverLLMCallable",
]
