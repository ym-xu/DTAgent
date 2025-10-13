from __future__ import annotations

from typing import List, Optional
from collections import defaultdict

from llama_index.core import VectorStoreIndex, StorageContext, QueryBundle
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor


class PageCollapsePostprocessor(BaseNodePostprocessor):
    """Group NodeWithScore by page and keep top-K pages.

    Chooses the representative chunk per page as the highest scoring one,
    and assigns the aggregated page score according to `agg`.
    """

    def __init__(self, k_pages: int = 5, agg: str = "max") -> None:
        self.k_pages = int(k_pages)
        self.agg = str(agg)

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        buckets: dict = defaultdict(list)
        for nws in nodes:
            md = nws.node.metadata or {}
            pg = md.get("page_idx")
            if pg is None and "page_range" in md:
                rng = md.get("page_range")
                if isinstance(rng, list) and rng:
                    pg = rng[0]
            buckets[pg].append(nws)

        scored: List[NodeWithScore] = []
        for pg, arr in buckets.items():
            scores = [(x.get_score() or 0.0) for x in arr]
            if not scores:
                continue
            if self.agg == "mean":
                s = sum(scores) / max(1, len(scores))
            elif self.agg == "sum":
                s = sum(scores)
            else:
                s = max(scores)
            top = max(arr, key=lambda x: x.get_score() or 0.0)
            top.score = s
            top.node.metadata["page_idx_collapsed"] = pg
            scored.append(top)

        scored.sort(key=lambda x: x.get_score() or 0.0, reverse=True)
        return scored[: self.k_pages]


def build_index_and_query_engine(
    leaf_nodes: List[TextNode],
    storage_context: StorageContext,
    similarity_top_k: int = 12,
    auto_merge_ratio: float = 0.35,
    page_view_cfg: Optional[dict] = None,
):
    index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, show_progress=True)
    base_retriever = index.as_retriever(similarity_top_k=int(similarity_top_k))

    am_retriever = AutoMergingRetriever(
        vector_retriever=base_retriever,
        storage_context=storage_context,
        simple_ratio_thresh=float(auto_merge_ratio),
        verbose=False,
    )

    postprocessors = []
    if page_view_cfg and page_view_cfg.get("enabled"):
        postprocessors.append(
            PageCollapsePostprocessor(
                k_pages=page_view_cfg.get("k_pages", 5), agg=page_view_cfg.get("agg", "max")
            )
        )

    qe_default = RetrieverQueryEngine.from_args(retriever=am_retriever)
    qe_page = (
        RetrieverQueryEngine.from_args(retriever=am_retriever, node_postprocessors=postprocessors)
        if postprocessors
        else None
    )
    return index, qe_default, qe_page

