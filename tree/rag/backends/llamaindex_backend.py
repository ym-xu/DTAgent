from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import RetrieverBackend, Response, Hit
from ..ingestion.build_nodes import build_nodes_from_tree
from ..index.embeddings_and_index import init_embedding, init_storage_and_index, persist
from ..retrieval.retrieval import build_index_and_query_engine


class LlamaIndexBackend(RetrieverBackend):
    def __init__(self) -> None:
        self._qe_default = None
        self._qe_page = None
        self._sc = None
        self._cfg: Dict[str, Any] = {}

    def build(self, doc: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        # embeddings + storage
        dim = init_embedding(cfg["embedding"]["model"], cfg["embedding"]["dim"])
        sc = init_storage_and_index(dim, cfg["index"]["persist_dir"])

        # ingestion
        all_nodes, leaf_nodes = build_nodes_from_tree(doc, cfg)
        sc.docstore.add_documents(all_nodes)

        # index + retriever(s)
        index, qe_default, qe_page = build_index_and_query_engine(
            leaf_nodes=leaf_nodes,
            storage_context=sc,
            similarity_top_k=cfg["retrieval"]["similarity_top_k"],
            auto_merge_ratio=cfg["retrieval"]["auto_merge_ratio"],
            page_view_cfg=cfg["retrieval"].get("page_view"),
        )
        self._qe_default = qe_default
        self._qe_page = qe_page
        self._sc = sc

    def query(self, q: str, *, page_view: bool = False, top_k: Optional[int] = None) -> Response:
        qe = self._qe_page if (page_view and self._qe_page is not None) else self._qe_default
        if qe is None:
            raise RuntimeError("Backend not built. Call build() first.")
        resp = qe.query(q)
        hits: List[Hit] = []
        for i, sn in enumerate(getattr(resp, "source_nodes", []) or [], 1):
            node = sn.node
            md = node.metadata or {}
            text = node.get_content(metadata_mode="none")
            hits.append(Hit(rank=i, score=float(sn.score or 0.0), text=text, metadata=md))
        return Response(query=q, answer=getattr(resp, "response", None), hits=hits, backend="llamaindex")

    def persist(self) -> None:
        if self._sc is not None:
            persist(self._sc)

