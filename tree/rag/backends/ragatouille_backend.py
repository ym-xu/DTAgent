from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import RetrieverBackend, Response, Hit
import os
from ..ingestion.build_nodes import build_nodes_from_tree


class RAGatouilleBackend(RetrieverBackend):
    """RAGatouille (ColBERT) backend.

    Notes:
    - Expects `ragatouille` to be installed.
    - Disables automatic split_documents; uses our leaf chunks as documents.
    - Persists under cfg["index"]["persist_dir"]/ragatouille/<index_name>
    """

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}
        self._index_name: Optional[str] = None
        self._model = None

    def _ensure_model(self, model_name: str):
        try:
            from RAGatouille import RAGPretrainedModel  # type: ignore
        except Exception as e:
            raise RuntimeError("ragatouille is not installed. `pip install ragatouille`. Error: %s" % e)
        if self._model is None:
            self._model = RAGPretrainedModel.from_pretrained(model_name)

    def build(self, doc: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        model_name = cfg.get("ragatouille", {}).get("model", "colbert-ir/colbertv2.0")
        index_name = cfg.get("ragatouille", {}).get("index_name") or (doc.get("doc_id") or "index")
        self._ensure_model(model_name)

        # Ingest leaf chunks
        _, leaf_nodes = build_nodes_from_tree(doc, cfg)
        docs: List[Dict[str, Any]] = []
        for lf in leaf_nodes:
            text = lf.get_content(metadata_mode="none")
            md = lf.metadata or {}
            docs.append({
                "id": md.get("source_node_ids", [lf.node_id])[0] if hasattr(lf, "node_id") else None,
                "text": text,
                "metadata": md,
            })

        # Build index; disable split_documents
        persist_dir = cfg["index"]["persist_dir"]
        out_dir = persist_dir.rstrip("/") + "/ragatouille"
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        self._model.index(
            documents=docs,
            index_name=index_name,
            split_documents=False,
            overwrite=True,
            save_dir=out_dir,
        )
        self._index_name = index_name

    def query(self, q: str, *, page_view: bool = False, top_k: Optional[int] = None) -> Response:
        if self._model is None or not self._index_name:
            raise RuntimeError("Backend not built. Call build() first.")
        k = int(top_k or self._cfg.get("retrieval", {}).get("similarity_top_k", 12))
        # Search
        results = self._model.search(index_name=self._index_name, query=q, k=k)
        # RAGatouille returns a list; attempt to parse into unified hits
        hits: List[Hit] = []
        for i, r in enumerate(results or [], 1):
            # heuristic extraction
            text = r.get("content") or r.get("text") or str(r)
            score = float(r.get("score", 0.0)) if isinstance(r, dict) else 0.0
            md = r.get("document_metadata") or r.get("metadata") or {}
            hits.append(Hit(rank=i, score=score, text=text, metadata=md))
        return Response(query=q, answer=None, hits=hits, backend="ragatouille")

    def persist(self) -> None:
        # RAGatouille persists during index(); nothing needed here
        return None
