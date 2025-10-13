"""
Lightweight vector index for Planner hint/embedding recall.

Artifacts:
- index dict matches the requested schema and can be pickled to `index.pkl`:
  {
    "doc_id": str,
    "model": str,
    "dim": int,
    "nodes": [ {node meta...} ],
    "emb": np.ndarray shape (N, D), float32, L2-normalized,
    "id2idx": {node_id: idx}
  }

Provides convenience functions to build, save, load, and query.

Note on hints: entries' embed_text should be built from title + summary + hints.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Iterable, List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from .embed_models import Embedder


def _as_array_f32(x):
    if np is None:
        return x
    return np.asarray(x, dtype=np.float32)


def _l2_normalize(arr):
    if np is None:
        # list-of-lists fallback
        out = []
        for v in arr:
            s = sum(x * x for x in v) or 1.0
            n = s ** 0.5
            out.append([x / n for x in v])
        return out
    v = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (v / norms).astype(np.float32)


def build_index(
    *,
    doc_id: str,
    embedder: Embedder,
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build an in-memory index dict from node entries.

    entries: list of dicts, each must include:
      - node_id (str)
      - embed_text (str): title + summary + hints
      - minimal node meta fields for the output: role, level?, page_idx?, kind?
    """
    texts = [e.get("embed_text") or "" for e in entries]
    X = embedder.encode(texts)

    # Normalize (most embedders already normalize, but double-sure)
    X = _l2_normalize(X)

    id2idx: Dict[str, int] = {}
    nodes_meta: List[Dict[str, Any]] = []
    for i, e in enumerate(entries):
        nid = str(e.get("node_id"))
        id2idx[nid] = i
        meta = {
            "node_id": nid,
        }
        # Copy selected fields if present
        for f in (
            "level",
            "role",
            "page_idx",
            "kind",
            "parent_section",
            "parent_title",
            "bbox",
            "orig_node_id",
            "chunk_idx",
            # optional descriptive fields for terminal preview/debugging
            "title",
            "summary",
            "hints",
            "raw_text",
            "label",
            "figure_no",
            "table_no",
        ):
            if e.get(f) is not None:
                meta[f] = e.get(f)
        nodes_meta.append(meta)

    # Wrap into index dict
    idx: Dict[str, Any] = {
        "doc_id": doc_id,
        "model": embedder.name,
        "dim": int(embedder.dim),
        "nodes": nodes_meta,
        "emb": _as_array_f32(X),
        "id2idx": id2idx,
    }
    return idx


def save_index_pkl(index_obj: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "wb") as f:
        pickle.dump(index_obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def search(
    index_obj: Dict[str, Any],
    embedder: Embedder,
    query: str,
    topk: int = 8,
) -> List[Tuple[str, float]]:
    """Simple inner-product search over normalized vectors.

    Returns: list of (node_id, score) sorted by score desc
    """
    if np is None:
        # Fallback: cosine using list-of-lists
        qv = embedder.encode([query])[0]
        scores: List[Tuple[str, float]] = []
        for meta, vec in zip(index_obj["nodes"], index_obj["emb"]):
            s = sum(a * b for a, b in zip(qv, vec))
            scores.append((meta["node_id"], float(s)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

    q = embedder.encode([query])  # (1, D)
    E = _as_array_f32(index_obj["emb"])  # (N, D)
    qv = q if hasattr(q, "shape") else _as_array_f32(q)
    # (N,) inner products
    sims = (E @ qv.T).reshape(-1)
    order = np.argsort(sims)[::-1][:topk]
    out: List[Tuple[str, float]] = []
    for i in order.tolist():
        out.append((index_obj["nodes"][i]["node_id"], float(sims[i])))
    return out
