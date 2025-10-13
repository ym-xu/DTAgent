"""
M2 Pipeline Entrypoint: Summaries + Embedding index

Reads a DocTree JSON (doctree.mm.json), produces:
- indexes/summary.json
- JSONL views: dense_{coarse,leaf}.jsonl, sparse_{coarse,leaf}.jsonl
- Vector indices: {coarse,leaf}.faiss (or .vectors.npy + .meta.json fallback)
- BM25F JSON dirs: bm25_{coarse,leaf}/

CLI:
  # 单文档
  python -m src.index.build_index \
    --in-file /path/to/dataname/<doc_id>/doctree.mm.json \
    --out-dir /path/to/dataname/<doc_id>/indexes \
    --embed-model BAAI/bge-m3 \
    --summary-model heuristic-v1 \
    --max-tokens 120

  # 批量：扫描根目录递归查找 */doctree.mm.json 并写入对应 <doc_dir>/indexes
  python -m src.index.build_index \
    --in-dir /path/to/dataname \
    --embed-model BAAI/bge-m3 \
    --include-leaves

Design:
- Heuristic summarizer by default; optional LLM summarizer via --use-llm.
- Embedding via sentence-transformers; model不可用时报错。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from .embed_models import get_embedder
# legacy semantic_index-based pickle builders are deprecated; not imported
from .node_summarizer import build_summary


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def infer_doc_id(doc_path: str, root: Any | None = None) -> str:
    if isinstance(root, dict) and isinstance(root.get("doc_id"), str):
        return root["doc_id"]
    return os.path.basename(os.path.dirname(doc_path)) or "document"


def run(
    doc_path: str,
    out_dir: str,
    embed_model: str | None,
    summary_model: str,
    max_tokens: int,
    include_leaves: bool = False,
    use_llm: bool = False,
    llm_backend: str = "auto",
    llm_model: str = "gpt-4o-mini",
) -> tuple[str, list[str]]:
    # Summaries
    ensure_dir(out_dir)
    doc_id = infer_doc_id(doc_path)
    summary_obj, _coarse_legacy, _leaf_legacy, dense_coarse, dense_leaf, sparse_coarse, sparse_leaf, graph_edges, id_maps = build_summary(
        doctree_path=doc_path,
        out_dir=out_dir,
        doc_id=doc_id,
        model=summary_model,
        max_tokens=max_tokens,
        include_leaves=include_leaves,
        use_llm=use_llm,
        llm_backend=llm_backend,
        llm_model=llm_model,
    )
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    # Write new JSONL artifacts per README.md
    def _write_jsonl(path: str, records: list[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as wf:
            for rec in records:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    created: list[str] = []
    def _mark(p: str) -> None:
        created.append(p)

    p_dense_coarse = os.path.join(out_dir, "dense_coarse.jsonl")
    _write_jsonl(p_dense_coarse, dense_coarse)
    _mark(p_dense_coarse)
    if include_leaves:
        p_dense_leaf = os.path.join(out_dir, "dense_leaf.jsonl")
        _write_jsonl(p_dense_leaf, dense_leaf)
        _mark(p_dense_leaf)
    p_sparse_coarse = os.path.join(out_dir, "sparse_coarse.jsonl")
    _write_jsonl(p_sparse_coarse, sparse_coarse)
    _mark(p_sparse_coarse)
    if include_leaves:
        p_sparse_leaf = os.path.join(out_dir, "sparse_leaf.jsonl")
        _write_jsonl(p_sparse_leaf, sparse_leaf)
        _mark(p_sparse_leaf)
    # graph_edges 输出
    p_edges = os.path.join(out_dir, "graph_edges.jsonl")
    _write_jsonl(p_edges, graph_edges)
    _mark(p_edges)
    p_idmaps = os.path.join(out_dir, "id_maps.json")
    with open(p_idmaps, "w", encoding="utf-8") as f:
        json.dump(id_maps, f, ensure_ascii=False, indent=2)
    _mark(p_idmaps)

    # Optional: build FAISS indices if available (coarse/leaf)
    def _build_faiss(name: str, records: list[dict]) -> None:
        try:
            import numpy as np  # noqa: F401
            import faiss  # type: ignore
        except Exception:
            # Fallback: store vectors as .npy + meta
            import numpy as np
            emb = get_embedder(embed_model)
            texts = [r.get("dense_text") or "" for r in records]
            ids = [r.get("node_id") or r.get("id") for r in records]
            X = emb.encode(texts)
            X = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
            np.save(os.path.join(out_dir, f"{name}.vectors.npy"), X)
            with open(os.path.join(out_dir, f"{name}.meta.json"), "w", encoding="utf-8") as f:
                json.dump({"ids": ids}, f, ensure_ascii=False, indent=2)
            return
        # Build FAISS inner-product index
        import numpy as np
        emb = get_embedder(embed_model)
        texts = [r.get("dense_text") or "" for r in records]
        ids = [r.get("node_id") or r.get("id") for r in records]
        X = emb.encode(texts)
        X = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        faiss.write_index(index, os.path.join(out_dir, f"{name}.faiss"))
        with open(os.path.join(out_dir, f"{name}.ids.json"), "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)

    _build_faiss("coarse", dense_coarse)
    if include_leaves:
        _build_faiss("leaf", dense_leaf)

    # Extra: build raw-text leaf blocks for highest recall (merge short, split long)
    def _mk_leaf_raw_blocks(records: list[dict], *,
                            min_block_chars: int = 280,
                            max_block_chars: int = 720,
                            joiner: str = " ",
                            overlap_chars: int = 80) -> list[dict]:
        # group by (parent_section, page_idx)
        groups: Dict[Tuple[Any, Any], List[dict]] = {}
        for r in records:
            f = r.get("filters") or {}
            key = (f.get("parent_section"), f.get("page_idx"))
            groups.setdefault(key, []).append(r)
        blocks: List[dict] = []
        for (sec, page), arr in groups.items():
            # stable sort: orig_node_id, chunk_idx, node_id as tiebreaker
            def _key(x: dict) -> Tuple[str, int, str]:
                oid = str(x.get("orig_node_id") or "")
                ci = x.get("chunk_idx")
                ci = int(ci) if isinstance(ci, int) or (isinstance(ci, str) and ci.isdigit()) else 0
                nid = str(x.get("node_id") or "")
                return (oid, ci, nid)
            arr.sort(key=_key)
            i = 0
            bidx = 0
            while i < len(arr):
                cur_texts: List[str] = []
                members: List[str] = []
                # seed
                raw = arr[i].get("raw_text") or ""
                t = str(raw)
                cur_len = len(t)
                cur_texts.append(t)
                members.append(str(arr[i].get("node_id")))
                i += 1
                # merge forward until min_block_chars or end
                while cur_len < min_block_chars and i < len(arr):
                    raw2 = arr[i].get("raw_text") or ""
                    t2 = str(raw2)
                    if not t2:
                        i += 1
                        continue
                    cur_texts.append(t2)
                    members.append(str(arr[i].get("node_id")))
                    cur_len += len(t2) + len(joiner)
                    i += 1
                # join current block
                joined = joiner.join([s for s in cur_texts if s])
                # split if too long
                start = 0
                while start < len(joined):
                    end = min(start + max_block_chars, len(joined))
                    chunk = joined[start:end]
                    if not chunk.strip():
                        break
                    block_id = f"leafraw:{str(sec) or 'None'}:{str(page)}:b{bidx}"
                    blocks.append({
                        "block_id": block_id,
                        "role": "leaf_block",
                        "text": chunk,
                        "members": members,
                        "filters": {**({"parent_section": sec} if sec else {}), **({"page_idx": page} if page is not None else {})},
                    })
                    bidx += 1
                    if end >= len(joined):
                        break
                    # overlap
                    start = max(0, end - overlap_chars)
        return blocks

    leaf_raw_blocks: List[dict] = []
    if include_leaves:
        try:
            leaf_raw_blocks = _mk_leaf_raw_blocks(dense_leaf)
        except Exception:
            leaf_raw_blocks = []
        if leaf_raw_blocks:
            p_leaf_raw = os.path.join(out_dir, "leaf_raw.jsonl")
            _write_jsonl(p_leaf_raw, leaf_raw_blocks)
            _mark(p_leaf_raw)

    def _build_faiss_text(name: str, records: list[dict], text_key: str, id_key: str) -> None:
        try:
            import numpy as np  # noqa: F401
            import faiss  # type: ignore
        except Exception:
            # Fallback: vectors.npy + meta.json
            import numpy as np
            emb = get_embedder(embed_model)
            texts = [str(r.get(text_key) or "") for r in records]
            ids = [str(r.get(id_key) or "") for r in records]
            X = emb.encode(texts)
            X = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
            np.save(os.path.join(out_dir, f"{name}.vectors.npy"), X)
            with open(os.path.join(out_dir, f"{name}.meta.json"), "w", encoding="utf-8") as f:
                json.dump({"ids": ids}, f, ensure_ascii=False, indent=2)
            return
        # FAISS path
        import numpy as np
        emb = get_embedder(embed_model)
        texts = [str(r.get(text_key) or "") for r in records]
        ids = [str(r.get(id_key) or "") for r in records]
        if not texts:
            return
        X = emb.encode(texts)
        X = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        faiss.write_index(index, os.path.join(out_dir, f"{name}.faiss"))
        with open(os.path.join(out_dir, f"{name}.ids.json"), "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)

    if include_leaves and leaf_raw_blocks:
        _build_faiss_text("leaf_raw", leaf_raw_blocks, text_key="text", id_key="block_id")
    # Mark FAISS or npy artifacts if present
    for name in ("coarse", "leaf", "leaf_raw"):
        faiss_fp = os.path.join(out_dir, f"{name}.faiss")
        if os.path.exists(faiss_fp):
            _mark(faiss_fp)
        else:
            vec_fp = os.path.join(out_dir, f"{name}.vectors.npy")
            ids_fp = os.path.join(out_dir, f"{name}.meta.json")
            if os.path.exists(vec_fp):
                _mark(vec_fp)
            if os.path.exists(ids_fp):
                _mark(ids_fp)

    # Optional: build BM25F-like sparse index (JSON) for demo purposes
    def _tokenize(s: str) -> list[str]:
        import re as _re
        return [t.lower() for t in _re.findall(r"[A-Za-z0-9_\-]+", s or "")]

    def _bm25f_build(name: str, records: list[dict]) -> None:
        import math as _m
        os.makedirs(os.path.join(out_dir, f"bm25_{name}"), exist_ok=True)
        docs = []
        fields = ["title", "caption", "table_schema", "labels", "aliases", "body"]
        weights = {"title": 2.0, "caption": 1.5, "table_schema": 2.5, "labels": 3.0, "aliases": 1.2, "body": 1.0}
        df: dict[str,int] = {}
        postings: dict[str, dict[str, dict[str,int]]] = {}
        for r in records:
            doc_id = r.get("id") or r.get("node_id")
            field_toks: dict[str, list[str]] = {}
            for f in fields:
                field_toks[f] = _tokenize(r.get(f) or "")
                # update df once per doc per term
                for t in set(field_toks[f]):
                    df[t] = df.get(t, 0) + 1
            docs.append({"id": doc_id, "filters": r.get("filters")})
            # postings store raw term freq per field
            for f in fields:
                for t in field_toks[f]:
                    postings.setdefault(t, {}).setdefault(doc_id, {}).setdefault(f, 0)
                    postings[t][doc_id][f] += 1
        meta = {"N": len(docs), "fields": fields, "weights": weights}
        with open(os.path.join(out_dir, f"bm25_{name}", "docs.jsonl"), "w", encoding="utf-8") as wf:
            for d in docs:
                wf.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, f"bm25_{name}", "postings.json"), "w", encoding="utf-8") as wf:
            json.dump({"df": df, "postings": postings}, wf)
        with open(os.path.join(out_dir, f"bm25_{name}", "meta.json"), "w", encoding="utf-8") as wf:
            json.dump(meta, wf, ensure_ascii=False, indent=2)

    _bm25f_build("coarse", sparse_coarse)
    if include_leaves:
        _bm25f_build("leaf", sparse_leaf)
    # Mark BM25 dirs
    for name in ("coarse", "leaf"):
        d = os.path.join(out_dir, f"bm25_{name}")
        if os.path.isdir(d):
            _mark(d)

    # Legacy .pkl indices are deprecated and no longer generated to avoid confusion.
    return summary_path, created


def main() -> None:
    ap = argparse.ArgumentParser(description="Build M2 summaries and embedding index from a DocTree")
    ap.add_argument("--in-file", dest="in_file", required=False, help="Path to doctree.mm.json")
    ap.add_argument("--in-dir", dest="in_dir", required=False, help="Scan this directory recursively for */doctree.mm.json and write to <doc_dir>/indexes")
    ap.add_argument("--out-dir", required=False, default=None, help="Output directory for indexes (default: <doc_dir>/indexes)")
    # 默认使用当前本地开源效果较好的通用模型（质量优先）
    ap.add_argument("--embed-model", default="BAAI/bge-m3", help="Sentence Transformers model name (e.g., BAAI/bge-m3)")
    ap.add_argument("--summary-model", default="heuristic-v1", help="Summarizer model tag (metadata only unless replaced)")
    ap.add_argument("--max-tokens", type=int, default=120, help="Max tokens for summary metadata (advisory)")
    ap.add_argument("--include-leaves", action="store_true", help="同时为 text/list/equation 生成摘要与向量")
    ap.add_argument("--use-llm", action="store_true", help="使用 LLM 生成摘要（否则为启发式）")
    ap.add_argument("--llm-backend", default="auto", choices=["auto", "gpt", "qwen"], help="LLM 后端选择")
    ap.add_argument("--llm-model", default="gpt-4o-mini", help="LLM 模型名（如 gpt-4o-mini 或 qwen2.5-32b-instruct）")
    args = ap.parse_args()

    if not args.in_file and not args.in_dir:
        ap.error("One of --in-file or --in-dir must be provided.")

    if args.in_dir:
        root = args.in_dir
        if not os.path.isdir(root):
            raise NotADirectoryError(root)
        # Recursively find doctree.mm.json
        tasks: list[tuple[str, str]] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            if "doctree.mm.json" in filenames:
                doc_path = os.path.join(dirpath, "doctree.mm.json")
                out_dir = os.path.join(dirpath, "indexes")
                tasks.append((doc_path, out_dir))
        if not tasks:
            print(f"No doctree.mm.json found under {root}")
            return
        print(f"Found {len(tasks)} doctrees. Building indexes...")
        for i, (doc_path, out_dir) in enumerate(tasks, 1):
            try:
                sp, created = run(
                    doc_path,
                    out_dir,
                    args.embed_model,
                    args.summary_model,
                    args.max_tokens,
                    include_leaves=args.include_leaves,
                    use_llm=args.use_llm,
                    llm_backend=args.llm_backend,
                    llm_model=args.llm_model,
                )
                print(f"[{i}/{len(tasks)}] Wrote:")
                print(f"  {sp}")
                for p in created:
                    print(f"  {p}")
            except Exception as e:
                print(f"[{i}/{len(tasks)}] Failed: {doc_path}: {e}")
        return

    # single file mode
    doc_path = args.in_file
    if not os.path.exists(doc_path):
        raise FileNotFoundError(doc_path)

    out_dir = args.out_dir or os.path.join(os.path.dirname(doc_path), "indexes")
    sp, created = run(
        doc_path,
        out_dir,
        args.embed_model,
        args.summary_model,
        args.max_tokens,
        include_leaves=args.include_leaves,
        use_llm=args.use_llm,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
    )
    print("Wrote:")
    print(f"  {sp}")
    for p in created:
        print(f"  {p}")


if __name__ == "__main__":
    main()
