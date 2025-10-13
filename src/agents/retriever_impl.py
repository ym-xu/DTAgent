# retriever_impl.py
# Minimal RetrievalPort implementation over JSONL indexes
# -------------------------------------------------------
# Requires: numpy
# Optional: your own encode_fn for dense embeddings (e.g., SentenceTransformers)

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Callable
import json, os, re, math
import numpy as np

from .types import Candidate


# --------- Utilities ----------
_WORD = re.compile(r"[A-Za-z0-9%.\-]+")

def _tok(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "") if w]

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def _cosine(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    # a: (d,), B: (n,d)
    a = a.astype(np.float32)
    B = B.astype(np.float32)
    an = np.linalg.norm(a) + 1e-8
    Bn = np.linalg.norm(B, axis=1) + 1e-8
    return (B @ a) / (Bn * an)

def _normalize_label(lbl: str) -> str:
    # normalize variants: figure 1 / fig. 1 / Table 2b / 图 1
    s = (lbl or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^fig\.", "figure", s)
    s = re.sub(r"^图", "figure", s)
    s = re.sub(r"^表", "table", s)
    s = s.title()  # "Figure 1", "Table 2B"
    return s


# --------- BM25F (lightweight, in-memory) ----------
class BM25FIndex:
    def __init__(self, docs: List[Dict[str, Any]], field_weights: Optional[Dict[str, float]] = None):
        """
        docs: list of sparse_coarse.jsonl records with fields:
            id, role, title, caption, table_schema, aliases, labels, body, filters
        """
        self.docs = docs
        self.N = len(docs)
        self.fields = ["title", "caption", "table_schema", "labels", "aliases", "body"]
        self.w = field_weights or {
            "title": 2.0, "caption": 1.5, "table_schema": 1.7,
            "labels": 2.2, "aliases": 1.8, "body": 1.0
        }
        # per-field tokenized docs and lengths
        self.tokens: Dict[str, List[List[str]]] = {f: [] for f in self.fields}
        self.len: Dict[str, List[int]] = {f: [] for f in self.fields}
        # df per field
        self.df: Dict[str, Dict[str, int]] = {f: {} for f in self.fields}
        # inverted index: field -> term -> set(doc_idx)
        self.inv: Dict[str, Dict[str, set]] = {f: {} for f in self.fields}
        self._build()

    def _build(self):
        for i, d in enumerate(self.docs):
            for f in self.fields:
                toks = _tok(d.get(f) or "")
                self.tokens[f].append(toks)
                self.len[f].append(len(toks))
                seen = set()
                for t in toks:
                    if t not in seen:
                        self.df[f][t] = self.df[f].get(t, 0) + 1
                        self.inv[f].setdefault(t, set()).add(i)
                        seen.add(t)
        self.avglen = {f: (sum(self.len[f]) / max(1, len(self.len[f]))) for f in self.fields}

    def search(self, query: str, roles: set[str], topK: int = 200,
               k1: float = 1.2, b: float = 0.75) -> List[Tuple[int, float, List[str]]]:
        q_toks = _tok(query)
        if not q_toks:
            return []
        # candidates: union of postings across fields
        cand = set()
        for f in self.fields:
            for t in q_toks:
                if t in self.inv[f]:
                    cand |= self.inv[f][t]
        # if empty, scan all (small doc counts per file)
        if not cand:
            cand = set(range(self.N))
        scores: List[Tuple[int,float,List[str]]] = []
        for i in cand:
            d = self.docs[i]
            if roles and d.get("role") not in roles:
                continue
            why: List[str] = []
            s = 0.0
            for f in self.fields:
                if self.w.get(f, 0.0) <= 0:
                    continue
                tf = {}
                for t in self.tokens[f][i]:
                    tf[t] = tf.get(t, 0) + 1
                Ld = self.len[f][i]
                avgL = self.avglen[f] or 1.0
                for t in q_toks:
                    df = self.df[f].get(t, 0)
                    if df == 0:
                        continue
                    idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                    term_tf = tf.get(t, 0)
                    if term_tf == 0:
                        continue
                    # BM25 term
                    denom = term_tf + k1 * (1 - b + b * (Ld / avgL))
                    bm25 = idf * (term_tf * (k1 + 1)) / max(1e-6, denom)
                    s += self.w[f] * bm25
                    why.append(f"bm25:{f}:{t}")
            if s > 0:
                scores.append((i, s, why))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topK]


# --------- Retriever Implementation ----------
class JsonlRetriever:
    """
    RetrievalPort implementation over JSONL indexes.
    - Dense: cosine over encode_fn(dense_text) (no faiss required)
    - Sparse: in-memory BM25F over sparse_coarse.jsonl
    - Graph: graph_edges.jsonl (out + reverse build for 'parent')
    - IdMaps: id_maps.json (label2id / figure / table)
    """
    def __init__(
        self,
        index_dir: str,
        encode_fn: Callable[[List[str]], np.ndarray],
        field_weights: Optional[Dict[str, float]] = None,
    ):
        self.index_dir = index_dir
        self.encode_fn = encode_fn

        # Load dense_coarse
        dense_path = os.path.join(index_dir, "dense_coarse.jsonl")
        self.dense_docs: List[Dict[str, Any]] = _read_jsonl(dense_path)
        self.n_dense = len(self.dense_docs)

        # Pre-encode dense_texts
        dense_texts = [d.get("dense_text") or d.get("summary") or "" for d in self.dense_docs]
        if not callable(self.encode_fn):
            raise RuntimeError("encode_fn must be a callable: List[str] -> np.ndarray[float32]")
        embs = self.encode_fn(dense_texts)
        if not isinstance(embs, np.ndarray):
            embs = np.array(embs)
        self.dense_vecs = embs.astype(np.float32)  # (N, D)

        # Map for convenience
        self._dense_by_id = {d.get("node_id"): d for d in self.dense_docs}

        # Load sparse_coarse and build BM25F
        sparse_path = os.path.join(index_dir, "sparse_coarse.jsonl")
        self.sparse_docs: List[Dict[str, Any]] = _read_jsonl(sparse_path)
        self._sparse_idx = BM25FIndex(self.sparse_docs, field_weights=field_weights)

        # Graph edges
        edges_path = os.path.join(index_dir, "graph_edges.jsonl")
        self.out_edges: Dict[str, List[Tuple[str, str, Dict[str,Any]]]] = {}
        self.in_edges: Dict[str, List[Tuple[str, str, Dict[str,Any]]]] = {}
        if os.path.exists(edges_path):
            for e in _read_jsonl(edges_path):
                src, dst, et = e.get("src"), e.get("dst"), e.get("type")
                self.out_edges.setdefault(src, []).append((et, dst, e))
                self.in_edges.setdefault(dst, []).append((et, src, e))

        # Id maps
        idmaps_path = os.path.join(index_dir, "id_maps.json")
        mp = _load_json(idmaps_path)
        self.id_label2id: Dict[str,str] = {}
        self.id_fig: Dict[str,str] = {}
        self.id_tab: Dict[str,str] = {}
        if mp:
            # 支持两种包装：直接 keys 或 {"label2id":..., "figure":..., "table":...}
            if "label2id" in mp or "figure" in mp or "table" in mp:
                self.id_label2id = { _normalize_label(k): v for k,v in (mp.get("label2id") or {}).items() }
                self.id_fig = { str(k): v for k,v in (mp.get("figure") or {}).items() }
                self.id_tab = { str(k): v for k,v in (mp.get("table") or {}).items() }
            else:
                self.id_label2id = { _normalize_label(k): v for k,v in (mp or {}).items() }

    # ---------- RetrievalPort methods ----------

    def dense(self, query: str, roles: set[str], topK: int) -> List[Candidate]:
        qv = self.encode_fn([query])
        if not isinstance(qv, np.ndarray):
            qv = np.array(qv)
        qv = qv.astype(np.float32).reshape(1, -1)[0]
        sims = _cosine(qv, self.dense_vecs)  # (N,)
        idx = np.argsort(-sims)[:max(1, topK)]
        out: List[Candidate] = []
        for i in idx:
            d = self.dense_docs[int(i)]
            if roles and d.get("role") not in roles:
                continue
            why = [f"dense@{len(out)+1}"]
            out.append(Candidate(
                node_id=d.get("node_id"),
                role=d.get("role"),
                score=float(sims[int(i)]),
                why=why,
                filters=d.get("filters") or {},
                afford=d.get("affordances") or d.get("afford") or {}
            ))
            if len(out) >= topK:
                break
        return out

    def sparse(self, query: str, roles: set[str], topK: int) -> List[Candidate]:
        scored = self._sparse_idx.search(query, roles=roles, topK=topK)
        out: List[Candidate] = []
        for i, s, why in scored:
            d = self.sparse_docs[i]
            out.append(Candidate(
                node_id=d.get("id"),
                role=d.get("role"),
                score=float(s),
                why=why[:10],  # 截断 why 片段，避免过长
                filters=d.get("filters") or {},
                afford={}      # sparse 不带 afford，Planner 可用 dense 的 afford 补
            ))
        return out

    def graph_neighbors(self, node_id: str, types: Tuple[str, ...]) -> List[Tuple[str, str, Dict[str,Any]]]:
        res = []
        for t, dst, e in self.out_edges.get(node_id, []):
            if t in types:
                res.append((t, dst, e))
        # 允许查询 parent：用 in_edges
        if "parent" in types:
            for t, src, e in self.in_edges.get(node_id, []):
                if t in ("child","parent"):  # child 的反向是 parent
                    res.append(("parent", src, e))
        return res

    def idmap_lookup(self, label: str) -> Optional[str]:
        norm = _normalize_label(label)
        if norm in self.id_label2id:
            return self.id_label2id[norm]
        # 尝试 Figure/Table 数字直达
        m = re.search(r"(figure|table)\s*([0-9]+[a-z]?)", norm.lower())
        if m:
            kind = m.group(1)
            num  = m.group(2)
            if kind == "figure":
                return self.id_fig.get(str(num))
            else:
                return self.id_tab.get(str(num))
        return None

    def page_of(self, node_id: str) -> Optional[int]:
        d = self._dense_by_id.get(node_id)
        if not d:
            # fallback from sparse
            for sd in self.sparse_docs:
                if sd.get("id") == node_id:
                    return (sd.get("filters") or {}).get("page_idx")
            return None
        return (d.get("filters") or {}).get("page_idx")

    def get_attr(self, node_id: str, key: str) -> Optional[Any]:
        d = self._dense_by_id.get(node_id)
        if d and (d.get("filters") or {}).get(key) is not None:
            return d["filters"][key]
        # try sparse
        for sd in self.sparse_docs:
            if sd.get("id") == node_id and (sd.get("filters") or {}).get(key) is not None:
                return sd["filters"][key]
        return None

    def filter_nodes(self, role: str, filters: List[Dict[str,Any]], scope_nodes: Optional[List[str]]=None) -> List[str]:
        """
        filters: list of {field, op("="/contains), value}
        scope_nodes: if provided, only search within this whitelist
        """
        res = []
        scope = set(scope_nodes) if scope_nodes else None
        for d in self.dense_docs:
            if role and d.get("role") != role:
                continue
            nid = d.get("node_id")
            if scope and nid not in scope:
                continue
            ok = True
            F = d.get("filters") or {}
            txt = {
                "label": str(F.get("label") or ""),
                "chart_type": str(F.get("chart_type") or ""),
                "units": " ".join(F.get("units_set") or []),
                "parent_section": str(F.get("parent_section") or ""),
                "parent_title": str(F.get("parent_title") or "")
            }
            for flt in (filters or []):
                field = flt.get("field"); op = flt.get("op","="); val = str(flt.get("value") or "")
                if field is None:
                    continue
                hv = str(F.get(field) or txt.get(field) or "")
                if op == "=" and hv != val:
                    ok = False; break
                if op == "contains" and (val.lower() not in hv.lower()):
                    ok = False; break
            if ok:
                res.append(nid)
        return res

    def expand_nodes(self, base: List[str], types: Tuple[str,...]) -> List[str]:
        seen = set(base or [])
        out  = list(base or [])
        for b in list(base or []):
            for t,d,e in self.graph_neighbors(b, types=types):
                if d not in seen:
                    seen.add(d); out.append(d)
        return out


# --------- Example encode_fn ----------
def build_sentence_transformers_encoder(model_name: str):
    """
    Helper to create an encode_fn using sentence-transformers, if you use it.
    Usage:
        encode_fn = build_sentence_transformers_encoder('gte-small')
        R = JsonlRetriever(index_dir, encode_fn)
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    def _encode(texts: List[str]) -> np.ndarray:
        return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    return _encode
