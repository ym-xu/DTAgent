# -*- coding: utf-8 -*-
"""
planner_llm_strict.py — LLM-Only Planner (Planlet-based)

依赖注入：
- RetrievalPort: 你现有的 FAISS/BM25/Graph/IDMaps 胶水（见下方 Protocol）
- llm_call: 你现有的 LLM 客户端（例如 utils.llm_clients.gpt_llm_call / qwen_llm_call）

核心能力（四个 LLM Hook，任何失败抛 PlannerError）：
- H1: intent_v2()    → 解析任务/对象/roles_seed(prob)/filters/aggregation/selector/columns/scope/need_leaf
- H2: rewrite()      → 生成 {"dense": "...", "sparse_terms": ["...", ...]}
- H3: planlet()      → 产出 1-3 步可执行步骤（op/node/args/expect/success_if/produce/on_fail/budget）
- H4: on_fail()      → 基于 gaps 生成修复步骤（参数修/对象切换/范围放宽）

Planner 不执行任何工具，只产出 ObservationPlan（JSON 可序列化结构）。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json, re, math, time

# =========================
# 数据结构 / 契约
# =========================

from .types import Candidate, RetrievalPort

@dataclass
class PlanStep:
    op: str                        # 仅允许枚举（见 ALLOWED_OPS）
    node: str
    args: Dict[str, Any] = field(default_factory=dict)
    expect: str = ""
    success_if: str = ""           # Observer 判定成功的信号名（如 schema_parsed/col_resolved/...）
    produce: List[str] = field(default_factory=list)
    on_fail: Dict[str, Any] = field(default_factory=dict)   # {"next":"RETRY_FIND_COL" ...}
    budget: int = 120

@dataclass
class ObservationPlan:
    question: str
    assumptions: Dict[str, Any]              # {"intent": {...}, "query": {...}, ...}
    candidates: List[Candidate]
    steps: List[PlanStep]
    budgets: Dict[str, int]                  # {"tokens":1600,"steps":8}
    visited: List[str] = field(default_factory=list)

class PlannerError(RuntimeError):
    pass

# =========================
# 检索 / 图接口（按此 Protocol 适配你的实现）
# =========================

"""RetrievalPort imported from .types"""

# =========================
# 常量 / JSON 约束
# =========================

ALLOWED_OPS = {
    "OPEN_TABLE_SCHEMA", "FIND_COL", "FIND_ROW", "READ_CELL",
    "OPEN_CAPTION", "SCAN_PARAS",
    "FIND_NODES", "COUNT"     # 新增：用于统计/枚举
}

DEFAULT_BUDGETS = {"tokens": 1600, "steps": 8}
REPLAN_BUDGETS  = {"tokens":  800, "steps": 4}

# =========================
# RRF 融合 / 图扩圈 / 重排
# =========================

def _ranks(cs: List[Candidate]) -> Dict[str, int]:
    return {c.node_id: i+1 for i,c in enumerate(cs)}

def fuse_rrf(dense: List[Candidate], sparse: List[Candidate], k: int = 60) -> List[Candidate]:
    rd, rs = _ranks(dense), _ranks(sparse)
    pool: Dict[str, Candidate] = {}
    for c in dense + sparse:
        pool.setdefault(c.node_id, c)
    fused: List[Candidate] = []
    for nid, c in pool.items():
        r1, r2 = rd.get(nid, 10**6), rs.get(nid, 10**6)
        score = 1.0/(k+r1) + 1.0/(k+r2)
        why = list(c.why)
        if nid in rd: why.append(f"dense@{r1}")
        if nid in rs: why.append(f"bm25@{r2}")
        fused.append(Candidate(nid, c.role, score, why, c.filters, c.afford))
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused

def graph_expand(cands: List[Candidate], R: RetrievalPort, seed:int=10, topK:int=80) -> List[Candidate]:
    seen = {c.node_id for c in cands}
    out  = list(cands)
    for c in cands[:seed]:
        # section → child/same_page
        if c.role == "section":
            for t,d,e in R.graph_neighbors(c.node_id, types=("child","same_page")):
                if d not in seen:
                    role = "image" if d.startswith("img_") else ("table" if d.startswith("tab_") else "section")
                    out.append(Candidate(d, role, c.score*0.6, c.why+["expand:"+t], e, c.afford))
                    seen.add(d)
        # image/table → parent/ref/same_page
        if c.role in ("image","table"):
            for t,d,e in R.graph_neighbors(c.node_id, types=("parent","ref","same_page")):
                if d not in seen:
                    out.append(Candidate(d, "section", c.score*0.5, c.why+["expand:"+t], e, c.afford))
                    seen.add(d)
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:topK]

def rerank(query: str, cands: List[Candidate], topK:int=40) -> List[Candidate]:
    ql = query.lower()
    rescored=[]
    for c in cands:
        s, why = c.score, list(c.why)
        f, a = c.filters or {}, c.afford or {}
        if f.get("label") and str(f["label"]).lower() in ql: s += 0.6; why.append("label_hit")
        if any((u in ql) for u in (f.get("units_set") or [])): s += 0.2; why.append("unit_hit")
        if a.get("supports_COMPARE") and any(w in ql for w in ["compare","bigger","higher","rank","max","min",">","<"]):
            s += 0.2; why.append("compare_prior")
        # 角色先验（软）
        if "figure" in ql and c.role=="image": s += 0.2
        if "table"  in ql and c.role=="table": s += 0.2
        # 同页/同节先验（用 filters.parent_section/page_idx 时可加入 page 距离）
        rescored.append(Candidate(c.node_id, c.role, s, why, f, a))
    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored[:topK]

# =========================
# LLM JSON 工具（你也可注入自定义 llm_json_fn）
# =========================

def _loose_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        st = s.find("{")
        if st < 0: return None
        depth = 0
        for i,ch in enumerate(s[st:], st):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    frag = s[st:i+1]
                    return json.loads(frag)
    except Exception:
        return None
    return None

def default_llm_json(messages: List[Dict[str,str]], llm_call, max_tokens:int=256) -> dict:
    """
    llm_call(messages, json_mode=True, max_tokens=...) → str
    """
    raw = llm_call(messages=messages, json_mode=True, max_tokens=max_tokens)
    if not raw: return {}
    obj = _loose_json(raw)
    return obj or {}

# =========================
# JSON 校验
# =========================

def _validate_roles_seed(x) -> List[dict]:
    if not isinstance(x, list) or not x:
        raise PlannerError("roles_seed must be non-empty list")
    for item in x:
        if not isinstance(item, dict) or "set" not in item or not item["set"]:
            raise PlannerError("roles_seed item must have non-empty 'set'")
        for r in item["set"]:
            if r not in {"section","image","table"}:
                raise PlannerError(f"invalid role '{r}' in roles_seed")
        if "p" in item and not isinstance(item["p"], (int, float)):
            raise PlannerError("roles_seed.p must be number")
    return x

def validate_intent_v2(obj: dict) -> dict:
    need = {"task","targets","roles_seed","filters","aggregation","selector","columns","scope","need_leaf"}
    if not (isinstance(obj, dict) and need.issubset(obj.keys())):
        raise PlannerError("LLM intent missing required keys")
    # 基本校验（宽）
    _validate_roles_seed(obj["roles_seed"])
    if not isinstance(obj.get("targets"), list): raise PlannerError("targets must be list")
    if obj.get("selector") not in {"max","min","rank","compare","trend","none"}:
        raise PlannerError("selector invalid")
    if obj.get("scope") not in {"document","same_section","same_page","refs","children"}:
        raise PlannerError("scope invalid")
    if not isinstance(obj.get("need_leaf"), bool): raise PlannerError("need_leaf must be bool")
    # 任务枚举放宽一些
    return obj

def validate_steps(steps: List[dict]) -> List[PlanStep]:
    if not isinstance(steps, list) or not steps:
        raise PlannerError("LLM returned empty steps")
    out: List[PlanStep] = []
    for s in steps:
        op = s.get("op","")
        node = s.get("node","")
        if op not in ALLOWED_OPS: raise PlannerError(f"invalid op '{op}'")
        if not node: raise PlannerError(f"step missing node: {s}")
        out.append(PlanStep(
            op=op, node=node,
            args=s.get("args") or {},
            expect=s.get("expect",""),
            success_if=s.get("success_if",""),
            produce=s.get("produce") or [],
            on_fail=s.get("on_fail") or {},
            budget=int(s.get("budget") or 120),
        ))
    return out

# =========================
# Planner（LLM-Only）
# =========================

class PlannerLLMStrict:
    """
    LLM-Only Planner：
      - require LLM for intent/rewrite/planlet/on-fail（失败抛 PlannerError）
      - 不执行工具，只产出 ObservationPlan
    """
    def __init__(
        self,
        retriever: RetrievalPort,
        llm_call,                             # 你的 LLM 客户端函数：fn(messages, json_mode=True, max_tokens=...) -> str
        model_name: str = "gpt-4o-mini",
        retry: int = 1,
        topk_candidates: int = 8,
        min_pool_size: int = 10,
    ):
        self.R = retriever
        self.llm_call = llm_call
        self.model_name = model_name
        self.retry = retry
        self.topk_candidates = topk_candidates
        self.min_pool_size = min_pool_size

    # ---------- Public ----------
    def plan(self, question: str, visited: List[str] | None = None) -> ObservationPlan:
        visited = visited or []
        # H1: Intent v2（带 roles_seed）
        intent = self._call_intent_v2(question)

        # 选择 roles_seed（可能多组，按概率+召回诊断选择；召回弱自动换下一组）
        pool, seed_used, query_used = None, None, None
        for i, seed in enumerate(sorted(intent["roles_seed"], key=lambda x: x.get("p", 0), reverse=True)):
            roles = set(seed["set"])
            # H2: Rewrite
            rewrite = self._call_rewrite(question, intent, tries=(self.retry+1))
            # 检索融合 + 扩圈 + 过滤 visited + 重排
            dense  = self.R.dense(rewrite["dense"], roles, topK=200)
            sparse = self.R.sparse(" ".join(rewrite["sparse_terms"]), roles, topK=200)
            fused  = fuse_rrf(dense, sparse)
            expanded = graph_expand(fused, self.R, seed=10, topK=80)
            expanded = [c for c in expanded if c.node_id not in set(visited)]
            ranked   = rerank(rewrite["dense"], expanded, topK=40)
            # 召回诊断：太少或缺关键域时，尝试下一个 seed
            if self._recall_weak(ranked, intent, threshold_size=self.min_pool_size):
                continue
            pool, seed_used, query_used = ranked, seed, rewrite
            break

        # 如果所有 seed 都不理想，放宽为全角色重试一轮
        if pool is None:
            roles_all = {"section","image","table"}
            rewrite = self._call_rewrite(question, intent, tries=(self.retry+1))
            dense  = self.R.dense(rewrite["dense"], roles_all, topK=200)
            sparse = self.R.sparse(" ".join(rewrite["sparse_terms"]), roles_all, topK=200)
            fused  = fuse_rrf(dense, sparse)
            expanded = graph_expand(fused, self.R, seed=10, topK=80)
            expanded = [c for c in expanded if c.node_id not in set(visited)]
            pool     = rerank(rewrite["dense"], expanded, topK=40)
            seed_used, query_used = {"set": list(roles_all), "p": 0.0}, rewrite
            if self._recall_weak(pool, intent, threshold_size=self.min_pool_size):
                raise PlannerError("recall too weak for all roles; cannot plan")

        # H3: Planlet（1~3 步）
        steps = self._call_planlet(question, intent, pool[:self.topk_candidates])

        # ObservationPlan
        plan = ObservationPlan(
            question=question,
            assumptions={
                "intent": intent,
                "query":  query_used,
                "seed_used": seed_used
            },
            candidates=pool[:10],
            steps=steps,
            budgets=DEFAULT_BUDGETS,
            visited=visited
        )
        return plan

    def replan(self, question: str, evidence: Dict[str,Any], gaps: List[Dict[str,Any]], visited: List[str]) -> ObservationPlan:
        # H4: On-fail repair（严格 JSON）
        steps = self._call_on_fail(question, gaps, evidence)
        return ObservationPlan(
            question=question,
            assumptions={"replan": True, "gaps": gaps},
            candidates=[],
            steps=steps,
            budgets=REPLAN_BUDGETS,
            visited=visited
        )

    # ---------- LLM Hooks ----------
    def _call_intent_v2(self, question: str) -> dict:
        sys = (
            "Return JSON ONLY with fields:\n"
            "{"
            "\"task\":\"lookup_value|compare|rank|verify|count|list|summarize|explain\","
            "\"targets\":[\"image|table|section|caption|paragraph|document|page\"],"
            "\"roles_seed\":[{\"set\":[\"section|image|table\"],\"p\":0.5}],"
            "\"filters\":[{\"field\":\"\",\"op\":\"=|contains\",\"value\":\"\"}],"
            "\"aggregation\":{\"op\":\"count|unique|exists|sum|avg\",\"dedup_by\":\"figure_no|table_no|label\"},"
            "\"selector\":\"max|min|rank|compare|trend|none\","
            "\"columns\":[{\"name\":\"\",\"aliases\":[]}],"
            "\"scope\":\"document|same_section|same_page|refs|children\","
            "\"need_leaf\":true|false"
            "}"
            "\nNo explanations."
        )
        msg=[{"role":"system","content":sys},{"role":"user","content":question}]
        for _ in range(self.retry+1):
            obj = default_llm_json(msg, self.llm_call, max_tokens=260)
            try:
                return validate_intent_v2(obj)
            except Exception:
                continue
        raise PlannerError("LLM intent_v2 failed")

    def _call_rewrite(self, question: str, intent: dict, tries:int=1) -> dict:
        sys = (
            "Rewrite the question for retrieval. Return JSON ONLY:\n"
            "{\"dense\":\"<short semantic query>\", \"sparse_terms\":[\"kw1\",\"alias\",\"%\",\"Figure 2\"]}\n"
            "No explanations."
        )
        user = json.dumps({"question":question, "intent":intent}, ensure_ascii=False)
        msg=[{"role":"system","content":sys},{"role":"user","content":user}]
        for _ in range(tries):
            obj = default_llm_json(msg, self.llm_call, max_tokens=220)
            dense = (obj.get("dense") or question).strip()
            sparse_terms = obj.get("sparse_terms") or [question]
            if not isinstance(sparse_terms, list): sparse_terms = [str(sparse_terms)]
            return {"dense": dense, "sparse_terms": [t for t in sparse_terms if t]}
        raise PlannerError("LLM rewrite failed")

    def _call_planlet(self, question: str, intent: dict, pool: List[Candidate]) -> List[PlanStep]:
        ops = list(ALLOWED_OPS)
        ctx = {
            "question": question,
            "intent": intent,
            "allowed_ops": ops,
            "candidates": [
                {"node_id":c.node_id, "role":c.role, "filters":c.filters, "afford":c.afford, "why":c.why}
                for c in pool
            ]
        }
        sys = (
          "Plan 1-3 executable actions (planlet) for a retrieval agent on a document graph.\n"
          "Allowed ops: [\"OPEN_TABLE_SCHEMA\",\"FIND_COL\",\"FIND_ROW\",\"READ_CELL\",\"OPEN_CAPTION\",\"SCAN_PARAS\",\"FIND_NODES\",\"COUNT\"].\n"
          "You MUST choose 'node' from candidates[].node_id. Do NOT invent ids. 'node' MUST be a non-empty string.\n"
          "Return JSON ONLY in this exact shape:\n"
          "{\"steps\":[{\"op\":\"\",\"node\":\"\",\"args\":{},\"expect\":\"\",\"success_if\":\"\",\"produce\":[],\"on_fail\":{\"next\":\"\",\"scope\":\"\"},\"budget\":0}]}\n"
          "No explanations."
        )
        msg=[{"role":"system","content":sys},{"role":"user","content":json.dumps(ctx, ensure_ascii=False)}]
        for _ in range(self.retry+1):
            obj = default_llm_json(msg, self.llm_call, max_tokens=320)
            steps = validate_steps(obj.get("steps") or [])
            return steps
        raise PlannerError("LLM planlet failed")

    def _call_on_fail(self, question: str, gaps: List[Dict[str,Any]], evidence: Dict[str,Any]) -> List[PlanStep]:
        sys = (
          "Given 'gaps' after execution, propose 1-2 repair actions using allowed ops:\n"
          "[\"OPEN_TABLE_SCHEMA\",\"FIND_COL\",\"FIND_ROW\",\"READ_CELL\",\"OPEN_CAPTION\",\"SCAN_PARAS\",\"FIND_NODES\",\"COUNT\"].\n"
          "Return JSON ONLY with the same 'steps' schema as planlet."
        )
        ctx={"question":question, "gaps":gaps, "evidence_summary": list((evidence or {}).keys())[:8]}
        msg=[{"role":"system","content":sys},{"role":"user","content":json.dumps(ctx, ensure_ascii=False)}]
        for _ in range(self.retry+1):
            obj = default_llm_json(msg, self.llm_call, max_tokens=240)
            steps = validate_steps(obj.get("steps") or [])
            return steps
        raise PlannerError("LLM on_fail failed")

    # ---------- 召回诊断 / roles_seed 稳健 ----------
    def _recall_weak(self, ranked: List[Candidate], intent: dict, threshold_size:int=10) -> bool:
        if len(ranked) < threshold_size:
            return True
        # 若问图/表且关键域命中很少，也视为偏弱（简化启发式）
        want_img = "image" in (intent.get("targets") or [])
        want_tab = "table" in (intent.get("targets") or [])
        if want_img:
            hit = sum(1 for c in ranked[:20] if c.role=="image")
            if hit <= 2: return True
        if want_tab:
            hit = sum(1 for c in ranked[:20] if c.role=="table")
            if hit <= 2: return True
        return False
