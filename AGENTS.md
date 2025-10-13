# AGENTS.md — Multimodal DocTree Agent System

## 🌲 1. System Overview
Our goal is to perform **explainable multimodal long-document QA** over structured DocTrees.  
The pipeline consists of four major modules:

| Stage | Module | Purpose |
|-------|---------|----------|
| M0 | Adapter + Enhancer + Builder | Parse PDF → Enhanced content → Hierarchical DocTree |
| M1 | Indexer | Build node-level embeddings and summaries |
| M2 | Agent System (Planner–Observer–Reasoner) | Navigate tree, gather evidence, reason to answer |
| M3 | Evaluation | Measure retrieval & reasoning performance vs RAG baseline |

---

## 🧩 2. Core Agent Architecture
The agent operates through an **iterative navigation–observation–reasoning loop**.

### 2.1 Planner (Navigation)
- **Goal:** Select which DocTree nodes (sections/pages) to inspect next.  
- **Inputs:**  
  - User question (Q)  
  - Node summaries + embeddings  
  - Structural context (parent/child hierarchy)  
- **Outputs:**  
  - A ranked plan of candidate nodes (e.g., `HIT_SUMMARY` and `HEADINGS_TOPK`)  
  - Optional action directives: `MOVE_TO(section=X)`, `EXPAND_CHILDREN`, etc.  
- **Prompt context:** Focused summaries + structure + hints.

### 2.2 Observer (Perception)
- **Goal:** Retrieve multimodal evidence from nodes chosen by the Planner.  
- **Subagents:**  
  - `TextObserver`: extracts paragraphs/lists/equations.  
  - `ImageObserver`: calls VLM for chart/table interpretation.  
  - `TableObserver`: reads and converts tables to text.  
- **Output:** structured observations `{node_id, modality, payload}`.

### 2.3 Reasoner (Inference)
- **Goal:** Synthesize all evidence, generate or verify the final answer.  
- **Input:** question + observed contents.  
- **Output:**  
  - Final JSON: `{answer, confidence, support_nodes}`  
  - If uncertainty high → emit `REPLAN` signal.

### 2.4 Router (Optional)
- **When:** complex question requiring decomposition.  
- **Action:** produce sub-questions → recursively call planner/observer/reasoner.  
- **Example:** “List top-3 findings and explain their data sources.”

### 2.5 Memory (Cache)
- Caches node summaries, embedding hits, and previous observations.
- Prevents re-querying the same sections and saves reasoning trace for debugging.

---

## ⚙️ 3. Data Flow Summary
```text
Question → Planner → Candidate nodes → Observer → Evidence → Reasoner → Answer
                  ↑                                               ↓
                 └─────────────[REPLAN if insufficient info]──────┘
```

## 🧠 4. Context Engineering Principles
	1.	Question Rewriting: Add domain hints (e.g., “figure count”, “chart title”).
	2.	Structured Prompt Windows: Combine heading, summary, and neighbor nodes in one coherent context block.
	3.	Role Conditioning:
	•	Planner prompt uses “You are a document navigator”
	•	Reasoner prompt uses “You are an analyst reasoning over structured evidence”
	4.	Summaries-as-Index:
	•	Each node stores a summary (≈100 tokens) used in HEADINGS_TOPK.
	•	Enables cheap semantic navigation before full-text access.

## 🗂️ 5. Directory Structure
```
project_root/
│
├── data/                           # 存储中间与最终数据文件            
│   ├── dataset_n/  
|   |   ├── doc_id_1/           
│   │   │   ├── full.pdf
│   │   │   ├── layout.json
│   │   │   ├── content_list.json
│   │   │   ├── content_list.adapted.json # M1
│   │   │   ├── content_list.enhanced.json # M1
│   │   │   ├── doctree.mm.json # M1
│   │   │   └── index/          # M2
│   │   │       ├── summary.json
│   │   │       └── ...
│   │   
│   │   ├── doc_id_n/
│   │   │   ├── full.pdf
│   │   │   └── ...
│   │   └── qa/                     # QA任务文件与结果
│
├── src/
│   ├── adapter/                    # M0: PDF→content_list
│   │   ├── adapter_v2.py
│   │   └── ...
│
│   ├── enhancer/                   # M1: 图像增强+标题修复
│   │   ├── enhancer_2.py
│   │   ├── merge_subfigures.py     # 合并误分割子图
│   │   ├── caption_guard.py        # 图题/章节标题区分
│   │   ├── heading_cleanup.py      # 页眉页脚去重
│   │   └── __init__.py
│
│   ├── builder/                    # M1+: 树结构构建
│   │   ├── tree_builder.py
│   │   ├── tree_utils.py
│   │   └── __init__.py
│
│   ├── index/                      # M2: 节点摘要 + 向量索引
│   │   ├── node_summarizer.py      # 节点摘要生成（LLM）
│   │   ├── semantic_index.py       # 向量化与索引
│   │   ├── embed_models.py         # 统一embedding接口
│   │   └── __init__.py
│
│   ├── agents/                     # M3: 核心多模态Agent系统
│   │   ├── planner.py              # 规划导航（结构+语义检索）
│   │   ├── observer.py             # 多模态观测
│   │   ├── reasoner.py             # 推理与答案整合
│   │   ├── router.py               # 子问题拆分与调度
│   │   ├── memory.py               # 缓存与上下文记忆
│   │   ├── agent_loop.py           # 主循环 (planner→observer→reasoner)
│   │   └── agent_config.yaml       # Agent系统配置
│
│   ├── eval/                       # M4: 实验评测
│   │   ├── metrics.py              # hit@k, F1, exact match
│   │   ├── ablation_runner.py      # 模块消融测试
│   │   ├── eval_baselines.py       # RAG/全文模型对比
│   │   └── __init__.py
│
│   ├── rag_baseline/               # M4: RAG基线实验
│   │   ├── rag_retrieval.py
│   │   ├── rag_eval.py
│   │   ├── rag_index.py
│   │   └── rag_config.yaml
│
│   ├── utils/                      # 通用函数与工具
│   │   ├── llm_clients.py          # GPT/Qwen调用统一接口
│   │   ├── layout_utils.py         # bbox与几何工具
│   │   ├── logging_utils.py        # 日志系统
│   │   ├── tree_nav.py             # 树结构导航/搜索
│   │   ├── json_io.py              # JSON IO与缓存
│   │   └── __init__.py
│
│   └── __init__.py
│
├── experiments/
│   ├── run_agents.py               # 主Agent实验入口
│   ├── run_rag.py                  # RAG基线实验入口
│   ├── config.yaml                 # 实验配置文件
│   └── analysis.ipynb              # 结果可视化与分析
│
├── results/
│   ├── logs/
│   ├── predictions/
│   ├── traces/                     # reasoning路径与Agent状态
│   └── metrics_summary.json
│
└── AGENTS.md                       # Codex协作指南
```

## 🧪 6. Example Interaction
Q: “How many line plots are there in the report?”

Planner:
	•	Retrieve nodes with summary hints containing figure, plot, chart.
	•	Choose pages labeled as containing “line charts”.

Observer:
	•	Collect all image nodes with metadata "kind": "statistical".
	•	Use VLM to classify chart type.

Reasoner:
	•	Count charts where type = “line”
	•	Return: {answer: 7, support_nodes: [img_3, img_17, img_45]}

## 🧱 7. Development Phases
| Phase | Goal | Key Output |
|-------|---------|----------|
| M0 | PDF → Adapted JSON | content_list.adapted.json |
| M1 | Enhance + DocTree | doctree.mm.json |
| M2 | Summaries + Embedding | index.pkl, summary.json |
| M3 | Agents System | answers.json, reasoning traces |
| M4 | Evaluation | metrics + ablation results |

## 🧩 8. References
	•	Anthropic (2025). Effective Context Engineering for AI Agents.

## 🧭 9. Notes
	•	Always start from DocTree (.mm.json).
	•	Never feed full document text to the Planner directly.
	•	Maintain consistent JSON I/O between modules.

---

## 🧱 10. Index Features (Design + Examples)

This section defines the complete feature schema for agent-controlled retrieval over DocTrees. It introduces dense/sparse views, hierarchical section features, filters/facets, affordances, and explicit graph edges, with concrete JSON examples.

### 10.1 Goals
- Navigable: Sections act as hierarchical routers (coarse → fine).
- Precise: Symbolic fields (labels, schema, units) are queryable via sparse/BM25F.
- Explainable: All results map to nodes and edges; paths can be replayed.
- Performant: Separate ANN per view; BM25F per role. Supports fusion and rerank.

### 10.2 Artifacts and Files
Under `indexes/<doc_id>/`:

```
summary.json
dense_coarse.jsonl       # sections/images/tables (incl. variants for sections)
dense_leaf.jsonl         # leaves (paragraph/list/caption/equation)
sparse_coarse.jsonl      # BM25F docs for coarse nodes
sparse_leaf.jsonl        # BM25F docs for leaf nodes
graph_edges.jsonl        # parent/child/same_page/ref/prev_next/has_col...
id_maps.json             # "Figure 1"→img_id, "Table 2"→tab_id, etc.
coarse.faiss / coarse.vectors.npy + coarse.meta.json
leaf.faiss   / leaf.vectors.npy   + leaf.meta.json
bm25_coarse/  bm25_leaf/
```

Notes:
- For sections, multiple dense variants are emitted (see 10.3.1) either as separate records with `variant` field and `node_id` suffix (e.g., `sec_12#h/#g/#c/#p`) or unified then split during index build.

### 10.3 Schemas

#### 10.3.1 Dense (coarse) — Sections with variants
Variants per section optimize hierarchical navigation and intent routing:
- `heading_text` (`variant: "heading"`): only the section title, for global coarse recall.
- `local_gist` (`variant: "gist"`): 2–3 sentence MMR summary from the section’s own text.
- `child_surface` (`variant: "child"`): synthesized from child captions, table schemas, child headings.
- `path_text` (`variant: "path"`): concatenated ancestor heading path.

Example (one variant shown):

```json
{
  "node_id": "sec_12#c",
  "variant": "child",
  "role": "section",
  "dense_text": "Figure 3 accuracy over time; Table 2 hyper-parameters. Accuracy (%) and Latency (ms) discussed.",
  "title": "2.3 Training",
  "summary": "Optimization details and empirical settings.",
  "filters": {
    "page_idx": 4,
    "level": 2,
    "parent_section": "sec_6",
    "parent_title": "2 Method",
    "page_span": [3, 7],
    "chart_types": ["statistical"],
    "units_set": ["%", "ms"]
  },
  "affordances": {
    "supports_ROUTE_TO_image": true,
    "supports_ROUTE_TO_table": true,
    "has_numbers": true,
    "supports_COMPARE": true,
    "supports_LOOKUP": true,
    "supports_TREND": true
  },
  "subtree_sketch": {
    "keywords_topk": ["accuracy", "%", "latency", "ms", "hyper-parameters"],
    "representatives": {
      "top_image": "img_31",
      "top_table": "tab_21",
      "top_paragraph": "t_230#c0"
    }
  }
}
```

Other variants of the same section share `role/filters/affordances` but differ in `node_id/variant/dense_text`.

#### 10.3.2 Dense (coarse) — Images/Tables

```json
{
  "node_id": "img_14",
  "role": "image",
  "dense_text": "Line chart comparing accuracy across models over time.",
  "title": "Figure 3: Accuracy over time",
  "summary": "A statistical line chart showing models' accuracy changes.",
  "filters": {
    "page_idx": 5,
    "parent_section": "sec_12",
    "parent_title": "2.3 Training",
    "chart_type": "statistical",
    "label": "Figure 3",
    "figure_no": "3",
    "units_set": ["%"]
  },
  "affordances": {
    "has_numbers": true,
    "supports_COMPARE": true,
    "supports_TREND": true,
    "supports_LOOKUP": false
  }
}
```

```json
{
  "node_id": "tab_21",
  "role": "table",
  "dense_text": "Hyper-parameters with accuracy (%) and latency (ms) per configuration.",
  "title": "Table 2: Hyper-parameters and metrics",
  "summary": "Lists accuracy and latency per setting.",
  "filters": {
    "page_idx": 6,
    "parent_section": "sec_12",
    "label": "Table 2",
    "table_no": "2",
    "units_set": ["%", "ms"]
  },
  "affordances": {
    "has_numbers": true,
    "supports_COMPARE": true,
    "supports_LOOKUP": true,
    "supports_TREND": false
  }
}
```

#### 10.3.3 Dense (leaf) — Paragraph/List/Caption/Equation

```json
{
  "node_id": "t_230#c0",
  "orig_node_id": "t_230",
  "role": "paragraph",
  "dense_text": "We report accuracy (%) and latency (ms) for each model.",
  "raw_text": "We report accuracy (%) and latency (ms)...",
  "filters": { "page_idx": 5, "parent_section": "sec_12", "units_set": ["%", "ms"] },
  "refs": [ { "type": "ref", "label": "Figure 3", "target": "img_14" } ]
}
```

#### 10.3.4 Sparse (coarse/leaf) — BM25F Docs
Fields with per-field weights enable precise matches on labels/schema/aliases.

```json
{
  "id": "sec_12",
  "role": "section",
  "heading": "2.3 Training",
  "headings_path": "2 Method > 2.3 Training",
  "child_labels": "Figure 3; Table 2",
  "child_table_schema": "Accuracy (%); Latency (ms)",
  "caption_bag": "line chart accuracy over time; hyper-parameters table",
  "aliases": "acc accuracy pct percent ms millisecond",
  "body": "Optimization details and empirical settings.",
  "filters": { "level": 2, "page_span": [3, 7] }
}
```

```json
{
  "id": "tab_21",
  "role": "table",
  "title": "Table 2: Hyper-parameters and metrics",
  "caption": "Accuracy (%) and latency (ms) per configuration.",
  "table_schema": "Accuracy (%); Latency (ms)",
  "aliases": "acc accuracy pct percent ms millisecond",
  "labels": "Table 2",
  "body": "Lists metrics per setting.",
  "filters": { "page_idx": 6, "parent_section": "sec_12" }
}
```

#### 10.3.5 Graph Edges and ID Maps

```json
{ "src": "sec_12",  "dst": "img_14", "type": "child" }
{ "src": "sec_12",  "dst": "tab_21", "type": "child" }
{ "src": "t_230#c0", "dst": "img_14", "type": "ref", "label": "Figure 3" }
{ "src": "P5",      "dst": "img_14", "type": "same_page" }
{ "src": "tab_21",  "dst": "tab_21:col:Accuracy(%)", "type": "has_col", "unit": "%" }
```

```json
{
  "label2id": { "Figure 3": "img_14", "Table 2": "tab_21" },
  "figure": { "3": "img_14" },
  "table": { "2": "tab_21" }
}
```

### 10.4 Retrieval Strategy (Planner-Ready)
- Intent detection → choose roles/views/filters:
  - Find table/column/unit: prefer `section.child_surface` + `table.schema` hits; `supports_LOOKUP`↑.
  - Find figure/trend: prefer `image.chart_type=statistical`; `supports_TREND`↑; caption/labels hits.
  - Concept/definition: prefer `section.heading/path + local_gist`.
- Dual-channel recall and fusion:
  - Dense ANN (per variant) and BM25F in parallel, each Top-K.
  - Fusion score (default): `0.45*dense + 0.35*bm25f + 0.10*tree_prior + 0.10*intent_bonus`.
  - Tree prior: same_section, page_distance, depth_prior.
- Tree expansion:
  - Hit section → representatives (top_image/table/paragraph), children≤2, siblings≤2, same_page≤8.
  - Hit image/table → add caption, parent_section, referring paragraphs via `ref` edges.
- Lightweight rerank features:
  - `is_title_hit, is_path_hit, is_label_hit, schema_hit, alias_hit, unit_match, page_dist, same_section, supports_* ∧ intent`.

### 10.5 Observer Packaging
- Image/Table: caption + schema_brief + optional cells; always include anchors `page_idx/bbox/cell`.
- Section: very short `summary_gist`.
- Paragraph: when used as evidence, include `raw_text`.

### 10.6 Defaults
- MMR: section k=3 (long k=6), image/table k=3, leaf k=1–2, λ≈0.72.
- Recall: K_dense=200, K_sparse=200 → rerank to 30–50.
- Expansion: parent=1, siblings≤2, children≤2, same_page≤8.
- ANN (reference): HNSW M=32, efC=200, efS=64.

### 10.7 Rationale
- Separate semantic vs symbolic channels reduces noise and improves explainability.
- Section variants align with human navigation: heading/path (global) → child_surface/gist (local).
- Graph edges and affordances support controllable exploration and precise Observer ops.

dense :
  coarse.faiss (section 摘要)
  leaf.faiss （leaf 摘要）
  leaf_raw.faiss（leaf 原文）

sparse:
  bm25_coarse/
  bm25_leaf/

JSON/JSONL
  - summary.json
  - dense_coarse.jsonl：section 多视图（heading/gist/child/path）+ image/table 的 dense
  - dense_leaf.jsonl：叶子摘要（含 raw_text，仅摘要用于向量）
  - sparse_coarse.jsonl：coarse 稀疏（含 title/caption/table_schema/labels/aliases/body）
  - sparse_leaf.jsonl：leaf 稀疏（body=raw_text）
  - graph_edges.jsonl：parent/child/same_page/prev_next/ref/has_col
  - id_maps.json：图表号映射