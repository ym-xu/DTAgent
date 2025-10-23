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

### 2.0 End-to-End Overview
```
┌──────────────────────────────────────────────────────────────────────────┐
│                               User Question                              │
└───────────────┬──────────────────────────────────────────────────────────┘
                ▼
        ┌───────────────────────┐
        │  1) Router (LLM)      │  ← 解析意图与信号
        │  ─ query_type         │  → RouterDecision (JSON)
        │  ─ signals / risk ... │
        └──────────┬────────────┘
                   ▼
        ┌───────────────────────┐
        │  2) Planner (LLM)     │  ← 承接 RouterDecision
        │  ─ StrategyPlan(JSON) │  → stages / steps / rerank / pack /
        │    • methods          │    coverage_gate / fallbacks / final
        └──────────┬────────────┘
                   ▼
        ┌────────────────────────────────────────────────────────────────┐
        │                3) ToolHub / Executor (非 LLM)                   │
        │                                                                │
        │  3.1 并发检索 lanes（bm25_node.search / table_index.search / …） │
        │          │hits[]              │hits[]              │hits[]      │
        │          ▼                    ▼                    ▼            │
        │  3.2 RRF + rerank(features: year/unit/toc_distance/…)           │
        │          ▼ ranked_hits[]                                        │
        │  3.3 structure.expand → candidates[]                            │
        │  3.4 pack.mmr_knapsack → evidence_pack + coverage               │
        │  3.5 coverage < gate ? ──(yes)─ apply fallbacks ─→ 回到 3.1     │
        └──────────┬──────────────────────────────────────────────────────┘
                   ▼
        ┌────────────────────────────────────────────────────────────────┐
        │                4) Steps 执行器（非 LLM）                        │
        │   顺序执行 Planner.steps：locate / find_regions / extract /     │
        │   vlm_count / compute.eval → 产出中间变量（N_cars、Pct、ANS…） │
        └──────────┬─────────────────────────────────────────────────────┘
                   ▼
   ┌─────────────────────────────────────────────────────────┐
   │ 5) Reader (LLM，可选)                                   │
   │   • 使用 evidence_pack 按约束作答                       │
   └──────────┬──────────────────────────────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────────┐
   │ 6) Judger (LLM/规则)                                    │
   │   • entail/unit/time/conflict/format 校验               │
   │   • 失败 → fallback 或 REPLAN                           │
   └──────────┬──────────────────────────────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────────┐
   │ 7) Finalizer                                            │
   │   • 输出 {answer, format, support, metrics, trace}      │
   └─────────────────────────────────────────────────────────┘
```

**旁路组件**
- DocGraphNavigator：提供树/图邻域（供 `structure.expand` 与 `observation_plan` 使用）
- Memory：缓存问题、Router/Strategy 历史、命中与观察，驱动重试与去重
- Trace Logger（规划中）：记录 plan/hits/ROI/reader/judger/tool-calls，支持复现

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

## 🚀 2.6 agents_v2 Implementation Snapshot

**Design Goals**
- 保持“规划→检索→观察→推理”主循环，但全部组件模块化，便于替换策略或后端。
- 检索前先做策略判定，明确需要的通道（位置跳转 / 稀疏 / 稠密 / 混合）。
- 观察阶段输出结构化证据（文本片段、表格行列、图像说明），Reasoner 统一消费。
- 必须使用 LLM 做最终推理（缺答案时返回 `REPLAN`），便于追踪支持节点。

**模块职责**
- `src/agents_v2/schemas.py`：统一动作、策略、观测、回答等数据结构。
- `src/agents_v2/memory.py`：缓存问题、策略、检索结果与观察证据。
- `src/agents_v2/router.py`：LLM Router，识别问题类型与结构化信号。
- 旧版 `strategy_planner` 已移除，所有规划统一走 Router→Planner→Plan-IR→StrategyPlan 主链路。
- `src/agents_v2/planner.py`：从 Router 决策自动生成检索计划（阶段/步骤/回退），并把策略转换为检索与观察动作。
- `src/agents_v2/toolhub/`：Tool Cards 规范实现，含 ToolRegistry/Executor 与各工具适配器骨架。
- `src/agents_v2/retriever/manager.py`：执行 `jump_to` / 稀疏 / 稠密 / 混合检索，当前使用 JSON 索引（向量、BM25 仍为轻量实现）。
- `src/agents_v2/loaders.py`：加载 summary / dense / sparse / graph_edges，抽取表格结构、图像元信息。
- `src/agents_v2/observer.py`：根据节点类型封装证据；支持注入 LLM 图像分析器（基于 caption 描述）。
- `src/agents_v2/reasoner.py`：过滤证据 → 构造 LLM prompt（包含表格结构、图像描述）→ 解析回答；无启发式兜底。
- `src/agents_v2/orchestrator.py`：协调整个循环，处理 `REPLAN`。
- `src/agents_v2/cli.py`：单文档调试入口，默认启用内置 GPT4o-mini + GPT4o 图像链路，可查看策略、命中、观测和最终回答。

**✅ 已完成**
- 基础模块与数据结构全部落地，并有单元测试覆盖策略、检索、观察、推理及 CLI。
- Loader 会从 `summary.json`/`dense_coarse.jsonl`/`graph_edges.jsonl` 读取索引，拼出 `DocGraphNavigator` 与 `RetrieverResources`，并区分逻辑页/物理页索引供视觉链路使用。
- Planner 已对接新版 ToolHub：表格问题自动生成 `table_index.search → extract.column → compute.filter`，图表问题走 `chart_index.search → extract.chart_read_axis`，视觉问题采用 `page_locator.locate → figure_finder.find_regions → vlm.answer` 三段式。
- Router 会输出 query_type 候选列表；Planner 依据候选自动构造多阶段策略并根据覆盖度/置信度门控是否执行后续阶段。
- Orchestrator 现在仅通过 Router → Planner 生成检索计划，旧版 `RetrievalStrategyPlanner` 已移除，不再提供兜底策略。
- ToolHub 执行层统一为 `CommonHit/CommonError/ToolResult` 契约，核心检索/抽取适配器（dense/bm25/table/chart/page/structure 等）均输出节点级命中及 provenance/meta，便于后续 Pack 与 Reasoner 消费。
- `pack.mmr_knapsack` 与 `compute.eval` 工具已经在 ToolHub 完成实现，可支持证据打包与基础数值运算。
- 图像链路已加载 `figure_spans.jsonl` 并在 `figure_finder`/`chart_screener` 中使用，能够返回 ROI/角色信息并基于启发式识别图表。
- CLI 改为写死默认 LLM/VLM，无需再传入模型参数；仍保留可通过注入 callable 覆盖的测试钩子。
- Reasoner 强制走 LLM 流程；若 LLM 返回空结果，则请求 `REPLAN`，并结合 Judger 得分纳入阶段质量函数。

**⏳ 未完成 / 待补充**
- 表格 HTML 解析：`_parse_table_node` 目前不会解析 HTML `<table>`；需引入解析器提取列/行。
- 真正的视觉模型集成：默认 `vlm.answer` 仍调用 LLM 推理，需要对接真实像素推理服务并补充 ROI 语义。
- 向量 / BM25 检索仍是轻量文本匹配，尚未对接 `.faiss` 或 BM25 文件。
- Reasoner prompt 尚未针对不同任务做更细粒度模板（如数值验证、列名指令等）。
- 缺少完整 trace / metrics 导出，尚未把阶段质量与 Judger 结果写入日志。
- 表格/图像证据仍需要更细粒度的字段（例如单元格定位、坐标信息），同时需要扩展 `compute.*` 链路以覆盖百分比换算、差值等复杂算子。
- 图像/图表链路仍缺真实像素推理及 ROI 坐标级信号，当前 `chart_screener` 仅依赖启发式，需要接入视觉模型提升准确度。
- Judger/格式化工具尚未落地，需要在新的 `ToolResult` 框架上实现 `format.enforce_*` 等以控制证据预算并规范答案形式。
- 检索仍依赖旧版 `RetrieverManager` 的 LLM rerank；后续需改造为“本地检索 → 结构扩展 → `rank.llm` 工具卡”流程，并保留是否启用 LLM 的配置开关。

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

## 🔧 Tool Cards & Execution Skeleton（进行中）

- **RouterDecision (JSON)**：`query_type`、`signals`（page_hint / objects / units / years / operations / expected_format …）、`risk`、`constraints`、`confidence`
- **StrategyPlan (JSON)**：
  ```json
  {
    "stages": [{"stage": "primary", "methods": ["bm25_node.search", "..."], "k_pages": 8, "k_nodes": 50, "page_window": 1}],
    "steps": [{"step_id": "T1", "tool": "bm25_node.search", "args": {...}, "save_as": "H1"}],
    "rerank": {"fuse": "RRF", "features": ["year","unit","toc_distance"], "diversify_by": "section"},
    "pack": {"mmr_lambda": 0.7, "ctx_tokens": 1500, "per_page_limit": 2, "attach": ["caption","table_header"]},
    "coverage_gate": 0.5,
    "fallbacks": [
      {"condition": "coverage<0.5", "action": {"expand_window": 1}},
      {"condition": "visual_low_conf", "action": {"enable": "vlm_verify_topk", "topk": 6}}
    ],
    "final": {"answer_var": "ANS", "format": "int|float|string|list"}
  }
  ```
- **ToolHub 规范**：
  - `ToolCall(tool_id, args)` → `ToolResult(status, data, metrics, error?)`
  - 统一命中结构 `Hit`：`{eid, doc_id, page_idx, node_id, modality, snippet, bbox, score_raw, method, extra}`
  - tool_id 命名：检索 `{name}.search`，定位 `{name}.locate/find_regions`，处理 `{name}.pack|extract|compute`，视觉 `{name}.count/screen`，阅读 `reader.answer`，核验 `judger.verify`
  - 目录 `src/agents_v2/toolhub/`：
    - `types.py`：ToolCall / ToolResult / Hit
    - `registry.py`：ToolRegistry（注册/查找）
    - `executor.py`：ToolExecutor（延迟记录 + 错误收敛）
    - `adapters/`：工具卡按 `{检索 → 扩展 → 打包 → 计算 → 阅读/裁决}` 链路划分（bm25_node/page、table_index、chart_index、structure.expand、pack.mmr_knapsack、page_locator、figure_finder、chart_screener、vlm_count、extract.*、compute.eval、reader.answer、judger.verify）
- **当前状态**：
  - 大部分检索/抽取/打包工具已可用，少量（如 `chart_screener`、`vlm_count`）仍为占位实现。
  - Orchestrator 会先调用 ToolHub（写入占位结果），随后在全部步骤失败时才回退至旧版 `RetrieverManager`。
- **优先事项**：
  1. 实现最小工具集：将 `RetrieverManager`、pack、VLM 计数等逻辑迁移至 ToolHub 适配器。
  2. 完成 ToolHub 执行链：并发检索 → RRF → expand → pack → coverage gate → steps → Reader/Judger → Finalizer。
  3. 新增 `structure.children`、`extract.heading_first` 等卡片，支持“首个子标题”场景。
  4. Trace & Metrics：记录每次 tool 调用的 `tool_id`、`status`、`n_hits`、`latency_ms`、`tokens`，支撑科研复现与性能评估。
