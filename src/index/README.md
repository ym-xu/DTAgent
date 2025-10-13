
# Index

下面是一份可直接落地的正式 Indexes 设计方案 + 代码改造意见（v1.0）。它覆盖：产物与文件结构、各索引视图的 Schema、构建与检索流程、与 Planner/Observer/Reasoner 的接口契约、评测与迁移。你可以把这份文档当作实现蓝图。

1. 目标与原则
	•	召回更稳：兼顾语义泛化与符号精确命中（列名、图号、单位、缩写）。
	•	可解释：节点级锚点（page_idx/bbox/cell）与路径可回放。
	•	分职责：dense 只做语义，sparse/filters 承担结构与符号，affordances 服务可执行算子。
	•	可扩展：百万级节点延迟可控（HNSW + BM25F），支持增量更新。
	•	与图上推理一致：所有检索结果都能落在文档图节点上并沿边扩圈。

⸻

2. 总体架构（四视图 + 图关系）

对每个可检索节点（section/image/table/paragraph/list/caption/equation）生成四类视图：
	1.	Dense 语义视图：
	•	字段：dense_text（由“句级打分 + MMR”得到的 1–3 句短摘要）
	•	用途：向量召回与轻重排
	•	约束：不拼 label/page/role/列名/单位等非语义字段
	2.	Sparse 词法视图（BM25F/SPLADE）：
	•	多域：title, caption, table_schema, labels, aliases, body
	•	用途：图号/列名/单位/缩写/数式等精确命中
	3.	Filters/Facets（结构化先验）：
	•	role, page_idx, level, parent_section, chart_type, has_numbers, units_set, col_types,...
	•	用途：Planner 的 must/should 过滤与加权
	4.	Affordances（可执行能力）：
	•	supports_LOOKUP/COMPARE/RANK/DIVIDE/DIFF/TREND...
	•	用途：Observer 选择算子与参数

同时显式维护图关系：
	•	parent/child/sibling/same_page/prev_next/ref（段落→图表/表格引用）
	•	表结构边：table→column→cell（轻量 schema）

⸻

3. 产物与文件组织

建议输出到单一目录 indexes/<doc_id>/：
summary.json                                # 节点摘要/元数据（供可视化/调试）
dense_coarse.jsonl                          # section/image/table 的 dense 记录
dense_leaf.jsonl                            # paragraph/list/caption/equation 的 dense 记录（可选）
sparse_coarse.jsonl                         # coarse 的 BM25F 文档
sparse_leaf.jsonl                           # leaf 的 BM25F 文档（小体量）
graph_edges.jsonl                           # 引用/同页/父子等边
coarse.faiss / coarse.hnsw                  # 向量索引（coarse）
leaf.faiss  / leaf.hnsw                     # 向量索引（leaf，可选）
bm25_coarse/                                # 稀疏索引（Elastic/Lucene/Whoosh 其一）
bm25_leaf/                                  # 稀疏索引（可选）
id_maps.json                                # label→node_id / figure_no→image_id / table_no→table_id 等映射

4. 各记录的 Schema（JSON 约定）

4.1 Dense（coarse）
{
  "node_id": "img_14",
  "role": "image",                   // {"section","image","table"}
  "dense_text": "An example of SciTAB... reasoning graph ...",
  "title": "Figure 1: ...",          // 原始标题（可为空）
  "summary": "2-3 sentences ...",     // 与 dense_text 一致或更详细
  "filters": {
    "page_idx": 1,
    "level": 2,                       // section 有，image/table 无则省略
    "parent_section": "sec_6",
    "parent_title": "1 Introduction",
    "chart_type": "statistical",      // image 可有
    "label": "Figure 1",
    "figure_no": "1"                  // image 可有
  },
  "affordances": {
    "has_numbers": true,
    "supports_COMPARE": true,
    "supports_LOOKUP": false,
    "supports_TREND": false
  }
}

4.2 Dense（leaf）
{
  "node_id": "t_23#c0",
  "orig_node_id": "t_23",
  "role": "paragraph",                // {"paragraph","list","caption","equation"}
  "dense_text": "False Claims. A fact-checking dataset requires ...",
  "raw_text": "原文片段（证据用）",
  "filters": { "page_idx": 2, "parent_section": "sec_22" },
  "refs": [ { "type":"ref", "label":"Figure 2", "target":"img_25" } ]  // 解析到的引用边
}

4.3 Sparse 文档（coarse/leaf 通用）
{
  "id": "tab_33",
  "role": "table",
  "title": "Table 1: Comparison of SCITAB ...",
  "caption": "Table 1: ...",
  "table_schema": "Statistics; TabFact; FEVEROUS; ...; Prop. (%); ...",   // 解析后的首行列名/单位
  "aliases": "NEI not enough info Prod productivity acc accuracy ...",    // 别名/缩写展开
  "labels": "Table 1",                                                    // 图表号
  "body": "2-3句摘要或短正文",
  "filters": { "page_idx": 3, "parent_section": "sec_29" }
}

4.4 图边（graph_edges.jsonl）
{ "src": "sec_6",   "dst": "img_13", "type": "child" }
{ "src": "t_23#c0", "dst": "img_25", "type": "ref", "label": "Figure 2" }
{ "src": "P1",      "dst": "img_13", "type": "same_page" }
{ "src": "tab_33",  "dst": "tab_33:col:Prop.(%)", "type": "has_col", "unit": "%"}

5. 构建流程（Build Pipeline）

5.1 摘要（dense_text）= 句级打分 + MMR
	•	分句：中英混排正则，优先句边界；
	•	打分：TF-IDF(文内) × 标题对齐 × 位置衰减 × 线索词奖励(we propose/results show/compared with) × 数字/单位命中；
	•	MMR：k=3（section/image/table），k=1~2（leaf）；λ≈0.7；
	•	预算：section 128 tokens（长节 512），image/table 96–128，leaf 64。

只把 MMR 摘要写入 dense_text；原文与结构字段不拼入 dense。

5.2 表模式解析（schema + units）
	•	从 HTML 表首行解析列名，提取单位（括号内 %/ms/bps/Hz/USD/℃/...）。
	•	生成：
	•	filters.has_numbers, filters.units_set, filters.col_types([{name, dtype, unit}])
	•	sparse.table_schema（规范化列名串）
	•	affordances（是否支持 LOOKUP/COMPARE/RANK/DIVIDE/TREND）

5.3 图片结构（axis/legend/series）
	•	从 caption/description/VLM 描述中抽取 chart_type/x-axis/y-axis/legend/series_count（匹配“axis/legend/series/curve/bar/line”关键词），填入：
	•	filters.chart_type, affordances.has_numbers/supports_COMPARE/TREND

5.4 别名与单位归一（aliases）
	•	规则库（可 JSON 配置）：
	•	指标：{"f1":"f1 f1-score", "accuracy":"acc accuracy", "latency":"latency delay"}
	•	单位：{"%":"percent pct", "kbps":"kb/s kbps", "usd":"$ usd"}
	•	缩写展开：Long Form (LF) / LF (Long Form) → 两者都加入
	•	写入 sparse.aliases 域。

5.5 引用边（refs）
	•	段落/正文中解析 Figure|Fig.|Table + 编号（含小写字母如 3b），通过 id_maps.json 定位目标节点，写 leaf.refs[] 与 graph_edges.jsonl。

5.6 输出
	•	写出上节的所有 JSONL；
	•	构建 HNSW/FAISS 向量索引（coarse/leaf 分开）；
	•	构建 BM25F 稀疏索引（coarse 必建，leaf 视需要）。

⸻

6. 检索流程（Runtime）

6.1 查询规范化
	•	实体与别名展开（用 id_maps/aliases），单位同义词归一，数值表达规范（7.5 %→7.5%）。

6.2 意图识别 → 角色选择
	•	规则/轻模型判断是否偏 image/table/compare/rank/trend/%/figure/table/column；据此确定 roles ⊆ {section,image,table} 与 filters。

6.3 并发召回与融合
	•	dense：在 roles.dense_text 上做向量 Top-K（如 200）
	•	sparse：BM25F（字段：title,caption,table_schema,labels,aliases,body），Top-K（如 200）
	•	融合：RRF 或加权和（推荐起始权重）
final = 0.45 * dense + 0.35 * bm25f + 0.10 * tree_prior + 0.10 * intent_bonus
	•	tree_prior: same_section, page_distance, depth_prior
	•	intent_bonus: 命中 labels/units/col_names/chart_type 时加分

6.4 树扩圈
	•	命中 section → 加其 children(image/table) + 前2段 paragraph + 同页
	•	命中 image/table → 加其 caption, parent_section, 引用它的段落(refs)
	•	控制半径：siblings≤2, children≤2, same_page≤8

6.5 轻量重排
	•	文本对使用 dense_text；特征：is_title_hit, is_label_hit, has_units, supports_COMPARE, page_dist, same_section
	•	Top-K 收缩到 30–50。

6.6 预算打包（给 LLM 或 Observer）
	•	image/table：caption + schema_brief +（必要时）局部单元格片段
	•	section：summary_gist（很短）
	•	paragraph：仅作为证据时附 raw_text/chunk
	•	始终携带锚点：page_idx/bbox/cell

⸻

7. 与三大模块的接口契约

7.1 Planner 输出：ObservationPlan
{
  "question": "…",
  "candidates": [
    { "node_id":"img_14", "role":"image", "score":0.81, "why":["labels:Figure 1","same_section"] },
    { "node_id":"tab_33", "role":"table", "score":0.77, "why":["table_schema:Prop.(%)"] }
  ],
  "steps": [
    {"op":"OPEN_CAPTION", "node":"img_14", "budget":160},
    {"op":"OPEN_TABLE_SCHEMA", "node":"tab_33", "budget":120},
    {"op":"FIND_COL", "node":"tab_33", "args":{"name":"Prod.","aliases":["productivity","prod"]}},
    {"op":"LOCATE_CELL", "node":"tab_33", "args":{"row":"A","col":"Prod."}},
    {"op":"READ_CELL", "node":"tab_33", "args":{"row":"A","col":"Prod."}}
  ]
}

7.2 Observer 证据（结构化）
{
  "value": "57.5%",
  "type": "number",
  "unit": "%",
  "source": {"node_id":"tab_33", "page_idx":3, "cell":{"row":"A","col":"Prod."}},
  "method": "LOOKUP"
}

7.3 Reasoner Replan 请求
{
  "gaps":[
    {"need":"has_col", "arg":"Prod.", "scope":"same_section"},
    {"need":"figure_ref", "arg":"Figure 1", "scope":"global"}
  ],
  "confidence": 0.58
}

8. 代码改造点（最小侵入）

在你现有的 build_summary() 基础上做四项关键改动：
	1.	产出字段替换/扩展
	•	原 embed_text → 改为 dense_text（不拼 label/hints/filters）；
	•	为每条记录补：filters、affordances；
	•	生成 sparse 文档（coarse/leaf）与 graph_edges。
	2.	MMR 摘要替换截断
	•	引入 summarize_text_block(title, raw, budget_tokens)，在 section/image/table/leaf 分支分别调用，生成 dense_text/summary。
	3.	表模式解析
	•	在 summarize_table 增加 parse_table_schema(html_or_list)，写 filters.* 与 sparse.table_schema，并产出 affordances。
	4.	引用边
	•	在叶子构建时解析 Figure/Table 引用，填充 leaf.refs[] 并写 graph_edges.jsonl；同时维护 id_maps.json（如 "Figure 1"→"img_14"）。

返回值建议：
summary_obj, dense_coarse, dense_leaf, sparse_coarse, sparse_leaf, graph_edges, id_maps

⸻

9. 评测与监控
	•	检索层：nDCG@20、Recall@50、MRR；分题型（问图/问表/问段落）。
	•	证据层：证据命中率（正确 page/bbox/cell）、单位一致率。
	•	端到端：EM/F1、Count-Acc、带证据一致性评分。
	•	性能：P50/P95 延迟；索引构建时间/内存；缓存命中率。
	•	消融：仅 dense → +BM25F → +树先验 → +轻 reranker。

⸻

10. 默认参数（起步可用）
	•	MMR：k_section=3 (or 6 long), k_image/table=3, k_leaf=1~2，λ=0.72
	•	召回：K_dense=200, K_sparse=200 → K_rerank=40
	•	扩圈：parent=1, siblings≤2, children≤2, same_page≤8
	•	融合权重：dense 0.45 / bm25f 0.35 / tree_prior 0.10 / intent_bonus 0.10
	•	ANN：HNSW M=32, efConstruction=200, efSearch=64（参考）

⸻

11. 边界场景与容错
	•	多语种：dense 用多语模型（bge-m3 / gte-multilingual）；sparse 维持英/中并存；别名库多语支持。
	•	方程/公式：不做 dense 摘要，保留原式 + 上下文一句；sparse 加 body。
	•	扫描件：尽量依赖 caption/表头 OCR；失败时仅建 section/image/table 的 dense 与 title/caption 的 sparse。
	•	NEI：Reasoner 在 Δ证据不足/迭代上限时给出“信息不足”并返回已探索子图。
	•	图号歧义（1/1a/1b）：labels 保留完整字符串；id_maps 支持多键映射。

⸻

12. 迁移与兼容
	•	老的向量文件 index_coarse.pkl / index_leaves.pkl 可并行保留一版，切换阶段双写新产物；
	•	检索接口保持 retrieve(q, B) 不变，内部换为新融合流程；
	•	逐步把 Planner 的手写规则下沉为特征，交给轻量 reranker。

⸻

13. 清单（交付与产出）
	•	代码层：
	•	summarize_text_block（MMR）
	•	parse_table_schema（列名/单位/col_types）
	•	extract_aliases（别名/单位归一）
	•	build_sparse_doc（BM25F 文档）
	•	collect_refs（引用边）
	•	build_indexes()（写 JSONL + 构建 HNSW/BM25F）
	•	数据层：
	•	dense_coarse.jsonl / dense_leaf.jsonl
	•	sparse_coarse.jsonl / sparse_leaf.jsonl
	•	graph_edges.jsonl / id_maps.json
	•	coarse.faiss(.hnsw) / leaf.faiss(.hnsw)
	•	bm25_coarse/（与工具相关的索引目录）

⸻

附：从你的样例生成的示例记录（节选）

coarse.dense（image）
{
  "node_id": "img_14",
  "role": "image",
  "dense_text": "An example of the SciTAB dataset and its reasoning graph. Each entry includes paper name, id, table, a claim, and a label (Supported/Refuted/NEI).",
  "title": "Figure 1: An example of our SciTAB dataset ...",
  "summary": "An example of the SciTAB dataset ... claims are verified against tables with reasoning graphs.",
  "filters": { "page_idx": 1, "parent_section": "sec_6", "parent_title": "1 Introduction", "chart_type":"statistical", "label":"Figure 1", "figure_no":"1" },
  "affordances": { "has_numbers": true, "supports_COMPARE": true, "supports_TREND": false, "supports_LOOKUP": false }
}

coarse.sparse（table）
{
  "id": "tab_33",
  "role": "table",
  "title": "Table 1: Comparison of SCITAB ...",
  "caption": "Table 1: The function names, descriptions, and their proportions ...",
  "table_schema": "Function Names; Descriptions; Prop. (%)",
  "aliases": "prop proportion pct percent",
  "labels": "Table 1",
  "body": "Comparison across datasets and function proportions.",
  "filters": { "page_idx": 4, "parent_section": "sec_38" }
}

leaf.dense（paragraph with ref）
{
  "node_id": "t_23#c0",
  "orig_node_id": "t_23",
  "role": "paragraph",
  "dense_text": "False claims are created by generating counter-claims with minimal edits using InstructGPT.",
  "raw_text": "False Claims. A fact-checking dataset requires ...",
  "filters": { "page_idx": 2, "parent_section": "sec_22" },
  "refs": [{ "type":"ref", "label":"Figure 2", "target":"img_25" }]
}

这版方案把“该放 dense 的、该放 sparse 的、该放 filters/affordances 的”彻底拆清楚了，并把“树扩圈 + 图边引用 + 可执行能力”完整对齐到你的 Planner/Observer/Reasoner 流程上。
如果你将 build_summary() 改为返回本方案中的 5~7 个产物（dense/sparse/edges/maps…），你的上层模块几乎不用改接口，只需在检索实现里接入混合召回 + 融合 + 扩圈 + 重排即可。