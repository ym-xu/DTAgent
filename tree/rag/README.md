RAG over DocTree (LlamaIndex + FAISS)

Overview
- Ingest a DocTree JSON (tree_builder output) into LlamaIndex nodes.
- Chunk adjacent content within each section into retrievable leaf TextNodes.
- Build FAISS vector store (OpenAI embeddings) and AutoMerging retriever.
- Optional page-collapsed view to surface top pages.

Install
- pip install -U llama-index llama-index-embeddings-openai llama-index-vector-stores-faiss faiss-cpu pyyaml
- export OPENAI_API_KEY=...

Quickstart
- Prepare a doctree JSON, e.g., /path/to/doc_dir/doctree.mm.json
- Run:
  python -m tree.rag.app.main_pipeline --doc /path/to/doc_dir/doctree.mm.json \
    --config tree/rag/config/config.yaml \
    --query "示例问题"

Structure
- rag/ingestion/build_nodes.py: DocTree -> (all_nodes, leaf_nodes)
- rag/index/embeddings_and_index.py: init embeddings + FAISS storage helpers
- rag/retrieval/retrieval.py: build retrievers/query engines (with page collapse)
- rag/app/main_pipeline.py: small CLI driver

Notes
- Chunking keeps content within the same section and minimizes cross-page chunks.
- Leaf nodes carry metadata: doc_id, section_path, page_idx/page_range, source_node_ids.
- Parents (sections) are stored in DocStore for AutoMerging context, not indexed into vectors.

