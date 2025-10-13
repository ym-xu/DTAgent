"""
Embedding-based RAG pipeline for tree DocTree.

Modules:
- rag.ingestion.build_nodes: Build LlamaIndex nodes from a DocTree JSON.
- rag.index.embeddings_and_index: Initialize embeddings + storage (FAISS) helpers.
- rag.retrieval.retrieval: Build retrievers/query engines (with page collapse postprocessor).
- rag.app.main_pipeline: End-to-end CLI to build index and run a sample query.

This subpackage does not run on import; external deps (llama-index, faiss-cpu)
should be installed in your environment.
"""

