from __future__ import annotations

from typing import Optional

import faiss  # type: ignore
from llama_index.core import Settings, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.faiss import FaissVectorStore


def init_embedding(model: str = "text-embedding-3-small", dim: int = 1536) -> int:
    """Initialize global embedding model for LlamaIndex.

    Returns the embedding dimension.
    """
    Settings.embed_model = OpenAIEmbedding(model=model)
    return dim


def init_storage_and_index(embed_dim: int, persist_dir: Optional[str] = None) -> StorageContext:
    """Create FAISS vector store + docstore and wrap in a StorageContext.

    Inner product index is used (cosine after normalization handled by LI).
    """
    faiss_index = faiss.IndexFlatIP(embed_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    docstore = SimpleDocumentStore()
    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store, persist_dir=persist_dir
    )
    return storage_context


def persist(storage_context: StorageContext) -> None:
    storage_context.persist()


def load_index_from_persist(persist_dir: str):
    from llama_index.core import load_index_from_storage

    sc = StorageContext.from_defaults(persist_dir=persist_dir)
    idx = load_index_from_storage(storage_context=sc)
    return idx, sc

