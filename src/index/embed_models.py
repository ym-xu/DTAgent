"""
Sentence Transformers 嵌入器（强制依赖）。

系统仅面向英文文档，默认且仅支持 Sentence Transformers。
不可用时直接报错，不做哈希或其他兜底。

用法:
    from .embed_models import get_embedder
    emb = get_embedder("BAAI/bge-small-en-v1.5")
    X = emb.encode(["hello world"])  # np.ndarray (N, D), float32 且已归一化
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (arr / norms).astype(np.float32)


@dataclass
class Embedder:
    name: str
    dim: int

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """Sentence Transformers 封装。模型不可用时抛出异常。"""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "`sentence-transformers` 未安装或不可用，请安装并确保模型可用"
            ) from e

        self._model = SentenceTransformer(model_name)
        dim = getattr(self._model, "get_sentence_embedding_dimension", lambda: 384)()
        super().__init__(name=model_name, dim=int(dim))

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        X = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(X, dtype=np.float32)


def get_embedder(name: Optional[str] = None) -> Embedder:
    model_name = name or "BAAI/bge-small-en-v1.5"
    return SentenceTransformerEmbedder(model_name)
