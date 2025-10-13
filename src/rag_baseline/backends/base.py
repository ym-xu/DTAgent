from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Hit:
    rank: int
    score: float
    text: str
    metadata: Dict[str, Any]


@dataclass
class Response:
    query: str
    answer: Optional[str]
    hits: List[Hit]
    backend: str


class RetrieverBackend(ABC):
    """Abstract retriever backend interface.

    Concrete implementations should build an index from a DocTree JSON and
    support querying with unified Response output.
    """

    @abstractmethod
    def build(self, doc: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def query(self, q: str, *, page_view: bool = False, top_k: Optional[int] = None) -> Response:
        ...

    @abstractmethod
    def persist(self) -> None:
        ...

