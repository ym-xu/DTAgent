from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, Tuple


@dataclass
class Candidate:
    node_id: str
    role: str
    score: float
    why: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    afford: Dict[str, Any] = field(default_factory=dict)


class RetrievalPort(Protocol):
    def dense(self, query: str, roles: set[str], topK: int) -> List[Candidate]:
        ...
    def sparse(self, query: str, roles: set[str], topK: int) -> List[Candidate]:
        ...
    def graph_neighbors(self, node_id: str, types: Tuple[str, ...]) -> List[Tuple[str, str, Dict[str, Any]]]:
        ...
    def idmap_lookup(self, label: str) -> Optional[str]:
        ...
    def page_of(self, node_id: str) -> Optional[int]:
        ...
    def get_attr(self, node_id: str, key: str) -> Optional[Any]:
        ...
    def filter_nodes(self, role: str, filters: List[Dict[str,Any]], scope_nodes: Optional[List[str]]=None) -> List[str]:
        ...
    def expand_nodes(self, base: List[str], types: Tuple[str,...]) -> List[str]:
        ...

