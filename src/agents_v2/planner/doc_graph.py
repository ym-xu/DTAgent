from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set


@dataclass
class DocGraphNavigator:
    """Simple adjacency helper retained for compatibility."""

    children: Dict[str, List[str]] = field(default_factory=dict)
    parents: Dict[str, str] = field(default_factory=dict)
    same_page: Dict[str, List[str]] = field(default_factory=dict)
    siblings: Dict[str, List[str]] = field(default_factory=dict)

    def expand(self, node_id: str, *, include_self: bool = False) -> List[str]:
        related: List[str] = []
        seen: Set[str] = set()

        def _push(nodes: Iterable[str]) -> None:
            for nid in nodes:
                if nid and nid not in seen:
                    seen.add(nid)
                    related.append(nid)

        if include_self:
            _push([node_id])

        _push(self.children.get(node_id, []))
        parent = self.parents.get(node_id)
        if parent:
            _push([parent])
            _push(self.children.get(parent, []))
        _push(self.same_page.get(node_id, []))
        _push(self.siblings.get(node_id, []))

        if node_id in seen and not include_self:
            related = [nid for nid in related if nid != node_id]
        return related

