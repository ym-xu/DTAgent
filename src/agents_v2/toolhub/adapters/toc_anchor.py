"""toc_anchor locate adapter."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ...retriever.manager import RetrieverManager
from ...schemas import RetrievalHit
from ..types import ToolCall, ToolResult


def locate(call: ToolCall) -> ToolResult:
    manager = _require_manager(call.args.get("_retriever_manager"))
    keywords = call.args.get("keywords") or call.args.get("section_cues")
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = [k for k in keywords or [] if isinstance(k, str) and k.strip()]
    normalized = [_normalize(k) for k in keywords]
    resources = manager.resources
    heading_index = getattr(resources, "heading_index", {}) or {}
    heading_titles = getattr(resources, "heading_titles", {}) or {}

    if not heading_index:
        return ToolResult(status="empty", data={"hits": []}, metrics={"reason": "no-heading-index"})

    scores: Dict[str, float] = {}
    matched_heading: Dict[str, str] = {}

    for heading_text, node_ids in heading_index.items():
        score = _score_heading(heading_text, normalized)
        if score <= 0:
            continue
        for nid in node_ids:
            if nid:
                current = scores.get(nid, 0.0)
                if score > current:
                    scores[nid] = score
                    matched_heading[nid] = heading_text

    if not scores:
        return ToolResult(status="empty", data={"hits": []}, metrics={"reason": "no-match"})

    hits = [
        RetrievalHit(node_id=node_id, score=min(1.0, sc), tool="toc_anchor.locate")
        for node_id, sc in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]
    payload = {
        "hits": hits,
        "matched_headings": {
            nid: heading_titles.get(nid) or matched_heading.get(nid, "") for nid in scores
        },
    }
    return ToolResult(status="ok", data=payload, metrics={"n_hits": len(hits)})


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _score_heading(heading_text: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0 if heading_text else 0.0
    score = 0.0
    for kw in keywords:
        if kw and kw in heading_text:
            score += 1.0
    if score == 0.0:
        tokens: List[str] = []
        for kw in keywords:
            tokens.extend(kw.split())
        score = sum(1.0 for token in tokens if token and token in heading_text)
    if not keywords:
        return score
    return score / max(1.0, float(len(keywords)))


def _require_manager(manager: Optional[RetrieverManager]) -> RetrieverManager:
    if not isinstance(manager, RetrieverManager):
        raise RuntimeError("toc_anchor.locate requires RetrieverManager instance")
    return manager


__all__ = ["locate"]
