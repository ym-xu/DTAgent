from __future__ import annotations

from typing import Any, Dict, Optional

from ..schemas import RetrievalHit
from .types import CommonHit


def common_hit_from_retrieval(
    hit: RetrievalHit,
    *,
    tool_name: str,
    view: Optional[str] = None,
) -> CommonHit:
    view_name = (view or hit.metadata.get("view") or "section#gist")
    evidence_type = map_view_to_evidence(view_name)
    modality = map_view_to_modality(view_name)
    score = clamp_score(hit.score)
    provenance: Dict[str, Any] = {
        "tool": tool_name,
        "view": view_name,
    }
    metadata = dict(hit.metadata)
    page_idx = metadata.pop("page_idx", None)
    if page_idx is not None:
        provenance["page_idx"] = page_idx
    provenance_extra = metadata.pop("provenance", None)
    if isinstance(provenance_extra, dict):
        provenance.update(provenance_extra)
    affordances_raw = metadata.pop("affordances", []) or []
    affordances = _normalize_affordances(affordances_raw)
    meta_payload = metadata.pop("meta", {})
    if not isinstance(meta_payload, dict):
        meta_payload = {}
    meta_payload.update(metadata)
    return CommonHit(
        node_id=hit.node_id,
        evidence_type=evidence_type,
        score=score,
        provenance=provenance,
        modality=modality,
        affordances=affordances,
        meta=meta_payload,
    )


def map_view_to_evidence(view: str) -> str:
    view_low = view.lower()
    if "table" in view_low:
        return "table"
    if "image" in view_low or "fig" in view_low or "chart" in view_low:
        return "graphics"
    if "layout" in view_low or "heading" in view_low:
        return "layout"
    return "text"


def map_view_to_modality(view: str) -> str:
    view_low = view.lower()
    if "table" in view_low:
        return "table"
    if "image" in view_low or "chart" in view_low or "fig" in view_low:
        return "image"
    if "layout" in view_low or "heading" in view_low:
        return "layout"
    return "text"


def clamp_score(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _normalize_affordances(raw) -> list[str]:
    if isinstance(raw, dict):
        return [str(k) for k in raw.keys()]
    if isinstance(raw, list):
        return [str(item) for item in raw if isinstance(item, str)]
    if isinstance(raw, str):
        return [raw]
    return []

