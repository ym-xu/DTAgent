"""
Observer
========

根据 Planner 提供的节点列表采集证据。
这里实现一个基于内存字典的轻量观察器，并提供基于 LLM 的图像分析能力。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .memory import AgentMemory
from .schemas import Observation


@dataclass
class NodeEvidence:
    """节点对应的静态证据。"""

    node_id: str
    modality: str = "text"
    content: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)


ImageAnalyzer = Callable[[NodeEvidence], Dict[str, object]]


@dataclass
class Observer:
    """基于内存映射的观察器。"""

    store: Dict[str, NodeEvidence]
    image_analyzer: Optional[ImageAnalyzer] = None

    @classmethod
    def from_doc_dir(cls, doc_dir: "Path | str") -> "Observer":
        from pathlib import Path

        from .loaders import build_observer_store

        base = Path(doc_dir)
        store = build_observer_store(base)
        return cls(store=store)

    def observe(self, nodes: Iterable[str], memory: AgentMemory) -> List[Observation]:
        observations: List[Observation] = []
        for node_id in nodes:
            if node_id in memory.observations:
                observations.append(memory.observations[node_id])
                continue
            evidence = self.store.get(node_id)
            if not evidence:
                continue
            payload: Dict[str, object] = {"text": evidence.content, **evidence.extra}
            if evidence.modality == "table" and "table" in evidence.extra:
                payload.setdefault("structured_table", evidence.extra["table"])
            if evidence.modality == "image":
                if self.image_analyzer:
                    try:
                        vision = self.image_analyzer(evidence)
                    except Exception:
                        vision = {}
                    if vision:
                        payload.setdefault("vision", vision)
                        summary = vision.get("summary") if isinstance(vision, dict) else None
                        if summary and not payload.get("text"):
                            payload["text"] = summary
            obs = Observation(node_id=node_id, modality=evidence.modality, payload=payload)
            memory.remember_observation(obs)
            observations.append(obs)
        return observations


def build_llm_image_analyzer(
    *,
    backend: str = "gpt",
    model: str = "gpt-4o-mini",
    image_root: Optional[Path] = None,
    llm_callable: Optional[Callable[..., str]] = None,
) -> ImageAnalyzer:
    """基于 LLM 的图像分析器，返回结构化描述。"""

    def _call_llm(payload: dict, images: Optional[List[str]] = None) -> str:
        if llm_callable:
            return llm_callable(payload=payload, backend=backend, model=model, images=images)
        from src.utils.llm_clients import gpt_llm_call, qwen_llm_call  # type: ignore

        system = (
            "You are an image analyst. Given metadata about a chart or figure, respond with strict JSON "
            "containing thinking (string, <= 20 words), summary (string), key_entities (string[]), chart_type (string|null)."
        )
        user = json.dumps(payload, ensure_ascii=False)
        if backend == "qwen":
            return qwen_llm_call([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ], model=model, json_mode=True, images=images)
        return gpt_llm_call([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], model=model, json_mode=True, images=images)

    def analyzer(evidence: NodeEvidence) -> Dict[str, object]:
        payload = {
            "node_id": evidence.node_id,
            "caption": evidence.extra.get("image", {}).get("caption") if isinstance(evidence.extra.get("image"), dict) else evidence.extra.get("image"),
            "description": evidence.extra.get("image", {}).get("description") if isinstance(evidence.extra.get("image"), dict) else None,
            "label": evidence.extra.get("label"),
            "hints": evidence.extra.get("hints"),
        }
        image_paths: List[str] = []
        stored_path = evidence.extra.get("image_path")
        if isinstance(stored_path, str):
            resolved = _resolve_image_path(Path(stored_path), image_root)
            if resolved:
                data_url = _to_data_url(resolved)
                if data_url:
                    image_paths.append(data_url)
        debug_info = {
            "node_id": evidence.node_id,
            "image_paths": image_paths,
        }
        raw = _call_llm(payload, images=image_paths or None) or ""
        debug_info["raw"] = raw
        print(f"[ImageAnalyzer] {json.dumps(debug_info, ensure_ascii=False)[:400]}...")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        summary = data.get("summary") if isinstance(data, dict) else None
        chart_type = data.get("chart_type") if isinstance(data, dict) else None
        entities = data.get("key_entities") if isinstance(data, dict) else None
        thinking = data.get("thinking") if isinstance(data, dict) else None
        if isinstance(thinking, str):
            thinking = thinking.strip() or None
        if thinking:
            print(f"[Image Thinking] {thinking}")
        result = {
            "summary": summary or evidence.content,
            "chart_type": chart_type,
            "key_entities": entities if isinstance(entities, list) else None,
            "thinking": thinking,
        }
        return result

    return analyzer


def _to_data_url(path: Path) -> Optional[str]:
    try:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _resolve_image_path(path: Path, root: Optional[Path]) -> Optional[Path]:
    if path.is_absolute() and path.exists():
        return path
    candidates: List[Path] = []
    if root is not None:
        candidates.append(root / path)
        for parent in list(root.parents)[:3]:
            candidates.append(parent / path)
    candidates.append(path)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


__all__ = ["Observer", "NodeEvidence", "build_llm_image_analyzer", "ImageAnalyzer"]
