"""Generic visual question answering adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ...observer import _resolve_image_path, _to_data_url  # type: ignore
from ..types import ToolCall, ToolResult

try:
    from src.utils.llm_clients import gpt_llm_call, qwen_llm_call  # type: ignore
except ImportError:  # pragma: no cover
    gpt_llm_call = qwen_llm_call = None


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a vision assistant. Answer the user's question strictly based on the supplied image(s). "
    "Return JSON with keys: thinking (<=25 words), answer (string), confidence (float in [0,1]). "
    "If the answer cannot be determined from the image, set confidence to 0 and explain briefly in thinking."
)


def answer(call: ToolCall) -> ToolResult:
    question = _clean_str(call.args.get("question"))
    if not question:
        return ToolResult(status="error", data={}, metrics={}, error="question is required")

    image_inputs = call.args.get("images") or call.args.get("image_path")
    image_paths = _normalize_images(image_inputs)
    if not image_paths:
        return ToolResult(status="error", data={}, metrics={}, error="images are required")

    base_dir = call.args.get("base_dir")
    if base_dir:
        base_dir = Path(str(base_dir))

    data_urls: List[str] = []
    resolved_paths: List[str] = []
    for image_path in image_paths:
        resolved = _resolve_image_path(Path(image_path), base_dir) if base_dir else _resolve_image_path(Path(image_path), None)
        if not resolved:
            continue
        data_url = _to_data_url(resolved)
        if data_url:
            data_urls.append(data_url)
            resolved_paths.append(str(resolved))

    if not data_urls:
        return ToolResult(status="error", data={}, metrics={}, error="unable to load image paths")

    backend = _clean_str(call.args.get("backend")) or "gpt"
    model = _clean_str(call.args.get("model")) or ("gpt-4o" if backend == "gpt" else "qwen-vl-plus")
    system_prompt = _clean_str(call.args.get("system_prompt")) or DEFAULT_SYSTEM_PROMPT
    extra_context = _clean_str(call.args.get("context"))

    payload = {"question": question}
    if extra_context:
        payload["context"] = extra_context

    base_payload = {"question": question}
    context_text = _clean_str(call.args.get("context"))
    if context_text:
        base_payload["context"] = context_text

    rois = call.args.get("rois")
    roi_list = rois if isinstance(rois, list) else []

    details: List[Dict[str, object]] = []
    for idx, (data_url, resolved_path) in enumerate(zip(data_urls, resolved_paths)):
        payload = dict(base_payload)
        roi_context = _compose_roi_context(roi_list, idx)
        if roi_context:
            payload["context"] = f"{payload.get('context', '')}\n{roi_context}".strip()
        raw = _call_vlm(system_prompt, payload, backend, model, [data_url])
        logger.debug("VLM raw response: %s", raw)
        if not raw:
            details.append(
                {
                    "image": resolved_path,
                    "answer": "",
                    "thinking": None,
                    "confidence": 0.0,
                    "raw": raw,
                }
            )
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"answer": raw.strip(), "thinking": None, "confidence": None}
        details.append(
            {
                "image": resolved_path,
                "answer": _clean_str(parsed.get("answer")) or "",
                "thinking": _clean_str(parsed.get("thinking")),
                "confidence": _to_confidence(parsed.get("confidence")),
                "raw": raw,
            }
        )

    answers = [d["answer"] for d in details if d.get("answer")]
    confidences = [float(d["confidence"]) for d in details if isinstance(d.get("confidence"), float)]
    aggregate_answer = "\n".join(answers)
    aggregate_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    aggregate_thinking = next((d.get("thinking") for d in details if d.get("thinking")), None)

    data = {
        "answer": aggregate_answer,
        "thinking": aggregate_thinking,
        "confidence": aggregate_confidence,
        "backend": backend,
        "model": model,
        "images": resolved_paths,
        "details": details,
    }
    status = "ok" if answers else "empty"
    metrics = {"n_images": len(resolved_paths)}
    return ToolResult(status=status, data=data, metrics=metrics)


def _call_vlm(system: str, payload: dict, backend: str, model: str, images: Sequence[str]) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    if backend == "qwen":
        if qwen_llm_call is None:
            raise RuntimeError("qwen_llm_call is unavailable")
        return qwen_llm_call(messages, model=model, images=list(images), json_mode=True)
    if gpt_llm_call is None:
        raise RuntimeError("gpt_llm_call is unavailable")
    return gpt_llm_call(messages, model=model, images=list(images), json_mode=True)


def _normalize_images(images) -> List[str]:
    if not images:
        return []
    if isinstance(images, str):
        return [images]
    if isinstance(images, (list, tuple, set)):
        out: List[str] = []
        for item in images:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _clean_str(value) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _compose_roi_context(rois: List[dict], index: int) -> Optional[str]:
    if not rois or index >= len(rois):
        return None
    roi = rois[index]
    if not isinstance(roi, dict):
        return None
    parts: List[str] = []
    caption = roi.get("caption")
    if isinstance(caption, str) and caption.strip():
        parts.append(f"Caption: {caption.strip()}")
    description = roi.get("description")
    if isinstance(description, str) and description.strip():
        parts.append(f"Description: {description.strip()}")
    bbox = roi.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        parts.append(f"Bounding box: {bbox}")
    return " | ".join(parts) if parts else None


def _to_confidence(value) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, num))


__all__ = ["answer"]
