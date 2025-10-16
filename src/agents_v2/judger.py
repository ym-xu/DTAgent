"""
LLM Judger
==========

仅使用 LLM 校验 Reasoner 答案是否可接受。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from .schemas import Observation, ReasonerAnswer, RouterConstraints, RouterSignals


class JudgerLLMCallable(Protocol):
    def __call__(
        self,
        *,
        question: str,
        answer: str,
        payload: Dict[str, object],
        config: "JudgerLLMConfig",
    ) -> str: ...


@dataclass
class JudgerLLMConfig:
    backend: str = "gpt"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_response_tokens: int = 256


def _default_llm_callable(
    *,
    question: str,
    answer: str,
    payload: Dict[str, object],
    config: JudgerLLMConfig,
) -> str:
    from src.utils.llm_clients import gpt_llm_call, qwen_llm_call  # type: ignore

    system_prompt = (
        "You are the Judge of a DocTree QA agent. "
        "Given the question, the agent's answer, and the evidence it cited, decide whether the answer "
        "is correct, well-supported, and satisfies the specified constraints. "
        "Respond ONLY with JSON using this schema:\n"
        "{\n"
        '  "pass": true|false,\n'
        '  "issues": [string, ...],\n'
        '  "support_score": float in [0,1],\n'
        '  "recommendation": "accept" | "redo_answer" | "replan" | "return_unanswerable",\n'
        '  "explanation": "brief textual rationale (<= 40 words)"\n'
        "}\n"
        "Use `return_unanswerable` only if the evidence cannot satisfy the constraints and the task allows that. "
        "If the format is wrong but the evidence is sufficient, prefer `redo_answer`. "
        "Consider numerical thresholds strictly. "
    )
    payload_obj = {
        "question": question,
        "answer": answer,
        **payload,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload_obj, ensure_ascii=False)},
    ]
    if config.backend == "qwen":
        return qwen_llm_call(
            messages,
            model=config.model,
            json_mode=True,
            temperature=config.temperature,
            max_tokens=config.max_response_tokens,
        )
    return gpt_llm_call(
        messages,
        model=config.model,
        json_mode=True,
        temperature=config.temperature,
        max_tokens=config.max_response_tokens,
    )


@dataclass
class LLMJudger:
    llm_config: JudgerLLMConfig = field(default_factory=JudgerLLMConfig)
    llm_callable: Optional[JudgerLLMCallable] = None

    def verify(
        self,
        *,
        question: str,
        answer: ReasonerAnswer,
        observations: Dict[str, Observation],
        signals: RouterSignals,
        constraints: RouterConstraints,
    ) -> Optional[Dict[str, object]]:
        llm_call = self.llm_callable or _default_llm_callable
        evidence_payload = _prepare_payload(observations, answer.support_nodes)
        payload = {
            "expected_format": signals.expected_format,
            "units": signals.units,
            "operations": signals.operations,
            "threshold": signals.threshold,
            "comparator": signals.comparator,
            "allow_unanswerable": constraints.allow_unanswerable,
            "evidence": evidence_payload,
        }
        try:
            raw = llm_call(
                question=question,
                answer=answer.answer,
                payload=payload,
                config=self.llm_config,
            )
        except Exception:
            return {
                "pass": False,
                "issues": ["llm_judger_failed"],
                "support_score": 0.0,
                "recommendation": {"action": "replan"},
                "explanation": "LLM judger call failed",
            }
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        issues = data.get("issues") if isinstance(data.get("issues"), list) else []
        recommendation = data.get("recommendation")
        if recommendation not in {"accept", "redo_answer", "replan", "return_unanswerable"}:
            recommendation = "replan" if not data.get("pass") else "accept"

        return {
            "pass": bool(data.get("pass")),
            "issues": issues,
            "support_score": float(data.get("support_score")) if isinstance(data.get("support_score"), (int, float)) else 0.0,
            "recommendation": {"action": recommendation},
            "explanation": data.get("explanation"),
        }


def _prepare_payload(
    observations: Dict[str, Observation],
    support_nodes: List[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    def _truncate(text: str, limit: int = 280) -> str:
        return text if len(text) <= limit else text[: limit - 3] + "..."

    for node_id in support_nodes:
        obs = observations.get(node_id)
        if not obs:
            continue
        payload = obs.payload if isinstance(obs.payload, dict) else {}
        entry: Dict[str, object] = {"node_id": node_id, "modality": obs.modality}
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            entry["text"] = _truncate(text.strip())
        table = payload.get("structured_table")
        if isinstance(table, dict):
            entry["table"] = {
                "columns": (table.get("columns") or [])[:6],
                "rows": (table.get("rows") or [])[:4],
            }
        vision = payload.get("vision")
        if isinstance(vision, dict):
            entry["vision"] = {
                "answer": vision.get("answer"),
                "confidence": vision.get("confidence"),
                "summary": _truncate(vision.get("summary", "")) if isinstance(vision.get("summary"), str) else None,
            }
        records.append(entry)
    if not records:
        for node_id, obs in list(observations.items())[:3]:
            payload = obs.payload if isinstance(obs.payload, dict) else {}
            text = payload.get("text")
            records.append(
                {
                    "node_id": node_id,
                    "modality": obs.modality,
                    "text": _truncate(text) if isinstance(text, str) else None,
                }
            )
    return records


__all__ = ["LLMJudger", "JudgerLLMConfig"]
