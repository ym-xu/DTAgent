"""
Reasoner
========

支持启发式与可选 LLM 的答案综合模块。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol

from .schemas import Observation, ReasonerAnswer


class LLMCallable(Protocol):
    """约束 LLM 调用接口，便于依赖注入与测试。"""

    def __call__(self, *, question: str, context: str, config: "ReasonerLLMConfig") -> str: ...


@dataclass
class ReasonerLLMConfig:
    """LLM 调用配置。"""

    backend: str = "gpt"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_response_tokens: int = 256


def _default_llm_callable(*, question: str, context: str, config: ReasonerLLMConfig) -> str:
    """默认 LLM 调用实现，使用 utils.llm_clients 中的封装。"""
    from src.utils.llm_clients import gpt_llm_call, qwen_llm_call  # type: ignore

    system_prompt = (
        "You are a document analysis assistant. "
        "Evidence items may contain raw text, structured_table (columns + rows), and vision metadata. "
        "Read tables carefully—match the row/column relevant to the question—and only answer using supported facts. "
        "Return strict JSON with keys: thinking (string, <= 25 words), answer (string), confidence (float), "
        "support_nodes (string[]), reasoning (string). "
        "Thinking should briefly cite the key evidence you will rely on. "
        "If evidence is insufficient, set confidence to 0.0, list any inspected nodes in support_nodes, "
        "and explain the gap in reasoning."
    )
    user_prompt = json.dumps(
        {
            "question": question,
            "evidence": context,
        },
        ensure_ascii=False,
    )
    if config.backend == "qwen":
        result = qwen_llm_call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=config.model,
            json_mode=True,
        )
    else:
        result = gpt_llm_call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=config.model,
            json_mode=True,
        )
    return result or ""


@dataclass
class Reasoner:
    """LLM 驱动的推理器（必须启用 LLM）。"""

    min_confidence: float = 0.6
    use_llm: bool = True
    llm_config: ReasonerLLMConfig = field(default_factory=ReasonerLLMConfig)
    llm_callable: Optional[LLMCallable] = None

    def run(self, question: str, snapshot: Dict[str, object]) -> ReasonerAnswer:
        observations: Dict[str, Observation] = snapshot.get("observations", {})  # type: ignore[assignment]
        if not observations:
            return ReasonerAnswer(
                answer="",
                confidence=0.0,
                support_nodes=[],
                action="REPLAN",
                missing_intent={"need": "observations"},
            )

        selected_obs = self._prepare_observations(observations.values())

        if not self.use_llm:
            raise RuntimeError("Reasoner requires use_llm=True; heuristic fallback is disabled")

        llm_answer = self._run_llm(question, selected_obs)
        print("llm_answer :", llm_answer)
        if llm_answer:
            return llm_answer
        
        return ReasonerAnswer(
            answer="",
            confidence=0.0,
            support_nodes=[obs.node_id for obs in selected_obs],
            action="REPLAN",
            missing_intent={"need": "llm_response"},
        )

    def _prepare_observations(self, observations: Iterable[Observation]) -> List[Observation]:
        """Collect observations without heuristic filtering."""
        seen = set()
        ordered: List[Observation] = []
        for obs in observations:
            if obs.node_id in seen:
                continue
            seen.add(obs.node_id)
            ordered.append(obs)
        return ordered

    def _run_llm(self, question: str, observations: List[Observation]) -> Optional[ReasonerAnswer]:
        llm_call = self.llm_callable or _default_llm_callable
        context_blocks = []
        support_nodes: List[str] = []
        for obs in observations:
            block: Dict[str, object] = {
                "node_id": obs.node_id,
                "modality": obs.modality,
            }
            text = ""
            if isinstance(obs.payload, dict):
                txt = obs.payload.get("text")
                if isinstance(txt, str):
                    text = txt
                table = obs.payload.get("structured_table")
                if isinstance(table, dict):
                    table_copy = {
                        "columns": table.get("columns"),
                        "rows": (table.get("rows") or [])[:6],
                        "caption": table.get("caption"),
                    }
                    block["table"] = table_copy
                list_items = obs.payload.get("structured_list")
                if isinstance(list_items, list) and list_items:
                    cleaned_items = [str(item) for item in list_items if isinstance(item, str) and item.strip()]
                    if cleaned_items:
                        block["list_items"] = cleaned_items
                        if not text:
                            text = "\n".join(cleaned_items)
                        else:
                            text = text + "\n" + "\n".join(cleaned_items)
                vision = obs.payload.get("vision")
                if isinstance(vision, dict):
                    block["vision"] = vision
            block["text"] = text
            context_blocks.append(block)
            support_nodes.append(obs.node_id)
        context = json.dumps(context_blocks, ensure_ascii=False)
        try:
            raw = llm_call(question=question, context=context, config=self.llm_config)
            print(f"[Reasoner LLM Input] {context[:600]}...")
            print(f"[Reasoner LLM Output] {raw[:600]}...")
        except Exception as exc:
            print(f"[Reasoner LLM Error] {exc}")
            return None
        if not raw:
            return None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        thinking = obj.get("thinking") if isinstance(obj, dict) else None
        if isinstance(thinking, str):
            thinking = thinking.strip() or None
        answer = str(obj.get("answer") or "").strip()
        confidence = float(obj.get("confidence") or 0.0)
        reasoning = str(obj.get("reasoning") or "")
        llm_support = obj.get("support_nodes")
        if isinstance(llm_support, list) and llm_support:
            support_nodes = [str(x) for x in llm_support]
        if not answer:
            return None
        if thinking:
            print(f"[Reasoner Thinking] {thinking}")
        final_conf = confidence if confidence > 0 else self.min_confidence
        return ReasonerAnswer(
            answer=answer,
            confidence=final_conf,
            support_nodes=support_nodes,
            reasoning_trace=[reasoning] if reasoning else [],
            thinking=thinking,
        )


__all__ = ["Reasoner", "ReasonerLLMConfig", "LLMCallable"]
