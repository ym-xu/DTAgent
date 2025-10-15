"""
RetrievalStrategyPlanner
========================

保留原先基于 LLM 的检索计划生成逻辑，为平稳过渡到 Router+Planner 架构。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

from .schemas import StrategyKind, StrategyPlan, StrategyStep


class StrategyLLMCallable(Protocol):
    def __call__(self, *, question: str, config: "StrategyLLMConfig") -> str: ...


@dataclass
class StrategyLLMConfig:
    backend: str = "gpt"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_response_tokens: int = 256


def _default_llm_callable(*, question: str, config: StrategyLLMConfig) -> str:
    from src.utils.llm_clients import gpt_llm_call, qwen_llm_call  # type: ignore

    system_prompt = (
        "You are the planner for a DocTree QA agent. Decide which retrieval tools should run next, "
        "grounded in the question. Tools and arguments must follow the schema exactly.\n"
        "\n"
        "Valid tools:\n"
        "- jump_to_label(label: string)\n"
        "- jump_to_page(page: integer)\n"
        "- dense_search(query: string, view: string, queries?: string[])\n"
        "- sparse_search(query?: string, keywords?: string[], view: string)\n"
        "- hybrid_search(query?: string, keywords?: string[], view: string)\n"
        "\n"
        "Allowed dense/sparse views: section#gist, section#heading, section#child, section#path, image, table.\n"
        "Do not invent new view names. Prefer:\n"
        "- section#heading or jump_to_label when the question names a heading, figure, or table number.\n"
        "- section#child or sparse_search when column names, units, or specific attributes are needed.\n"
        "- hybrid_search when both semantic context and symbolic hints seem important.\n"
        "- Combine multiple tools only if each adds new coverage; avoid duplicate steps.\n"
        "\n"
        "Rewrite queries to highlight key entities, years, units, and synonyms. Remove conversational words "
        "such as 'what' or 'please'. Provide sparse_search keywords whenever possible (concise phrases, 1-4 words each), "
        "including plausible paraphrases the document might use (e.g., variations of 'research questions', 'key questions', "
        "'question list'). Avoid standalone generic words like 'number' or 'count' unless they are part of a meaningful phrase.\n"
        "\n"
        "Example 1 (fictional report \"Urban Mobility Outlook 2035\"):\n"
        "Question: \"What does Table 7 list about electric bus deployments?\"\n"
        "Response JSON:\n"
        "{\n"
        "  \"thinking\": \"Table label plus column lookup provide the needed detail.\",\n"
        "  \"strategy\": \"COMPOSITE\",\n"
        "  \"steps\": [\n"
        "    {\"tool\": \"jump_to_label\", \"args\": {\"label\": \"Table 7\"}},\n"
        "    {\"tool\": \"sparse_search\", \"args\": {\"keywords\": [\"electric bus\", \"deployment metrics\"], \"view\": \"section#child\"}},\n"
        "    {\"tool\": \"dense_search\", \"args\": {\"queries\": [\"electric bus deployment\", \"Table 7 deployment details\"], \"view\": \"section#gist\"}}\n"
        "  ],\n"
        "  \"confidence\": 0.87,\n"
        "  \"notes\": \"Table 7 is referenced directly; sparse search will capture column labels.\",\n"
        "  \"hints\": [\"inspect table caption\", \"collect column names\"],\n"
        "  \"retrieval_keys\": [\"electric bus deployment\", \"Table 7 metrics\"]\n"
        "}\n"
        "\n"
        "Example 2 (fictional whitepaper \"Coastal Energy Study\"):\n"
        "Question: \"Summarize the safety guidance mentioned in Section 4.\"\n"
        "Response JSON:\n"
        "{\n"
        "  \"thinking\": \"Need the section heading plus its summary context on safety guidance.\",\n"
        "  \"strategy\": \"COMPOSITE\",\n"
        "  \"steps\": [\n"
        "    {\"tool\": \"jump_to_page\", \"args\": {\"page\": 12}},\n"
        "    {\"tool\": \"dense_search\", \"args\": {\"query\": \"section 4 safety guidance\", \"view\": \"section#gist\"}},\n"
        "    {\"tool\": \"sparse_search\", \"args\": {\"keywords\": [\"safety guidance\", \"section 4\"], \"view\": \"section#heading\"}}\n"
        "  ],\n"
        "  \"confidence\": 0.78,\n"
        "  \"notes\": \"Page index inferred from Section 4 locator in the table of contents.\",\n"
        "  \"retrieval_keys\": [\"Section 4 safety guidance\"]\n"
        "}\n"
        "\n"
        "Return only strict JSON with keys: thinking (string, <= 25 words), strategy ('SINGLE'|'COMPOSITE'), "
        "steps (array), confidence (0-1 float), optional notes (string), optional hints (string array), optional retrieval_keys (string array)."
    )
    payload = json.dumps({"question": question}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]
    if config.backend == "qwen":
        return qwen_llm_call(messages, model=config.model, json_mode=True)
    result = gpt_llm_call(messages, model=config.model, json_mode=True)
    print("strategy raw:", result)
    return result


@dataclass
class RetrievalStrategyPlanner:
    """沿用旧版的 LLM 检索策略生成，后续将由 Router+StrategyCard 替换。"""

    llm_config: StrategyLLMConfig = field(default_factory=StrategyLLMConfig)
    llm_callable: Optional[StrategyLLMCallable] = None
    allowed_views: Sequence[str] = field(
        default_factory=lambda: (
            "section#gist",
            "section#heading",
            "section#child",
            "section#path",
            "image",
            "table",
        )
    )

    def decide(self, question: str) -> StrategyPlan:
        normalized = question.strip()
        if not normalized:
            raise ValueError("Question must be non-empty for strategy decision")

        llm_call = self.llm_callable or _default_llm_callable
        raw = llm_call(question=normalized, config=self.llm_config)
        if not raw:
            raise RuntimeError("Strategy LLM returned empty response")

        data = self._parse_json(raw)
        thinking = self._parse_thinking(data.get("thinking"))
        steps = self._build_steps(data.get("steps", []))
        if not steps:
            raise RuntimeError("Strategy LLM produced no retrieval steps")
        steps = self._normalize_steps(steps)
        if not steps:
            raise RuntimeError("Strategy LLM produced only duplicate or invalid steps")

        strategy_kind = self._parse_strategy_kind(data.get("strategy"), len(steps))
        confidence = self._parse_confidence(data.get("confidence"))
        notes = data.get("notes")
        hints = self._parse_hints(data.get("hints"))
        retrieval_keys = self._parse_retrieval_keys(data.get("retrieval_keys"))
        if thinking:
            print(f"[Strategy Thinking] {thinking}")
        if retrieval_keys:
            print(f"[Strategy Keys] {retrieval_keys}")

        return StrategyPlan(
            strategy=strategy_kind,
            steps=steps,
            confidence=confidence,
            notes=notes if isinstance(notes, str) and notes.strip() else None,
            hints=hints,
            thinking=thinking,
            retrieval_keys=retrieval_keys,
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, object]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Strategy LLM response is not valid JSON: {exc}") from exc

    @staticmethod
    def _build_steps(items) -> List[StrategyStep]:
        steps: List[StrategyStep] = []
        if not isinstance(items, list):
            return steps
        for item in items:
            if not isinstance(item, dict):
                continue
            tool = item.get("tool")
            if not isinstance(tool, str):
                continue

            base_args = item.get("args")
            args = dict(base_args) if isinstance(base_args, dict) else {}

            for key, value in item.items():
                if key in {"tool", "args", "weight"}:
                    continue
                args.setdefault(key, value)

            weight = item.get("weight")
            if isinstance(weight, (int, float)):
                steps.append(StrategyStep(tool=tool, args=args, weight=float(weight)))
            else:
                steps.append(StrategyStep(tool=tool, args=args))
        return steps

    @staticmethod
    def _parse_strategy_kind(value, step_count: int) -> StrategyKind:
        if isinstance(value, str):
            key = value.strip().upper()
            if key in StrategyKind.__members__:
                return StrategyKind[key]
        return StrategyKind.SINGLE if step_count == 1 else StrategyKind.COMPOSITE

    @staticmethod
    def _parse_confidence(value) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _parse_string_list(value) -> Optional[List[str]]:
        if not isinstance(value, list):
            return None
        result: List[str] = []
        for item in value:
            if isinstance(item, str):
                trimmed = item.strip()
                if trimmed:
                    result.append(trimmed)
        return result or None

    def _parse_hints(self, value) -> Optional[List[str]]:
        return self._parse_string_list(value)

    def _parse_retrieval_keys(self, value) -> Optional[List[str]]:
        return self._parse_string_list(value)

    @staticmethod
    def _parse_thinking(value) -> Optional[str]:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return None

    def _normalize_steps(self, steps: List[StrategyStep]) -> List[StrategyStep]:
        allowed_view_set = set(self.allowed_views)
        normalized: List[StrategyStep] = []
        seen_keys: set[tuple] = set()
        for step in steps:
            tool = step.tool
            args = dict(step.args or {})

            def _clean_query(value: object) -> Optional[str]:
                if not isinstance(value, str):
                    return None
                cleaned = " ".join(value.strip().split())
                return cleaned or None

            def _clean_list(value: object) -> List[str]:
                items: List[str] = []
                if isinstance(value, list):
                    for element in value:
                        if isinstance(element, str):
                            cleaned = " ".join(element.strip().split())
                            if cleaned:
                                items.append(cleaned)
                return items

            if tool == "jump_to_page":
                page = args.get("page")
                if isinstance(page, int):
                    tool = "page_locator.locate"
                    args = {"pages": [page]}
                else:
                    continue
            elif tool == "jump_to_label":
                label = args.get("label")
                if isinstance(label, str) and label.strip():
                    tool = "bm25_node.search"
                    args = {"keywords": [label.strip()], "view": "section#child"}
                else:
                    continue

            if tool in {"dense_search", "dense_node.search"}:
                tool = "dense_node.search"
                view = args.get("view")
                if not isinstance(view, str) or view not in allowed_view_set:
                    args["view"] = "section#gist"

                keywords = _clean_list(args.get("keywords"))
                if keywords:
                    args["keywords"] = keywords
                elif "keywords" in args:
                    args.pop("keywords", None)

                queries_list = _clean_list(args.get("queries"))
                if queries_list:
                    args["queries"] = queries_list
                elif "queries" in args:
                    args.pop("queries", None)

                query = _clean_query(args.get("query"))
                if not query and queries_list:
                    query = queries_list[0]
                if not query and keywords:
                    query = " ".join(keywords)
                if query:
                    args["query"] = query
                else:
                    args.pop("query", None)

                if not args.get("query") and not args.get("keywords"):
                    raise RuntimeError(f"{tool} requires a query or keywords")
            elif tool in {"sparse_search", "hybrid_search", "bm25_node.search"}:
                tool = "bm25_node.search"
                keywords = _clean_list(args.get("keywords"))
                if keywords:
                    args["keywords"] = keywords
                elif "keywords" in args:
                    args.pop("keywords", None)

                query = _clean_query(args.get("query"))
                if query:
                    args["query"] = query
                elif "query" in args:
                    args.pop("query", None)

                view = args.get("view")
                if isinstance(view, str) and view not in allowed_view_set:
                    args["view"] = "section#child"

                if not args.get("query") and not args.get("keywords"):
                    raise RuntimeError("bm25_node.search requires a query or keywords")

            def _canonical(value: object):
                if isinstance(value, list):
                    return tuple(value)
                return value

            key = (
                tool,
                tuple(sorted((k, _canonical(v)) for k, v in args.items())),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            normalized.append(StrategyStep(tool=tool, args=args, weight=step.weight))
        return normalized


__all__ = ["RetrievalStrategyPlanner", "StrategyLLMConfig", "StrategyLLMCallable"]
