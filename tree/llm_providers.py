from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProviderConfig:
    provider: str = "mock"  # qwen | gpt | mock
    model: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 0.1
    max_tokens: int = 1500
    timeout_s: int = 60


class BaseProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def run(self, messages: List[Dict[str, Any]], images: Optional[List[Any]] = None) -> str:
        """
        Execute the model call with chat-style messages and optional images.
        Must return raw text (ideally a JSON string per prompt contract).
        Implementations should enforce JSON-only if available and use low temperature.
        """
        raise NotImplementedError


class MockProvider(BaseProvider):
    def run(self, messages: List[Dict[str, Any]], images: Optional[List[Any]] = None) -> str:
        # A trivial identity jsonlist: caller must supply payload and interpret output.
        # We assume the last user message contains a JSON with an "elements" array for demo purposes.
        import json

        text = messages[-1].get("content", "") if messages else ""
        # naive extract node_ids (not robust; only for mock testing without LLM)
        try:
            start = text.index("\n{")
            payload = json.loads(text[start:])
            items = [{"node_id": e["node_id"], "level": 0} for e in payload.get("elements", [])]
            return json.dumps(items, ensure_ascii=False)
        except Exception:
            return json.dumps([], ensure_ascii=False)


class QwenProvider(BaseProvider):
    def run(self, messages: List[Dict[str, Any]], images: Optional[List[Any]] = None) -> str:
        raise NotImplementedError("Implement QwenProvider.run with your VLM client")


class GPTProvider(BaseProvider):
    def run(self, messages: List[Dict[str, Any]], images: Optional[List[Any]] = None) -> str:
        raise NotImplementedError("Implement GPTProvider.run with your LLM client")

