from __future__ import annotations

from typing import Any, Callable

from src.llm.clients.openai_compatible_client import OpenAICompatibleClient


class AnthropicClient(OpenAICompatibleClient):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 10.0,
        request_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__("anthropic", model_name, base_url, api_key, timeout, request_fn)
