from __future__ import annotations

import json
from typing import Any

from src.llm.clients.base import ModelCapabilities, ModelResponse


class MockModelClient:
    """Deterministic test client used by M5A and unit tests."""

    def __init__(
        self,
        response: dict[str, Any] | list[dict[str, Any]] | None = None,
        provider: str = "mock",
        model: str = "mock-model",
    ) -> None:
        if response is None:
            self.responses: list[dict[str, Any]] = [{"action": "wait"}]
        elif isinstance(response, list):
            self.responses = list(response)
        else:
            self.responses = [response]
        self.calls = 0
        self.provider = provider
        self.model = model

    def generate_action(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.calls += 1
        payload = self.responses.pop(0) if self.responses else {"action": "wait", "reason": "default safe action"}
        raw_text = json.dumps(payload, sort_keys=True)
        return ModelResponse(
            payload=payload,
            raw_text=raw_text,
            provider=self.provider,
            model=self.model,
            latency_ms=5.0,
            input_tokens=64,
            output_tokens=8,
            finish_reason="stop",
            request_id=f"mock-{self.calls}",
            capabilities=ModelCapabilities(native_structured_output=True),
        )
