from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from src.llm.clients.base import ModelCapabilities, ModelResponse


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs (including local servers)."""

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        request_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.request_fn = request_fn

    def generate_action(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        started = time.monotonic()
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if request_context:
            payload["metadata"] = request_context

        if self.request_fn is not None:
            raw_response = self.request_fn(payload)
        else:
            raw_response = self._request_http(payload)

        end = time.monotonic()
        content = None
        message = None
        if isinstance(raw_response, dict):
            choices = raw_response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
            usage = raw_response.get("usage") or {}
            if raw_response.get("model"):  # the model the provider says it served (provenance)
                self.actual_models_seen = getattr(self, "actual_models_seen", set()) | {str(raw_response["model"])}
            payload_value = content
            if isinstance(payload_value, str):
                try:
                    payload_value = json.loads(payload_value)
                except json.JSONDecodeError:
                    # M5B.1: an empty dict is the honest "couldn't parse a
                    # usable action" signal - never fabricate a WAIT here.
                    payload_value = {}
            elif payload_value is None:
                payload_value = {}
            if not isinstance(payload_value, dict):
                payload_value = {}
            return ModelResponse(
                payload=payload_value,
                raw_text=content,
                provider=self.provider_name,
                model=self.model_name,
                latency_ms=round((end - started) * 1000.0, 2),
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                # OpenAI reports prompt-cache hits under prompt_tokens_details (and
                # prompt_tokens includes them); older/compatible servers use cached_tokens
                cached_input_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens",
                                                                               usage.get("cached_tokens")),
                finish_reason=(choices[0].get("finish_reason") if choices else None),
                request_id=raw_response.get("id"),
                capabilities=ModelCapabilities(native_structured_output=True, token_usage=True),
            )

        return ModelResponse(
            payload={},
            raw_text=str(raw_response),
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=round((end - started) * 1000.0, 2),
            error_type="provider_error",
            error_message="unexpected provider response",
            capabilities=ModelCapabilities(native_structured_output=True),
        )

    def _request_http(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class OpenAIClient(OpenAICompatibleClient):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 10.0,
        request_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__("openai", model_name, base_url, api_key, timeout, request_fn)


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
