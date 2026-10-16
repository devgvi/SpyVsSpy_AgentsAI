"""Anthropic direct (FINAL-2 spec 9): the Anthropic Messages API, distinct from Anthropic models on Bedrock.

The existing `src.llm.clients.AnthropicClient` posts an OpenAI-style body to `<base>/chat/completions`, which the
Anthropic API does not serve; it is left untouched. This adapter keeps the same OpenAICompatibleClient (so the
controller contract is unchanged) and swaps only its transport: the chat payload is converted to a Messages request,
and the Messages response back to the chat-completion shape the client already parses.

ANTHROPIC_API_KEY is read per request, only when this provider is selected; it is sent as the x-api-key header and
never stored, logged or written to a replay. Status: implemented and tested against a fake HTTP layer; never run
against the live API in FINAL-2 (no provider call was approved), so it is labelled EXPERIMENTAL."""
from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Callable

from spy_vs_ai.providers.base import (DecisionProvider, PROVIDER_ACCESS_ERROR, PROVIDER_AUTH_ERROR, PROVIDER_HTTP_ERROR,
                                      PROVIDER_INVALID_RESPONSE, PROVIDER_SERVICE_ERROR, PROVIDER_THROTTLING,
                                      PROVIDER_TIMEOUT, PROVIDER_TRANSPORT_ERROR)
from spy_vs_ai.providers.secrets import Scrubber

KEY_ENV = "ANTHROPIC_API_KEY"
BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
MAX_TOKENS = 1024
_STATUS = {400: PROVIDER_HTTP_ERROR, 401: PROVIDER_AUTH_ERROR, 403: PROVIDER_ACCESS_ERROR, 404: PROVIDER_ACCESS_ERROR,
           408: PROVIDER_TIMEOUT, 429: PROVIDER_THROTTLING, 529: PROVIDER_SERVICE_ERROR}
_SCRUB = Scrubber(["anthropic"])


class AnthropicProviderError(RuntimeError):
    """A classified, redacted Anthropic request failure (the controller records it and falls back, as for OpenAI)."""

    def __init__(self, category: str, message: str, status: int | None = None) -> None:
        self.category = category
        self.status = status
        self.detail = _SCRUB.text(message)
        super().__init__(f"{category}: {self.detail}")


def _json_text(text: str) -> str:
    """The first complete JSON object of a text answer (the Messages API has no JSON mode)."""
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return text[match.start():match.start() + end]
    return stripped


class AnthropicMessagesTransport:
    """`request_fn` for OpenAICompatibleClient: one POST to /v1/messages, converted both ways."""

    def __init__(self, *, base_url: str = BASE_URL, timeout: float = 60.0, temperature: float | None = 0.0,
                 max_tokens: int = MAX_TOKENS, key_env: str = KEY_ENV, opener: Callable[..., Any] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.key_env = key_env
        self._opener = opener or urllib.request.urlopen

    def __repr__(self) -> str:
        return f"AnthropicMessagesTransport(base_url={self.base_url!r}, key_env={self.key_env!r})"

    def build_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        system = "\n\n".join(m["content"] for m in payload.get("messages", []) if m.get("role") == "system")
        messages = [{"role": m["role"], "content": m["content"]} for m in payload.get("messages", [])
                    if m.get("role") in ("user", "assistant")]
        body: dict[str, Any] = {"model": payload["model"], "max_tokens": self.max_tokens, "messages": messages}
        if system:
            body["system"] = system
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body  # response_format / metadata are OpenAI-only and stay local

    @staticmethod
    def to_chat(response: dict[str, Any]) -> dict[str, Any]:
        text = "".join(block.get("text", "") for block in response.get("content") or [] if block.get("type") == "text")
        usage = response.get("usage") or {}
        return {"id": response.get("id"), "model": response.get("model"),
                "choices": [{"message": {"role": "assistant", "content": _json_text(text) if text else text},
                             "finish_reason": response.get("stop_reason")}],
                "usage": {"prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens"),
                          "cached_tokens": usage.get("cache_read_input_tokens")}}

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = os.environ.get(self.key_env)
        if not key:
            raise AnthropicProviderError(PROVIDER_AUTH_ERROR, f"{self.key_env} is not set in the environment")
        request = urllib.request.Request(
            f"{self.base_url}/messages", data=json.dumps(self.build_body(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-api-key": key, "anthropic-version": API_VERSION},
            method="POST")
        try:
            with self._opener(request, timeout=self.timeout) as response:
                return self.to_chat(json.loads(response.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                err = json.loads(exc.read().decode("utf-8")).get("error") or {}
                detail = " ".join(str(err.get(k)) for k in ("type", "message") if err.get(k))
            except Exception:  # noqa: BLE001 - an unreadable body still yields a classified error
                detail = str(exc.reason or "")
            category = _STATUS.get(exc.code) or (PROVIDER_SERVICE_ERROR if exc.code >= 500 else PROVIDER_HTTP_ERROR)
            raise AnthropicProviderError(category, f"HTTP {exc.code} {detail}"[:300], status=exc.code) from None
        except (TimeoutError, socket.timeout):
            raise AnthropicProviderError(PROVIDER_TIMEOUT, f"request timed out after {self.timeout:.0f} s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError | socket.timeout):
                raise AnthropicProviderError(PROVIDER_TIMEOUT, f"request timed out after {self.timeout:.0f} s") from None
            raise AnthropicProviderError(PROVIDER_TRANSPORT_ERROR, f"connection failed: {exc.reason}") from None
        except json.JSONDecodeError:
            raise AnthropicProviderError(PROVIDER_INVALID_RESPONSE, "response body is not JSON") from None


class AnthropicProvider(DecisionProvider):
    provider_name = "anthropic"
    credential_env = (KEY_ENV,)


def build(model: str, options: dict[str, Any], *, scrub=None, opener: Callable[..., Any] | None = None) -> AnthropicProvider:
    from src.llm.clients.openai_compatible_client import OpenAICompatibleClient
    timeout = float(options.get("timeout_s") or 60.0)
    temperature = None if options.get("temperature", "0") == "provider-default" else float(options.get("temperature", 0.0))
    transport = AnthropicMessagesTransport(timeout=timeout, temperature=temperature, opener=opener)
    client = OpenAICompatibleClient("anthropic", model, BASE_URL, api_key=None, timeout=timeout, request_fn=transport)
    return AnthropicProvider(client, scrub=scrub)


def status(options: dict[str, Any] | None = None, *, env=None) -> dict[str, Any]:
    env = os.environ if env is None else env
    if KEY_ENV in env:
        return {"provider": "anthropic", "state": "configured", "detail": f"{KEY_ENV} is set (value not shown)"}
    return {"provider": "anthropic", "state": "unavailable", "detail": f"{KEY_ENV} is not set"}
