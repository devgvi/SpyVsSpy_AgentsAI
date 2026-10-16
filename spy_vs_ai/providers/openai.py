"""OpenAI direct (FINAL-2 spec 8): the existing OpenAIClient over the live game's OpenAIDirectTransport.

OPENAI_API_KEY is read by the transport per request, only when this provider is selected; it is never stored,
logged or written to a replay. OpenAI direct is distinct from OpenAI models served through Bedrock."""
from __future__ import annotations

import os
from typing import Any, Callable

from spy_vs_ai.providers.base import DecisionProvider

KEY_ENV = "OPENAI_API_KEY"


class OpenAIProvider(DecisionProvider):
    provider_name = "openai"
    credential_env = (KEY_ENV,)


def build(model: str, options: dict[str, Any], *, scrub=None, opener: Callable[..., Any] | None = None) -> OpenAIProvider:
    from src.llm.clients.openai_compatible_client import OpenAIClient
    from workshop.live.providers import OpenAIDirectTransport
    timeout = float(options.get("timeout_s") or 60.0)
    temperature = None if options.get("temperature", "0") == "provider-default" else float(options.get("temperature", 0.0))
    transport = OpenAIDirectTransport(timeout=timeout, temperature=temperature, opener=opener)
    return OpenAIProvider(OpenAIClient(model_name=model, api_key=None, timeout=timeout, request_fn=transport), scrub=scrub)


def status(options: dict[str, Any] | None = None, *, env=None) -> dict[str, Any]:
    env = os.environ if env is None else env
    if KEY_ENV in env:
        return {"provider": "openai", "state": "configured", "detail": f"{KEY_ENV} is set (value not shown)"}
    return {"provider": "openai", "state": "unavailable", "detail": f"{KEY_ENV} is not set"}
