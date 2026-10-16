"""Local Ollama (FINAL-2 spec 10): the existing OllamaClient (/api/chat), unchanged. No cloud credential.

The endpoint defaults to http://localhost:11434 and is configurable; installed models can be listed from /api/tags.
The rest of the application starts without Ollama."""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable

from spy_vs_ai.providers.base import DecisionProvider

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(DecisionProvider):
    provider_name = "ollama"
    credential_env = ()


def build(model: str, options: dict[str, Any], *, scrub=None, request_fn: Callable[[dict], dict] | None = None) -> OllamaProvider:
    from src.llm.clients.ollama_client import OllamaClient
    client = OllamaClient(model_name=model, base_url=options.get("base_url") or DEFAULT_BASE_URL,
                          timeout=float(options.get("timeout_s") or 60.0), request_fn=request_fn)
    return OllamaProvider(client, scrub=scrub)


def list_models(base_url: str | None = None, *, timeout: float = 0.5, opener: Callable[..., Any] | None = None) -> list[str]:
    """Installed model names from /api/tags ([] when Ollama is not reachable). A local request, never a model call."""
    opener = opener or urllib.request.urlopen
    try:
        with opener(f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/api/tags", timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - unreachable, refused, not JSON: all mean "no models listed"
        return []
    return sorted(str(m.get("name")) for m in body.get("models") or [] if m.get("name"))


def status(options: dict[str, Any] | None = None, *, env=None, opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    base = (options or {}).get("base_url") or DEFAULT_BASE_URL
    opener = opener or urllib.request.urlopen
    try:
        with opener(f"{base.rstrip('/')}/api/tags", timeout=0.5) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"provider": "ollama", "state": "unreachable", "detail": f"{base} ({type(exc).__name__})", "models": []}
    models = sorted(str(m.get("name")) for m in body.get("models") or [] if m.get("name"))
    return {"provider": "ollama", "state": "reachable", "detail": f"{base}: {len(models)} model(s)", "models": models}
