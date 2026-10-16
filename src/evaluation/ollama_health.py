"""M5E.1 spec section 20: a safe, read-only local-provider health snapshot
taken at the start and end of a diagnostic run. Never restarts/reloads
Ollama itself (spec section 20 is explicit: only a documented recovery
*step*, elsewhere, may do that) - this module only ever reads."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


def _get_json(url: str, timeout: float = 5.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _post_json(url: str, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any] | None:
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def snapshot_ollama_health(base_url: str = "http://localhost:11434", model_name: str = "gemma4:latest") -> dict[str, Any]:
    """Best-effort, read-only. Every field is None (never a fabricated
    default) if the corresponding endpoint is unreachable - spec section
    20 asks to record what's "available," not to guess the rest."""
    base_url = base_url.rstrip("/")
    version_payload = _get_json(f"{base_url}/api/version")
    show_payload = _post_json(f"{base_url}/api/show", {"model": model_name})
    tags_payload = _get_json(f"{base_url}/api/tags")
    model_names = None
    if tags_payload and isinstance(tags_payload.get("models"), list):
        model_names = sorted(m.get("name") for m in tags_payload["models"] if isinstance(m, dict))

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "provider_process_reachable": version_payload is not None,
        "ollama_version": (version_payload or {}).get("version"),
        "requested_model": model_name,
        "model_present_in_tags": (model_names is not None and model_name in model_names),
        "known_model_tags": model_names,
        "model_metadata": {
            "details": (show_payload or {}).get("details"),
            "model_info_keys": sorted((show_payload or {}).get("model_info", {}).keys()) if show_payload else None,
            # Directly relevant to the context-growth hypothesis (spec
            # section 6/18): the model's configured context window, so a
            # correlation analysis can say whether observed prompt sizes
            # ever came anywhere close to it.
            "context_length": next(
                (v for k, v in (show_payload or {}).get("model_info", {}).items() if k.endswith(".context_length")),
                None,
            ),
            "parameters": (show_payload or {}).get("parameters"),
            "template_present": bool((show_payload or {}).get("template")) if show_payload else None,
        }
        if show_payload is not None
        else None,
    }
