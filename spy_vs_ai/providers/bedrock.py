"""Amazon Bedrock (FINAL-2 spec 7): the existing, known-good BedrockClient (Converse), unchanged.

Credentials come only from the normal AWS credential chain (profile / environment / SSO) or
AWS_BEARER_TOKEN_BEDROCK; nothing is hard-coded or stored. The region is configurable (default us-east-1)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

from spy_vs_ai.providers.base import DecisionProvider

DEFAULT_REGION = "us-east-1"


class BedrockProvider(DecisionProvider):
    provider_name = "bedrock"
    credential_env = ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                      "AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "BEDROCK_REGION",
                      "AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE")


def prompt_cache_supported(model_id: str) -> bool:
    """An explicit cache point only for Anthropic models on Bedrock (workshop/live/providers.py, same rule)."""
    lowered = model_id.lower()
    return "anthropic" in lowered or "claude" in lowered


def build(model: str, options: dict[str, Any], *, scrub=None, client_factory: Callable[[], Any] | None = None) -> BedrockProvider:
    from src.llm.clients.bedrock_client import BedrockClient
    if options.get("aws_profile"):
        os.environ["AWS_PROFILE"] = options["aws_profile"]  # a profile name, not a secret (the live game does the same)
    temperature = None if options.get("temperature", "provider-default") == "provider-default" else float(options["temperature"])
    request_fields = options.get("request_fields")
    client = BedrockClient(model_name=model, region=options.get("region") or DEFAULT_REGION,
                           timeout=float(options.get("timeout_s") or 120.0), client_factory=client_factory,
                           temperature=temperature, accept_invoke_markup=True,
                           cache_system_prompt=prompt_cache_supported(model),
                           additional_request_fields=json.loads(request_fields) if isinstance(request_fields, str)
                           else request_fields)
    return BedrockProvider(client, scrub=scrub)


def status(options: dict[str, Any] | None = None, *, env=None) -> dict[str, Any]:
    """Configured / unavailable, from presence only: no credential value is read, no request is made."""
    env = os.environ if env is None else env
    if importlib.util.find_spec("boto3") is None:
        return {"provider": "bedrock", "state": "unavailable", "detail": "boto3 is not installed"}
    sources = [n for n in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE") if n in env]
    home = Path.home() / ".aws"
    if (home / "config").exists() or (home / "credentials").exists():
        sources.append("~/.aws config")
    if (options or {}).get("aws_profile"):
        sources.append("configured profile")
    if not sources:
        return {"provider": "bedrock", "state": "unavailable", "detail": "no AWS credential source found"}
    return {"provider": "bedrock", "state": "configured",
            "detail": f"credential source present ({', '.join(sources)}; not verified, values not read); region "
                      f"{(options or {}).get('region') or DEFAULT_REGION}"}
