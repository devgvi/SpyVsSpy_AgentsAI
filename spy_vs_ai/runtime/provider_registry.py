"""The provider registry (FINAL-2 spec 3, 6-11, 21).

A provider is where model inference happens (bedrock, openai, anthropic, ollama; plus `fake`, the offline demo /
test stand-in). Adapter modules are imported only when their provider is selected, so a match never loads an SDK,
reads a credential or opens a connection for a provider it does not use."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from spy_vs_ai.providers.base import ProviderModelClient
from spy_vs_ai.providers.secrets import SECRET_ENV, Scrubber


@dataclass(frozen=True)
class ProviderDescriptor:
    id: str
    label: str
    module: str
    credential_env: tuple[str, ...]   # every variable the adapter may read (names; values are never shown)
    options: tuple[str, ...]          # accepted provider_options keys
    model_examples: tuple[str, ...]   # suggestions only: any provider-specific model id is accepted
    note: str
    user_facing: bool = True          # fake is offered for demos, labelled as such


PROVIDERS: dict[str, ProviderDescriptor] = {
    "bedrock": ProviderDescriptor(
        "bedrock", "Amazon Bedrock", "spy_vs_ai.providers.bedrock",
        ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
         "AWS_DEFAULT_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "BEDROCK_REGION", "AWS_CONFIG_FILE",
         "AWS_SHARED_CREDENTIALS_FILE"),
        ("region", "aws_profile", "timeout_s", "temperature", "request_fields"),
        ("us.anthropic.claude-sonnet-5-5", "us.openai.gpt-6.1-sol"),
        "the research provider: normal AWS credential chain or AWS_BEARER_TOKEN_BEDROCK; region default us-east-1"),
    "openai": ProviderDescriptor(
        "openai", "OpenAI (direct)", "spy_vs_ai.providers.openai", ("OPENAI_API_KEY",), ("timeout_s", "temperature"), (),
        "OPENAI_API_KEY, read only when selected; distinct from OpenAI models on Bedrock"),
    "anthropic": ProviderDescriptor(
        "anthropic", "Anthropic (direct)", "spy_vs_ai.providers.anthropic", ("ANTHROPIC_API_KEY",),
        ("timeout_s", "temperature"), (),
        "ANTHROPIC_API_KEY, read only when selected; Messages API; distinct from Anthropic models on Bedrock"),
    "ollama": ProviderDescriptor(
        "ollama", "Ollama (local)", "spy_vs_ai.providers.ollama", (), ("base_url", "timeout_s"), ("qwen", "gemma4"),
        "local server, default http://localhost:11434; installed models are listed from /api/tags"),
    "fake": ProviderDescriptor(
        "fake", "Fake (offline demo, no model)", "spy_vs_ai.providers.fake", (), ("latency_s", "mode", "fail_every", "seed"),
        ("fake-legal", "fake-slow"), "no network, no credential, no model: a seeded legal-action picker", user_facing=True),
}
PROVIDER_IDS = tuple(PROVIDERS)


def module(provider: str):
    return importlib.import_module(PROVIDERS[provider].module)


def scrubber(providers) -> Scrubber:
    return Scrubber([p for p in providers if p in SECRET_ENV])


def build_client(provider: str, model: str, options: dict[str, Any] | None = None, *, scrub: Scrubber | None = None,
                 **injections: Any) -> ProviderModelClient:
    """The ModelClient the frozen LLMController calls, over the provider's DecisionProvider adapter.

    `injections` are adapter-specific test seams (opener / request_fn / client_factory): never network in tests."""
    options = dict(options or {})
    scrub = scrub or scrubber([provider])
    decision = module(provider).build(model, options, scrub=scrub.text, **injections)
    timeout = float(options.get("timeout_s") or (120.0 if provider == "bedrock" else 60.0))
    return ProviderModelClient(decision, model, timeout_s=timeout)


def status(provider: str, options: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """configured / unavailable (cloud: presence of a credential source, never its value; ollama: reachable or not)."""
    return module(provider).status(options or {}, **kwargs)


def status_all(options: dict[str, dict[str, Any]] | None = None, *, ollama_opener=None) -> dict[str, dict[str, Any]]:
    options = options or {}
    out = {}
    for pid in PROVIDER_IDS:
        kwargs = {"opener": ollama_opener} if pid == "ollama" and ollama_opener is not None else {}
        try:
            out[pid] = status(pid, options.get(pid), **kwargs)
        except Exception as exc:  # noqa: BLE001 - a status check never stops the launcher
            out[pid] = {"provider": pid, "state": "unavailable", "detail": type(exc).__name__}
    return out


__all__ = ["PROVIDERS", "PROVIDER_IDS", "ProviderDescriptor", "build_client", "status", "status_all", "scrubber"]
