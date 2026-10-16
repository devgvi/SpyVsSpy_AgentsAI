"""M5D: compatibility preflight.

Answers exactly one question, before anything else happens: can this
(agent_runtime, provider, model) combination even run? Never starts an
episode, never touches GameState/Observation, never makes a gameplay
decision. This is the mechanism that keeps "the pipeline couldn't run"
distinct from "the agent lost the mission" (M5B.1's core lesson, extended
from model-output validity to whole-episode compatibility).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.evaluation.spec import VariantSpec


class CompatibilityStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    MODEL_CAPABILITY_MISMATCH = "MODEL_CAPABILITY_MISMATCH"
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"
    RUNTIME_INIT_FAILURE = "RUNTIME_INIT_FAILURE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_INIT_FAILURE = "PROVIDER_INIT_FAILURE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    UNKNOWN_COMPATIBILITY_FAILURE = "UNKNOWN_COMPATIBILITY_FAILURE"


@dataclass(frozen=True)
class CompatibilityResult:
    status: CompatibilityStatus
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.status is CompatibilityStatus.SUPPORTED

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "reason": self.reason, "detail": self.detail}


# Spec section 8: which capabilities each runtime actually requires from the
# provider/model. Native only ever sends/parses plain JSON text (M5A/M5B.1) -
# it never asks the provider for native tool calling. Strands' M5C.1 adapter
# always calls the choose_action tool, so the model must support Ollama's
# "tools" capability (confirmed live: qwen:latest lacks it, gemma4:latest
# has it - see M5C.1's CHANGELOG entry).
RUNTIME_REQUIRED_CAPABILITIES: dict[str, frozenset[str]] = {
    "native": frozenset(),
    "strands": frozenset({"tools"}),
    # M5D.7/M5D.8: native_tool_like is still plain completion-API JSON
    # (NativeToolLikeAgentAdapter/LLMController) - no Ollama "tools"
    # capability requirement, same as native.
    "native_tool_like": frozenset(),
    # M5E: "scripted" is a deterministic reference agent_runtime (spec
    # section 21) - ScriptedBotController, no model/provider call at all,
    # so no capability ever needed.
    "scripted": frozenset(),
}


def _query_ollama_capabilities(model: str, base_url: str | None, timeout: float = 5.0) -> tuple[bool, list[str] | None]:
    """Returns (server_reachable, capabilities). capabilities is None if the
    server responded but doesn't list this model at all."""
    url = (base_url or "http://localhost:11434").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return False, None
    for entry in data.get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            return True, list(entry.get("capabilities", []))
    return True, None


def check_compatibility(variant: VariantSpec) -> CompatibilityResult:
    """Pure-ish preflight (one read-only network call for Ollama capability
    metadata; nothing is mutated, no engine/Game object is ever created)."""
    required_caps = RUNTIME_REQUIRED_CAPABILITIES.get(variant.agent_runtime)
    if required_caps is None:
        return CompatibilityResult(
            CompatibilityStatus.CONFIGURATION_ERROR,
            f"unknown agent_runtime {variant.agent_runtime!r}",
        )

    if variant.agent_runtime == "strands":
        try:
            from src.agents.strands_agent import _import_strands

            _import_strands()
        except Exception as exc:  # StrandsUnavailableError, or an unexpected import-time error
            return CompatibilityResult(CompatibilityStatus.RUNTIME_UNAVAILABLE, str(exc))

    if required_caps and variant.provider == "ollama":
        reachable, capabilities = _query_ollama_capabilities(variant.model, variant.llm_base_url)
        if not reachable:
            return CompatibilityResult(
                CompatibilityStatus.PROVIDER_UNAVAILABLE,
                f"could not reach Ollama server at {variant.llm_base_url or 'http://localhost:11434'}",
            )
        if capabilities is None:
            return CompatibilityResult(
                CompatibilityStatus.MODEL_UNAVAILABLE,
                f"model {variant.model!r} was not found on the Ollama server",
            )
        missing = sorted(required_caps - set(capabilities))
        if missing:
            return CompatibilityResult(
                CompatibilityStatus.MODEL_CAPABILITY_MISMATCH,
                f"{variant.agent_runtime!r} requires capabilities {sorted(required_caps)}, "
                f"but {variant.model!r} only supports {capabilities} (missing: {missing})",
                detail={"capabilities": capabilities, "required": sorted(required_caps)},
            )

    return CompatibilityResult(CompatibilityStatus.SUPPORTED)
