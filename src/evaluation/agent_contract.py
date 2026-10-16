"""M6F spec sections 3-4, 9-11: the canonical, runtime-neutral AgentRequest
/ AgentResponse boundary (HP04 - "A multi-runtime agent system needs a
runtime-neutral request/response boundary. Otherwise runtime adapters can
silently redefine prompt, memory, tools, repair, or telemetry semantics.").

This module does not replace Native's or Strands' own call paths - it sits
ABOVE them, formalizing the logical contract both already implicitly
agree on (compact_observation_for_model + AgentMemory.to_compact_dict() +
legal_actions + RepairFacts + fixed generation settings), so that
agreement becomes an executable, hashable, testable object instead of an
implicit convention. `build_agent_request()` is the one place a canonical
request is assembled; `native_model_payload()` / `strands_model_payload()`
(the latter returns the former) derive each runtime's payload from that
same object, and `native_system_prompt()` adds the Native system prompt.
The production adapters still build their own payloads from an
Observation; parity tests (RP13-RP18) check that they agree with this
contract - never a second, independently-drifting construction of
either per caller (the HP02 "Sibling Runtime Drift" principle this
session has enforced since M6D).

Exact class shape may differ from the spec's own field list (spec section
3) - the requirement is that these semantics exist and are checked BEFORE
any runtime adapter runs, not any particular Python layout.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.llm.personality import render_personality_block
from src.llm_controller import (
    NATIVE_TOOL_LIKE_REPRESENTATION,
    NATIVE_TOOL_LIKE_SYSTEM_PROMPT,
    _normalize_action_name,
    compact_observation_for_model,
    serialize_observation,
    validate_model_output,
)

FIXED_TEMPERATURE = 0.0


def _hash_json(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentRequest:
    """The canonical, runtime-neutral decision request (spec section 3).

    `observation_payload` is the compact, tool-like-formatted observation
    (`compact_observation_for_model(..., representation_variant=
    NATIVE_TOOL_LIKE_REPRESENTATION)`) WITHOUT `strategy_memory` folded in -
    memory travels separately as `memory_payload` so each runtime adapter
    decides, on its own, how (not whether) to combine them. This is the
    one deliberate structural choice this module makes beyond the spec's
    own field list, and it is exactly what RP13 exists to verify never
    silently diverges.
    """

    actor_id: str
    episode_id: str
    observation_payload: dict[str, Any] = field(repr=False)
    memory_payload: dict[str, Any] = field(repr=False)
    personality_policy: str | None
    mission_context: dict[str, Any] = field(repr=False)
    legal_actions: tuple[dict[str, Any], ...] = field(repr=False)
    repair_context: dict[str, Any] | None = field(repr=False)
    generation_settings: dict[str, Any] = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def observation_hash(self) -> str:
        return _hash_json(self.observation_payload)

    def memory_hash(self) -> str:
        return _hash_json(self.memory_payload)

    def personality_hash(self) -> str:
        return _hash_json(self.personality_policy)

    def legal_actions_hash(self) -> str:
        return _hash_json(list(self.legal_actions))

    def repair_facts_hash(self) -> str | None:
        if self.repair_context is None:
            return None
        return _hash_json(self.repair_context)

    def generation_settings_hash(self) -> str:
        return _hash_json(self.generation_settings)

    def request_semantic_hash(self) -> str:
        """Spec section 9: the single hash that must match across runtimes
        before ANY behavioral interpretation is allowed (spec section 34).
        Deliberately excludes `metadata` and `episode_id` - those are
        diagnostic/bookkeeping fields, never part of the model-facing
        decision itself (an episode_id differing between a Native run and
        a Strands run must never register as a semantic divergence)."""
        return _hash_json({
            "actor_id": self.actor_id,
            "observation_payload": self.observation_payload,
            "memory_payload": self.memory_payload,
            "personality_policy": self.personality_policy,
            "mission_context": self.mission_context,
            "legal_actions": list(self.legal_actions),
            "repair_context": self.repair_context,
            "generation_settings": self.generation_settings,
        })

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["legal_actions"] = list(self.legal_actions)
        return d


@dataclass
class AgentResponse:
    """The canonical, runtime-neutral decision response (spec section 4)."""

    raw_model_output: dict[str, Any]
    normalized_action: dict[str, Any]
    visible_rationale: str | None
    tool_calls: int
    runtime_metadata: dict[str, Any]
    provider_metadata: dict[str, Any]
    validation_status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_agent_request(
    *,
    actor_id: str,
    episode_id: str,
    observation: Any,
    legal_actions: list[dict[str, Any]],
    memory_payload: dict[str, Any],
    personality_profile: str | None,
    model_id: str,
    provider_name: str,
    repair_context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentRequest:
    """The one canonical construction path (never duplicated per runtime -
    HP02). `legal_actions` is passed explicitly rather than re-derived from
    `observation`, matching every existing M6-family driver's own pattern
    (snapshot builders already compute and freeze it alongside the
    observation)."""
    observation_payload = serialize_observation(observation)
    observation_payload["legal_actions"] = legal_actions
    compact = compact_observation_for_model(observation_payload, representation_variant=NATIVE_TOOL_LIKE_REPRESENTATION)
    generation_settings = {
        "model_id": model_id,
        "provider": provider_name,
        "temperature": FIXED_TEMPERATURE,
        "max_tokens": None,
        "thinking": None,
    }
    return AgentRequest(
        actor_id=actor_id,
        episode_id=episode_id,
        observation_payload=compact,
        memory_payload=dict(memory_payload or {}),
        personality_policy=personality_profile,
        mission_context=dict(compact.get("mission", {})),
        legal_actions=tuple(legal_actions),
        repair_context=dict(repair_context) if repair_context is not None else None,
        generation_settings=generation_settings,
        metadata=dict(metadata or {}),
    )


def native_model_payload(request: AgentRequest, repair_instruction: str | None = None) -> dict[str, Any]:
    """RP13/RP16: Native's runtime-specific representation of the
    canonical request - the exact same compact-payload-plus-
    strategy_memory shape every M6-family Native driver has built since
    M6D (spec section 14: "adapters translate; they do not redefine")."""
    payload = dict(request.observation_payload)
    payload["strategy_memory"] = request.memory_payload
    if repair_instruction:
        from src.llm_controller import REPAIR_PROMPT_VERSION

        payload["repair_instruction"] = repair_instruction
        payload["repair_prompt_version"] = REPAIR_PROMPT_VERSION
    return payload


def native_system_prompt(request: AgentRequest) -> str:
    block = render_personality_block(request.personality_policy)
    return f"{NATIVE_TOOL_LIKE_SYSTEM_PROMPT}\n\n{block}" if block is not None else NATIVE_TOOL_LIKE_SYSTEM_PROMPT


def strands_model_payload(request: AgentRequest, repair_instruction: str | None = None) -> dict[str, Any]:
    """RP13/RP16: Strands' own `_invoke_strands` builds this identically
    (`dict(model_payload)` + `repair_instruction`/`repair_prompt_version`
    when present) - reproduced here, not reimplemented, so a test can
    compare it byte-for-byte against `native_model_payload`'s logical
    content without needing a live Strands call."""
    return native_model_payload(request, repair_instruction=repair_instruction)


def normalize_raw_output(raw: dict[str, Any]) -> dict[str, Any]:
    """RP17: equivalent model outputs must normalize equivalently before
    validation, regardless of which runtime produced them. Both runtimes'
    real proposal dicts already share this shape
    (`action`/`target_id`/`trap_type`/`reason`) - this makes that
    normalization explicit and independently testable rather than
    assumed."""
    return {
        "action": _normalize_action_name(raw.get("action")) if raw.get("action") is not None else None,
        "target_id": raw.get("target_id"),
        "trap_type": raw.get("trap_type"),
        "visible_rationale": raw.get("reason"),
    }


def build_agent_response(
    *,
    raw_model_output: dict[str, Any],
    observation: Any,
    legal_actions: list[dict[str, Any]],
    tool_calls: int,
    runtime_metadata: dict[str, Any],
    provider_metadata: dict[str, Any],
) -> AgentResponse:
    normalized = normalize_raw_output(raw_model_output or {})
    v = validate_model_output(observation, raw_model_output or {}, legal_actions)
    return AgentResponse(
        raw_model_output=dict(raw_model_output or {}),
        normalized_action=normalized,
        visible_rationale=normalized.get("visible_rationale"),
        tool_calls=tool_calls,
        runtime_metadata=dict(runtime_metadata or {}),
        provider_metadata=dict(provider_metadata or {}),
        validation_status="valid" if v.valid else f"invalid:{v.error_type}",
    )
