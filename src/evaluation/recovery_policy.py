"""M6D: retry/repair/fallback as a first-class, manipulable harness
variable (spec sections 3-17).

Three separate concepts, never used interchangeably:

- RETRY: repeating a model invocation after failure (a mechanism, not a
  policy in its own right here).
- REPAIR: the model receives factual information about why its proposal
  was rejected and gets exactly one more chance - the P0-P3 axis below.
- FALLBACK: the harness deterministically supplies an accepted action
  when repair does not succeed - the F0-F2 axis below.

Both axes reuse the exact same shared, production `validate_model_output`
(spec section 6: "all primary policies use the same validator") - this
module only varies WHAT repair TEXT is sent and WHAT fallback action is
selected when recovery fails, never how a response is judged valid.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from src.llm_controller import REPAIR_PROMPT_VERSION, ModelOutputValidation, _normalize_action_name, _repair_instruction, validate_model_output

# -- repair policies (spec section 7) ----------------------------------------

P0_NO_REPAIR_SAFE_FALLBACK = "P0_NO_REPAIR_SAFE_FALLBACK"
P1_CURRENT_ONE_REPAIR = "P1_CURRENT_ONE_REPAIR"
P2_MINIMAL_REPAIR = "P2_MINIMAL_REPAIR"
P3_FULL_LEGAL_REPAIR = "P3_FULL_LEGAL_REPAIR"
REPAIR_POLICIES = (P0_NO_REPAIR_SAFE_FALLBACK, P1_CURRENT_ONE_REPAIR, P2_MINIMAL_REPAIR, P3_FULL_LEGAL_REPAIR)

# -- fallback policies (spec section 15) -------------------------------------

F0_CURRENT_SAFE_FALLBACK = "F0_CURRENT_SAFE_FALLBACK"
F1_WAIT_FALLBACK = "F1_WAIT_FALLBACK"
F2_DETERMINISTIC_FIRST_LEGAL = "F2_DETERMINISTIC_FIRST_LEGAL"
FALLBACK_POLICIES = (F0_CURRENT_SAFE_FALLBACK, F1_WAIT_FALLBACK, F2_DETERMINISTIC_FIRST_LEGAL)

# -- intent-preservation categories (spec section 26) ------------------------

SAME_ACTION_SAME_TARGET = "SAME_ACTION_SAME_TARGET"
SAME_ACTION_DIFFERENT_TARGET = "SAME_ACTION_DIFFERENT_TARGET"
DIFFERENT_ACTION = "DIFFERENT_ACTION"
FALLBACK = "FALLBACK"


def _repair_hints_from_observation(observation: Any, legal_actions: list[dict[str, Any]]) -> tuple[list[str] | None, list[str] | None]:
    """Same source LLMController._attempt_repair already uses (spec section
    24: reuse, never reimplement) - only ever computed for a place_trap
    proposal."""
    from src.llm_controller import place_trap_repair_hints

    return place_trap_repair_hints(observation, legal_actions)


def minimal_repair_instruction(error_type: str | None, error_message: str | None, proposed_action: str | None, proposed_target_id: str | None) -> str:
    """P2 (spec section 10): ONLY the invalid field/value and the factual
    reason - no near_object_id, no legal_action_count, no trap-type or
    legal-target hints. Matches the spec's own example shape exactly
    ('trap_type=\"stun\" is invalid. Return one legal action.')."""
    reason = error_message or (error_type or "invalid")
    proposal_desc = f"{proposed_action}({proposed_target_id})" if proposed_target_id else (proposed_action or "your previous response")
    return f"Your previous proposal {proposal_desc} was rejected: {error_type or 'invalid'} - {reason}. Return one legal action."


def full_legal_repair_instruction(
    error_type: str | None, error_message: str | None, proposed_action: str | None, proposed_target_id: str | None,
    near_object_id: str | None, legal_actions: list[dict[str, Any]], legal_trap_types: list[str] | None,
    legal_place_trap_targets: list[str] | None,
) -> str:
    """P3 (spec section 11): factual reason + the COMPLETE current legal
    action/target set (not just a count, unlike P1) + trap-type/target
    hints where actor-visible - never a ranking or recommendation (spec
    sections 233, 296)."""
    from src.llm_controller import _REPAIR_REASONS

    reason = _REPAIR_REASONS.get(error_type or "", error_message or "it did not contain a valid action")
    proposal_desc = f"{proposed_action}({proposed_target_id})" if proposed_target_id else (proposed_action or "your previous response")
    near_desc = near_object_id if near_object_id else "nothing"
    compact_legal: dict[str, list[str]] = {}
    for entry in legal_actions:
        name = entry.get("action", "wait")
        compact_legal.setdefault(name, [])
        tgt = entry.get("target_id")
        if tgt is not None:
            compact_legal[name].append(tgt)
    trap_desc = ""
    if legal_trap_types is not None:
        trap_desc = f" Allowed trap types in your current inventory: {legal_trap_types}."
    target_desc = ""
    if legal_place_trap_targets is not None:
        target_desc = f" Legal PLACE_TRAP targets right now: {legal_place_trap_targets}."
    return (
        f"Your previous proposal {proposal_desc} was rejected: {error_type or 'invalid'} - {reason}. "
        f"You are currently near: {near_desc}. "
        f"Every currently legal action right now: {compact_legal}.{trap_desc}{target_desc} "
        "Choose exactly one action from the list above. Do not repeat the same rejected action/target."
    )


@dataclass
class RecoveryOutcome:
    policy: str
    original_action: str | None
    original_target: str | None
    repair_attempted: bool
    repair_prompt: str | None
    repair_raw_response: dict[str, Any] | None
    repair_valid: bool | None
    repair_error_type: str | None
    final_action: str | None
    final_target: str | None
    fallback_used: bool
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    intent_category: str = field(default="")
    # Spec section 31: kept strictly separate from repair_valid/fallback_used
    # - a genuine provider/network failure on the repair call itself, never
    # conflated with a content-level validation rejection.
    provider_error_type: str | None = field(default=None)


def classify_intent(original_action: str | None, original_target: str | None, final_action: str | None, final_target: str | None, fallback_used: bool) -> str:
    """Spec sections 26-27: a repaired-but-DIFFERENT action must never be
    called 'repair success' for the original action - it is
    VALID_AFTER_REPAIR (elsewhere) AND ACTION_CHANGED (here), reported as
    two separate facts.

    Compares NORMALIZED action names (`_normalize_action_name` - the same
    canonicalization `validate_model_output` itself applies) - a real bug
    found while first analyzing this milestone's own data: the raw model
    string is case-inconsistent ("inspect" vs "INSPECT" for the identical
    semantic action), which a literal `==` would have misclassified as a
    genuine action change."""
    if fallback_used:
        return FALLBACK
    normalized_original = _normalize_action_name(original_action) if original_action else original_action
    normalized_final = _normalize_action_name(final_action) if final_action else final_action
    if normalized_final == normalized_original:
        if final_target == original_target:
            return SAME_ACTION_SAME_TARGET
        return SAME_ACTION_DIFFERENT_TARGET
    return DIFFERENT_ACTION


def apply_fallback_policy(fallback_policy: str, legal_actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec sections 14-17: mechanically safe and deterministic - never a
    strategic/heuristic choice."""
    if fallback_policy in (F0_CURRENT_SAFE_FALLBACK, F1_WAIT_FALLBACK):
        # Spec section 14: current production fallback is WAIT - F0/F1 are
        # behaviorally identical in this game's current architecture,
        # since WAIT is always legal and is exactly what production
        # already does (documented explicitly, not silently assumed).
        return {"action": "wait"}
    if fallback_policy == F2_DETERMINISTIC_FIRST_LEGAL:
        # Spec section 43: candidate-order dependent BY DESIGN - a harness
        # bias, not model behavior. `enumerate_legal_actions` always adds
        # `{"action": "wait"}` first (see llm_controller.py), so "first
        # legal action" excluding that trivial always-present entry is the
        # only version of this policy that can differ from F0/F1 at all.
        for entry in legal_actions:
            if entry.get("action") != "wait":
                return {"action": entry.get("action"), "target_id": entry.get("target_id")}
        return {"action": "wait"}
    raise ValueError(f"unknown fallback policy {fallback_policy!r}")


def build_repair_prompt(
    policy: str, observation: Any, legal_actions: list[dict[str, Any]], invalid_raw_response: dict[str, Any],
    initial_validation: ModelOutputValidation,
) -> str:
    """The repair TEXT for P1/P2/P3 - shared, runtime-agnostic (spec
    section 6/HP02: one definition, never reimplemented per runtime). Never
    called for P0 (no repair text exists there)."""
    original_action = invalid_raw_response.get("action")
    original_target = invalid_raw_response.get("target_id")
    near_object_id = getattr(observation, "nearby_object_id", None)
    legal_trap_types = None
    legal_place_trap_targets = None
    if original_action == "place_trap":
        legal_trap_types, legal_place_trap_targets = _repair_hints_from_observation(observation, legal_actions)

    if policy == P1_CURRENT_ONE_REPAIR:
        return _repair_instruction(
            initial_validation.error_type, initial_validation.error_message,
            proposed_action=original_action, proposed_target_id=original_target, near_object_id=near_object_id,
            legal_action_count=len(legal_actions), legal_trap_types=legal_trap_types, legal_place_trap_targets=legal_place_trap_targets,
        )
    if policy == P2_MINIMAL_REPAIR:
        return minimal_repair_instruction(initial_validation.error_type, initial_validation.error_message, original_action, original_target)
    if policy == P3_FULL_LEGAL_REPAIR:
        return full_legal_repair_instruction(
            initial_validation.error_type, initial_validation.error_message, original_action, original_target,
            near_object_id, legal_actions, legal_trap_types, legal_place_trap_targets,
        )
    raise ValueError(f"unknown repair policy {policy!r}")


def run_recovery_policy(
    policy: str,
    observation: Any,
    legal_actions: list[dict[str, Any]],
    model_payload: dict[str, Any],
    invalid_raw_response: dict[str, Any],
    initial_validation: ModelOutputValidation,
    model_client: Any,
    system_prompt: str,
    output_schema: dict[str, Any],
    fallback_policy: str = F0_CURRENT_SAFE_FALLBACK,
) -> RecoveryOutcome:
    """The one shared recovery-replay function (spec section 37's
    "preferred causal method"): takes an ALREADY-INVALID raw response +
    its validation result - agnostic to whether that response was
    injected (Phase B) or organically captured from a real first call
    (Phase C) - and applies exactly one of P0-P3, using the same
    validator and (by default) the same fallback throughout Phase 1-3
    (spec section 12). Native call path (plain `generate_action`) - see
    `run_recovery_policy_strands` for the real-SDK Strands equivalent."""
    original_action = invalid_raw_response.get("action")
    original_target = invalid_raw_response.get("target_id")

    if policy == P0_NO_REPAIR_SAFE_FALLBACK:
        fallback_action = apply_fallback_policy(fallback_policy, legal_actions)
        outcome = RecoveryOutcome(
            policy=policy, original_action=original_action, original_target=original_target,
            repair_attempted=False, repair_prompt=None, repair_raw_response=None, repair_valid=None,
            repair_error_type=None, final_action=fallback_action.get("action"), final_target=fallback_action.get("target_id"),
            fallback_used=True, latency_ms=0.0, input_tokens=None, output_tokens=None,
        )
        outcome.intent_category = classify_intent(original_action, original_target, outcome.final_action, outcome.final_target, True)
        return outcome

    repair_prompt = build_repair_prompt(policy, observation, legal_actions, invalid_raw_response, initial_validation)
    repair_payload = dict(model_payload)
    repair_payload["repair_instruction"] = repair_prompt
    repair_payload["repair_prompt_version"] = REPAIR_PROMPT_VERSION
    repair_prompt_json = json.dumps(repair_payload, sort_keys=True)

    start = time.monotonic()
    response = model_client.generate_action(system_prompt, repair_prompt_json, output_schema)
    latency_ms = response.latency_ms if getattr(response, "latency_ms", None) is not None else (time.monotonic() - start) * 1000.0
    repair_raw_response = response.payload if isinstance(response.payload, dict) else {}

    repair_validation = validate_model_output(observation, repair_raw_response, legal_actions)

    if repair_validation.valid:
        final_action = repair_raw_response.get("action")
        final_target = repair_raw_response.get("target_id")
        fallback_used = False
    else:
        fallback_action = apply_fallback_policy(fallback_policy, legal_actions)
        final_action = fallback_action.get("action")
        final_target = fallback_action.get("target_id")
        fallback_used = True

    outcome = RecoveryOutcome(
        policy=policy, original_action=original_action, original_target=original_target,
        repair_attempted=True, repair_prompt=repair_prompt, repair_raw_response=repair_raw_response,
        repair_valid=repair_validation.valid, repair_error_type=repair_validation.error_type,
        final_action=final_action, final_target=final_target, fallback_used=fallback_used,
        latency_ms=latency_ms, input_tokens=getattr(response, "input_tokens", None), output_tokens=getattr(response, "output_tokens", None),
        provider_error_type=getattr(response, "error_type", None),
    )
    outcome.intent_category = classify_intent(original_action, original_target, final_action, final_target, fallback_used)
    return outcome


def run_recovery_policy_strands(
    policy: str,
    observation: Any,
    legal_actions: list[dict[str, Any]],
    model_payload: dict[str, Any],
    invalid_raw_response: dict[str, Any],
    initial_validation: ModelOutputValidation,
    strands_adapter: Any,
    fallback_policy: str = F0_CURRENT_SAFE_FALLBACK,
) -> RecoveryOutcome:
    """Spec sections 47-49 (Q7 - runtime independence): the real-SDK
    Strands equivalent of `run_recovery_policy`. Builds the identical
    P0-P3 repair TEXT via the exact same `build_repair_prompt` (never a
    second, independently-drifting repair-text implementation - the
    HP02 sibling-drift risk this whole module structure exists to avoid),
    but delivers/receives it through the real
    `StrandsAgentAdapter._invoke_strands` SDK call path rather than a
    plain `generate_action` - so this genuinely exercises Strands' own
    tool-calling loop, not a Native call mislabeled as Strands."""
    original_action = invalid_raw_response.get("action")
    original_target = invalid_raw_response.get("target_id")

    if policy == P0_NO_REPAIR_SAFE_FALLBACK:
        fallback_action = apply_fallback_policy(fallback_policy, legal_actions)
        outcome = RecoveryOutcome(
            policy=policy, original_action=original_action, original_target=original_target,
            repair_attempted=False, repair_prompt=None, repair_raw_response=None, repair_valid=None,
            repair_error_type=None, final_action=fallback_action.get("action"), final_target=fallback_action.get("target_id"),
            fallback_used=True, latency_ms=0.0, input_tokens=None, output_tokens=None,
        )
        outcome.intent_category = classify_intent(original_action, original_target, outcome.final_action, outcome.final_target, True)
        return outcome

    repair_prompt = build_repair_prompt(policy, observation, legal_actions, invalid_raw_response, initial_validation)
    invocation = strands_adapter._invoke_strands(model_payload, repair_instruction=repair_prompt)
    repair_raw_response = invocation.proposal if isinstance(invocation.proposal, dict) else {}
    repair_validation = validate_model_output(observation, repair_raw_response, legal_actions)

    if repair_validation.valid:
        final_action = repair_raw_response.get("action")
        final_target = repair_raw_response.get("target_id")
        fallback_used = False
    else:
        fallback_action = apply_fallback_policy(fallback_policy, legal_actions)
        final_action = fallback_action.get("action")
        final_target = fallback_action.get("target_id")
        fallback_used = True

    outcome = RecoveryOutcome(
        policy=policy, original_action=original_action, original_target=original_target,
        repair_attempted=True, repair_prompt=repair_prompt, repair_raw_response=repair_raw_response,
        repair_valid=repair_validation.valid, repair_error_type=repair_validation.error_type,
        final_action=final_action, final_target=final_target, fallback_used=fallback_used,
        latency_ms=invocation.latency_ms, input_tokens=invocation.input_tokens, output_tokens=invocation.output_tokens,
        provider_error_type=invocation.error_type,
    )
    outcome.intent_category = classify_intent(original_action, original_target, final_action, final_target, fallback_used)
    return outcome
