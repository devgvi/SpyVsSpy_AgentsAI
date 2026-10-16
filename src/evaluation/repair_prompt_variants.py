"""M6D.1: isolates repair-PROMPT REPRESENTATION from repair FACTUAL
CONTENT (spec sections 1, 6-9). `RepairFacts` is the one canonical,
structured semantic source of truth per case - every rendering (R0-R5)
derives from exactly the same `RepairFacts` instance, so a fact can never
silently differ between variants (only wording/order/structure can).

R0 is production, byte-identical, unchanged - it literally calls the same
`build_repair_prompt(P1_CURRENT_ONE_REPAIR, ...)` M6D itself used (spec
section 2: "R0 CURRENT_PRODUCTION remains unchanged"), never a
reimplementation. R1-R5 are new renderings of the identical facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.evaluation.recovery_corpus import CorpusCase
from src.evaluation.recovery_policy import P1_CURRENT_ONE_REPAIR, build_repair_prompt
from src.llm_controller import ModelOutputValidation, _REPAIR_REASONS, place_trap_repair_hints

R0_CURRENT_PRODUCTION = "R0_CURRENT_PRODUCTION"
R1_ERROR_FIRST = "R1_ERROR_FIRST"
R2_LEGAL_FIRST = "R2_LEGAL_FIRST"
R3_COMPACT_FACTUAL = "R3_COMPACT_FACTUAL"
R4_LABELED_STRUCTURED = "R4_LABELED_STRUCTURED"
R5_ACTION_ANCHORED = "R5_ACTION_ANCHORED"

REPAIR_VARIANTS = (R0_CURRENT_PRODUCTION, R1_ERROR_FIRST, R2_LEGAL_FIRST, R3_COMPACT_FACTUAL, R4_LABELED_STRUCTURED, R5_ACTION_ANCHORED)

# Spec section 8: banned strategic-advice phrasing - checked directly by
# tests against every non-R0 rendering.
_BANNED_PHRASES = (
    "try again", "prefer", "continue your previous plan", "choose the best",
    "recommend", "should choose", "optimal", "best target", "best action",
)


@dataclass(frozen=True)
class RepairFacts:
    """Spec section 6: only fields relevant to the case are populated -
    never a fabricated value for an irrelevant field."""

    invalid_action_type: str | None
    invalid_target: str | None
    invalid_parameter: dict[str, Any] | None
    validation_reason: str
    legal_action_types: list[str] | None = None
    legal_targets: list[str] | None = None
    legal_enum_values: list[str] | None = None
    near_object_id: str | None = None
    legal_action_count: int | None = None


def _legal_targets_for_action(action_name: str | None, legal_actions: list[dict[str, Any]]) -> list[str]:
    if not action_name:
        return []
    return sorted(e.get("target_id") for e in legal_actions if e.get("action") == action_name and e.get("target_id"))


def build_repair_facts(case: CorpusCase, initial_validation: ModelOutputValidation | None = None) -> RepairFacts:
    """The one extraction path every rendering derives from - never a
    second, independently-computed fact source per variant (spec section
    9's fact-equivalence requirement is satisfied by construction, not by
    post-hoc verification alone)."""
    v = initial_validation
    action = case.invalid_raw_response.get("action")
    target = case.invalid_raw_response.get("target_id")
    reason = (v.error_message if v else case.validation_error_message) or case.validation_error_message

    invalid_parameter = None
    legal_enum_values = None
    if action == "place_trap":
        trap_type = case.invalid_raw_response.get("trap_type")
        if trap_type is not None:
            invalid_parameter = {"trap_type": trap_type}
        legal_enum_values, legal_targets = place_trap_repair_hints(case.observation, case.legal_actions)
    else:
        legal_targets = _legal_targets_for_action(action, case.legal_actions) or None

    legal_action_types = sorted({e.get("action") for e in case.legal_actions if e.get("action")}) or None

    return RepairFacts(
        invalid_action_type=action, invalid_target=target, invalid_parameter=invalid_parameter,
        validation_reason=reason, legal_action_types=legal_action_types, legal_targets=legal_targets,
        legal_enum_values=legal_enum_values, near_object_id=getattr(case.observation, "nearby_object_id", None),
        legal_action_count=len(case.legal_actions),
    )


def repair_fact_hash(facts: RepairFacts) -> str:
    canonical = json.dumps(asdict(facts), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def repair_representation_hash(rendered_text: str) -> str:
    return hashlib.sha256(rendered_text.encode("utf-8")).hexdigest()


def _proposal_desc(facts: RepairFacts) -> str:
    if facts.invalid_target:
        return f"{facts.invalid_action_type}({facts.invalid_target})"
    return facts.invalid_action_type or "your previous response"


def render_r1_error_first(facts: RepairFacts) -> str:
    """Error explanation first, legal alternatives second."""
    parts = [f"Your previous proposal {_proposal_desc(facts)} was rejected: {facts.validation_reason}."]
    if facts.near_object_id:
        parts.append(f"You are currently near: {facts.near_object_id}.")
    if facts.legal_action_types is not None:
        parts.append(f"Legal action types right now: {facts.legal_action_types}.")
    if facts.legal_targets is not None:
        parts.append(f"Legal targets for this action right now: {facts.legal_targets}.")
    if facts.legal_enum_values is not None:
        parts.append(f"Legal values in your current inventory: {facts.legal_enum_values}.")
    parts.append("Choose exactly one legal action. Do not repeat the same rejected action/target.")
    return " ".join(parts)


def render_r2_legal_first(facts: RepairFacts) -> str:
    """Legal alternatives first, error explanation second - same facts as R1, reordered."""
    parts = []
    if facts.legal_action_types is not None:
        parts.append(f"Legal action types right now: {facts.legal_action_types}.")
    if facts.legal_targets is not None:
        parts.append(f"Legal targets for this action right now: {facts.legal_targets}.")
    if facts.legal_enum_values is not None:
        parts.append(f"Legal values in your current inventory: {facts.legal_enum_values}.")
    parts.append(f"Your previous proposal {_proposal_desc(facts)} was rejected: {facts.validation_reason}.")
    if facts.near_object_id:
        parts.append(f"You are currently near: {facts.near_object_id}.")
    parts.append("Choose exactly one legal action. Do not repeat the same rejected action/target.")
    return " ".join(parts)


def render_r3_compact_factual(facts: RepairFacts) -> str:
    """Short, terse factual wording - same facts, minimal words."""
    bits = [f"{_proposal_desc(facts)} invalid: {facts.validation_reason}."]
    if facts.legal_action_types is not None:
        bits.append(f"types={facts.legal_action_types}")
    if facts.legal_targets is not None:
        bits.append(f"targets={facts.legal_targets}")
    if facts.legal_enum_values is not None:
        bits.append(f"values={facts.legal_enum_values}")
    legal = ("Legal: " + ", ".join(bits[1:]) + ".") if len(bits) > 1 else ""
    near = f"Near: {facts.near_object_id}." if facts.near_object_id else ""
    return " ".join(p for p in (bits[0], legal, near, "One legal action only. No repeats.") if p)


def render_r4_labeled_structured(facts: RepairFacts) -> str:
    """Explicit labeled sections - same facts as R3, structured format."""
    lines = [
        f"INVALID PROPOSAL: {_proposal_desc(facts)}",
        f"REASON: {facts.validation_reason}",
    ]
    if facts.near_object_id:
        lines.append(f"CURRENTLY NEAR: {facts.near_object_id}")
    if facts.legal_action_types is not None:
        lines.append(f"LEGAL ACTION TYPES: {facts.legal_action_types}")
    if facts.legal_targets is not None:
        lines.append(f"LEGAL TARGETS: {facts.legal_targets}")
    if facts.legal_enum_values is not None:
        lines.append(f"LEGAL VALUES: {facts.legal_enum_values}")
    lines.append("INSTRUCTION: Choose exactly one legal action. Do not repeat the same rejected action/target.")
    return "\n".join(lines)


def render_r5_action_anchored(facts: RepairFacts) -> str:
    """Same facts, explicitly NAMES the original attempted action as its
    own callout - never recommends preserving it (spec section 7's own
    'without recommending that it be preserved')."""
    parts = [
        f"You attempted the action: {facts.invalid_action_type}.",
        f"That specific attempt ({_proposal_desc(facts)}) failed: {facts.validation_reason}.",
    ]
    if facts.near_object_id:
        parts.append(f"You are currently near: {facts.near_object_id}.")
    if facts.legal_action_types is not None:
        parts.append(f"Legal action types right now: {facts.legal_action_types}.")
    if facts.legal_targets is not None:
        parts.append(f"Legal targets for this action right now: {facts.legal_targets}.")
    if facts.legal_enum_values is not None:
        parts.append(f"Legal values in your current inventory: {facts.legal_enum_values}.")
    parts.append("Choose exactly one legal action (it need not be the one you just attempted). Do not repeat the same rejected action/target.")
    return " ".join(parts)


def render_variant(variant: str, facts: RepairFacts, case: CorpusCase, initial_validation: ModelOutputValidation) -> str:
    if variant == R0_CURRENT_PRODUCTION:
        # Byte-identical to M6D's own P1 - reuses the real production
        # function, never a reimplementation.
        return build_repair_prompt(P1_CURRENT_ONE_REPAIR, case.observation, case.legal_actions, case.invalid_raw_response, initial_validation)
    if variant == R1_ERROR_FIRST:
        return render_r1_error_first(facts)
    if variant == R2_LEGAL_FIRST:
        return render_r2_legal_first(facts)
    if variant == R3_COMPACT_FACTUAL:
        return render_r3_compact_factual(facts)
    if variant == R4_LABELED_STRUCTURED:
        return render_r4_labeled_structured(facts)
    if variant == R5_ACTION_ANCHORED:
        return render_r5_action_anchored(facts)
    raise ValueError(f"unknown repair variant {variant!r}")
