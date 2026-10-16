"""M6D.2: repair CONTRACT design + ROBUSTNESS to harmless representation
perturbations (spec sections 1, 5-13). Reuses M6D.1's canonical
`RepairFacts` unmodified (spec section 3) - this module only adds new
RENDERINGS and, for each, four semantically-harmless PERTURBATIONS
(P0 canonical / P1 section-order reversed / P2 whitespace variant /
P3 neutral-label synonyms), never a new fact source.

Goal is explicitly NOT to maximize retention (spec section 1) - it is to
measure whether a contract's accepted-action distribution stays similar
across perturbations that carry zero new information.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.recovery_corpus import CorpusCase
from src.evaluation.recovery_policy import P1_CURRENT_ONE_REPAIR, build_repair_prompt
from src.evaluation.repair_prompt_variants import RepairFacts, build_repair_facts  # noqa: F401 - re-exported
from src.llm_controller import ModelOutputValidation

N0_CURRENT_PRODUCTION = "N0_CURRENT_PRODUCTION"
N1_NEUTRAL_LABELED = "N1_NEUTRAL_LABELED"
N2_NEUTRAL_STRUCTURED = "N2_NEUTRAL_STRUCTURED"
CONTRACTS = (N0_CURRENT_PRODUCTION, N1_NEUTRAL_LABELED, N2_NEUTRAL_STRUCTURED)

P0_CANONICAL = "P0_CANONICAL"
P1_ORDER_REVERSED = "P1_ORDER_REVERSED"
P2_WHITESPACE_VARIANT = "P2_WHITESPACE_VARIANT"
P3_NEUTRAL_LABEL_SYNONYMS = "P3_NEUTRAL_LABEL_SYNONYMS"
PERTURBATIONS = (P0_CANONICAL, P1_ORDER_REVERSED, P2_WHITESPACE_VARIANT, P3_NEUTRAL_LABEL_SYNONYMS)

_BANNED_PHRASES = (
    "try again", "continue your plan", "preserve your original action", "choose another action",
    "prefer", "best", "recommended", "recommend", "should choose", "optimal",
)


def _field_and_value(facts: RepairFacts) -> tuple[str, str]:
    """Spec section 6-7's granular 'field'/'invalid_value' breakdown -
    derived only from RepairFacts, never a new fact source."""
    if facts.invalid_parameter:
        field = next(iter(facts.invalid_parameter))
        return field, str(facts.invalid_parameter[field])
    if facts.invalid_target:
        return "target_id", facts.invalid_target
    return "action", str(facts.invalid_action_type)


def _legal_constraints(facts: RepairFacts) -> dict[str, list[str]]:
    constraints: dict[str, list[str]] = {}
    if facts.legal_action_types is not None:
        constraints["action_types"] = facts.legal_action_types
    if facts.legal_targets is not None:
        constraints["targets"] = facts.legal_targets
    if facts.legal_enum_values is not None:
        constraints["trap_type"] = facts.legal_enum_values
    return constraints


# ---------------------------------------------------------------------------
# N0 - CURRENT_PRODUCTION (byte-identical prose baseline + 3 harmless perturbations)
# ---------------------------------------------------------------------------


def _render_n0(facts: RepairFacts, case: CorpusCase, v: ModelOutputValidation, perturbation: str) -> str:
    if perturbation == P0_CANONICAL:
        # Byte-identical to real production (spec section 2) - literally
        # calls the same function M6D/M6D.1 used, never reimplemented.
        return build_repair_prompt(P1_CURRENT_ONE_REPAIR, case.observation, case.legal_actions, case.invalid_raw_response, v)

    proposal_desc = f"{facts.invalid_action_type}({facts.invalid_target})" if facts.invalid_target else (facts.invalid_action_type or "your previous response")
    near_desc = facts.near_object_id or "nothing"
    constraints = _legal_constraints(facts)
    constraint_bits = []
    if "trap_type" in constraints:
        constraint_bits.append(f"Allowed trap types in your current inventory: {constraints['trap_type']}.")
    if "targets" in constraints:
        constraint_bits.append(f"Legal PLACE_TRAP targets right now: {constraints['targets']}.")
    rejection = f"Your previous proposal {proposal_desc} was rejected: {facts.validation_reason}."
    near = f"You are currently near: {near_desc}."
    count = f" ({facts.legal_action_count} available)" if facts.legal_action_count is not None else ""
    choose = f"Choose exactly one action from current legal_actions{count}."
    tail = "Do not repeat the same rejected action/target."

    if perturbation == P1_ORDER_REVERSED:
        # Legal-hints clause before the rejection clause - same facts,
        # reversed clause order (mirrors M6D.1's R1-vs-R2 finding, applied
        # here as a PERTURBATION of the production contract itself).
        parts = [" ".join(constraint_bits), rejection, near, choose, tail] if constraint_bits else [rejection, near, choose, tail]
        return " ".join(p for p in parts if p)
    if perturbation == P2_WHITESPACE_VARIANT:
        # Same words, same order - newline-separated instead of
        # space-separated (a purely cosmetic formatting change).
        parts = [rejection, near, choose] + constraint_bits + [tail]
        return "\n".join(p for p in parts if p)
    if perturbation == P3_NEUTRAL_LABEL_SYNONYMS:
        # Same facts/order - synonym wording substituted for key phrases.
        rejection_syn = f"Your previous proposal {proposal_desc} was declined: {facts.validation_reason}."
        choose_syn = f"Pick exactly one action from current legal_actions{count}."
        tail_syn = "Do not repeat the same declined action/target."
        parts = [rejection_syn, near, choose_syn] + constraint_bits + [tail_syn]
        return " ".join(p for p in parts if p)
    raise ValueError(perturbation)


# ---------------------------------------------------------------------------
# N1 - NEUTRAL_LABELED (short labeled factual sections)
# ---------------------------------------------------------------------------


def _n1_sections(facts: RepairFacts) -> tuple[list[str], list[str], list[str]]:
    field, value = _field_and_value(facts)
    error_lines = [f"field: {field}", f"invalid_value: {value}", f"reason: {facts.validation_reason}"]
    constraints = _legal_constraints(facts)
    legal_lines = [f"{k}: {v}" for k, v in constraints.items()]
    response_lines = ["Return one legal action."]
    return error_lines, legal_lines, response_lines


def _render_n1(facts: RepairFacts, perturbation: str) -> str:
    error_lines, legal_lines, response_lines = _n1_sections(facts)
    error_label, legal_label, response_label = "ERROR", "LEGAL CONSTRAINTS", "RESPONSE"

    if perturbation == P3_NEUTRAL_LABEL_SYNONYMS:
        error_label, legal_label, response_label = "INVALID", "ALLOWED VALUES", "INSTRUCTION"

    sections = [(error_label, error_lines), (legal_label, legal_lines), (response_label, response_lines)]
    if perturbation == P1_ORDER_REVERSED:
        sections = [sections[1], sections[0], sections[2]]  # RESPONSE always last (an instruction, not a fact section)

    if perturbation == P2_WHITESPACE_VARIANT:
        # Same labels/content, no blank lines between sections and a
        # single space instead of a colon-newline block - purely cosmetic.
        parts = []
        for label, lines in sections:
            parts.append(f"{label}: " + " ".join(lines))
        return " | ".join(parts)

    blocks = ["\n".join([label] + lines) for label, lines in sections]
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# N2 - NEUTRAL_STRUCTURED (compact machine-like structured representation)
# ---------------------------------------------------------------------------


def _n2_payload(facts: RepairFacts) -> dict[str, Any]:
    field, value = _field_and_value(facts)
    return {
        "error": {"field": field, "invalid_value": value, "reason": facts.validation_reason},
        "legal_constraints": _legal_constraints(facts),
        "response_contract": "one_legal_action",
    }


def _render_n2(facts: RepairFacts, perturbation: str) -> str:
    import json

    payload = _n2_payload(facts)
    key_error, key_legal, key_response = "error", "legal_constraints", "response_contract"

    if perturbation == P3_NEUTRAL_LABEL_SYNONYMS:
        key_error, key_legal, key_response = "invalid", "allowed", "instruction"
        payload = {key_error: payload["error"], key_legal: payload["legal_constraints"], key_response: payload["response_contract"]}

    if perturbation == P1_ORDER_REVERSED:
        # response_contract always last (an instruction, not a fact key).
        payload = {key_legal: payload[key_legal], key_error: payload[key_error], key_response: payload[key_response]}

    body = {"repair": payload}
    if perturbation == P2_WHITESPACE_VARIANT:
        return json.dumps(body, indent=2, sort_keys=False)
    return json.dumps(body, sort_keys=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def render_contract(contract: str, perturbation: str, facts: RepairFacts, case: CorpusCase, v: ModelOutputValidation) -> str:
    if contract == N0_CURRENT_PRODUCTION:
        return _render_n0(facts, case, v, perturbation)
    if contract == N1_NEUTRAL_LABELED:
        return _render_n1(facts, perturbation)
    if contract == N2_NEUTRAL_STRUCTURED:
        return _render_n2(facts, perturbation)
    raise ValueError(f"unknown repair contract {contract!r}")
