"""M5F.3A.3: deterministic, evaluation-only ordering control for
model-visible serialized lists (spec sections 5-7, 22-26, 30).

M5F.3A.2's decisive probe found that `Observation.objects`' list order -
not geometry, not label content - was the true driver of a >=80%
first-item selection rate in an otherwise fully symmetric harness. This
module exists to (a) quantify that effect precisely and (b) provide a
reusable, deterministic counterbalancing primitive so social-influence
experiments can be run without list-order confounding treatment
conditions.

Every function here operates on a COPY of an already-built `Observation`
(itself immutable - `@dataclass(frozen=True)`) via `dataclasses.replace`,
never on `GameState`/`Level`'s own authoritative collections (spec
section 30: "counterbalancing should operate on a copied serialized view
... do not reorder authoritative GameState collections"). Canonical
production gameplay (`Game`/`GameState.new_game`, never touched by this
module) is completely unaffected by default - this is opt-in evaluation
tooling only (spec section 7).
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

from src.observation import Observation


class OrderPolicy(Enum):
    CANONICAL = "CANONICAL"
    REVERSED = "REVERSED"
    COUNTERBALANCED = "COUNTERBALANCED"


def _permuted(items: list[Any], policy: OrderPolicy, trial_index: int) -> list[Any]:
    """Spec section 23's two-item rule (generalizes to any list length):
    CANONICAL never reorders; REVERSED always reverses; COUNTERBALANCED
    alternates by `trial_index` parity - deterministic, reproducible,
    never uncontrolled runtime randomness (spec section 6)."""
    if policy is OrderPolicy.CANONICAL:
        return list(items)
    if policy is OrderPolicy.REVERSED:
        return list(reversed(items))
    if policy is OrderPolicy.COUNTERBALANCED:
        return list(items) if trial_index % 2 == 0 else list(reversed(items))
    raise ValueError(f"unknown OrderPolicy: {policy!r}")


def permute_observation_objects(
    observation: Observation, policy: OrderPolicy, trial_index: int
) -> tuple[Observation, dict[str, Any]]:
    """Permutes ONLY `observation.objects` (spec section 8's first
    target). Real architectural finding this module documents (spec
    section 29's root-cause trace, `order_audit.md`): `enumerate_legal_actions`
    (src/llm_controller.py) iterates `observation.objects` directly to
    build MOVE targets, so this permutation necessarily cascades into the
    model-visible legal-actions list's order too - the two structures are
    NOT independently orderable in the current implementation. This is
    expected and correct for the object-order probe itself (spec section
    9's "keep fixed: ... legal actions" refers to legal-actions
    CARDINALITY/membership, never being ADDED or REMOVED - never their
    presentation order, which this function deliberately does change).
    Returns (new_observation, order_metadata) - the metadata records both
    the canonical and presented id order for per-decision telemetry (spec
    section 20)."""
    original = list(observation.objects)
    presented = _permuted(original, policy, trial_index)
    new_observation = dataclasses.replace(observation, objects=tuple(presented))
    metadata = {
        "structure_name": "objects",
        "policy": policy.value,
        "trial_index": trial_index,
        "canonical_order": [o.id for o in original],
        "presented_order": [o.id for o in presented],
    }
    return new_observation, metadata


def permute_observation_messages(
    observation: Observation, policy: OrderPolicy, trial_index: int
) -> tuple[Observation, dict[str, Any]]:
    """Permutes ONLY `observation.messages_received_recently` - a
    genuinely independent structure from `objects`/legal_actions (spec
    section 17's message-order probe): the payload builder serializes it
    as its own separate `communication.messages_received_recently` list
    (src/llm_controller.py), never derived from or feeding into
    `enumerate_legal_actions`. Manipulating this never touches objects or
    legal-actions order (spec section 16's "manipulate one ordered
    structure at a time")."""
    original = list(observation.messages_received_recently)
    presented = _permuted(original, policy, trial_index)
    new_observation = dataclasses.replace(observation, messages_received_recently=tuple(presented))
    metadata = {
        "structure_name": "messages_received_recently",
        "policy": policy.value,
        "trial_index": trial_index,
        "canonical_order": [m["message_id"] for m in original],
        "presented_order": [m["message_id"] for m in presented],
    }
    return new_observation, metadata


def semantically_equivalent(a: Observation, b: Observation) -> bool:
    """Spec section 31's required regression: true iff `a` and `b` carry
    the identical set of facts, differing (if at all) only in the ORDER
    of `objects`/`messages_received_recently`. Compares every other field
    for exact equality, and both permutable fields as sets/multisets
    (never as ordered sequences) - this is the one place order is
    deliberately ignored, everywhere else `Observation`'s own `==`
    equality (which IS order-sensitive for tuples) is too strict for this
    specific check."""
    a_fields = dataclasses.fields(a)
    b_fields = {f.name for f in dataclasses.fields(b)}
    if {f.name for f in a_fields} != b_fields:
        return False
    for f in a_fields:
        if f.name in ("objects", "messages_received_recently", "log_event"):
            continue
        if getattr(a, f.name) != getattr(b, f.name):
            return False
    if sorted(o.id for o in a.objects) != sorted(o.id for o in b.objects):
        return False
    if set(a.objects) != set(b.objects):
        return False
    a_msg_ids = sorted(m["message_id"] for m in a.messages_received_recently)
    b_msg_ids = sorted(m["message_id"] for m in b.messages_received_recently)
    return a_msg_ids == b_msg_ids
