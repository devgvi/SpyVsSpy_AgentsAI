"""M5F.3A.6: goal-identity / world-target / position cross-mapping
isolation. M5F.3A.5 found a 92.5% strategic-choice skew with ZERO
selectable door candidates - rejecting execution-level representation as
the cause. This module isolates the three remaining candidate
explanations (goal-id/token identity, world-target identity, residual
position) by swapping which opaque model-facing id maps to which real
world target, independently of presentation order.

Unlike every prior M5F.3A.x milestone (where the model-facing candidate
id WAS literally the real object's own neutral hex-token id - no
remapping needed mechanically), this milestone's ids are genuinely
opaque and carry NO inherent connection to any real object
(`ID_PAIRS` below, never derived from or equal to any real `GameObject`
id). This requires a real mapping layer: the model's own `target_id`
response is a bare opaque token (e.g. "k7f3") that is not a legal engine
id at all until resolved through `goal_identity_model_response_transform`
(built on M5F.3A.6's new `LLMControllerConfig.model_response_transform`
hook) into the real object id THIS TRIAL's mapping currently assigns.
Validation always re-derives legality from the true, untouched
Observation - never weakened by this rewrite (same invariant every prior
milestone's own transforms preserved).
"""

from __future__ import annotations

from typing import Any, Callable

from src.evaluation.order_control import OrderPolicy
from src.evaluation.representation_control import _permute_list

# Spec section 4: at least 3 matched opaque pairs, similar length/pattern,
# non-semantic, non-ordinal, never alpha/beta/left/right/numeric-only.
# Frozen before any live execution.
ID_PAIRS: tuple[tuple[str, str], ...] = (
    ("k7f3", "m9c2"),
    ("r4d8", "v2h6"),
    ("p8k1", "t5n7"),
)

MAPPING_1 = "MAPPING_1"  # id_pair[0] -> world_target X, id_pair[1] -> world_target Y
MAPPING_2 = "MAPPING_2"  # id_pair[0] -> world_target Y, id_pair[1] -> world_target X


def token_shape_audit(id_pairs: tuple[tuple[str, str], ...] = ID_PAIRS) -> dict[str, Any]:
    """Spec section 16: record id length/character-class/prefix-suffix
    similarity BEFORE live execution - descriptive only, never a
    tokenizer reverse-engineering exercise."""
    rows = []
    for pair in id_pairs:
        for gid in pair:
            rows.append({
                "id": gid,
                "length": len(gid),
                "has_digit": any(c.isdigit() for c in gid),
                "has_alpha": any(c.isalpha() for c in gid),
                "starts_with_digit": gid[0].isdigit() if gid else False,
            })
    lengths = {r["length"] for r in rows}
    return {"ids": rows, "all_same_length": len(lengths) == 1, "length": lengths.pop() if len(lengths) == 1 else sorted(lengths)}


def build_goal_mapping(id_pair: tuple[str, str], world_targets: tuple[str, str], mapping: str) -> dict[str, str]:
    """`mapping` is `MAPPING_1` (id_pair[0]->world_targets[0]) or
    `MAPPING_2` (swapped) - spec section 2's exact Mapping 1/Mapping 2
    definition."""
    id_a, id_b = id_pair
    x, y = world_targets
    if mapping == MAPPING_1:
        return {id_a: x, id_b: y}
    if mapping == MAPPING_2:
        return {id_a: y, id_b: x}
    raise ValueError(f"unknown mapping: {mapping!r}")


def goal_identity_model_payload_transform(
    *,
    id_pair: tuple[str, str],
    order_policy: OrderPolicy,
    trial_index: int,
    metadata_sink: dict[str, Any],
) -> Callable[[dict], dict]:
    """Builds the DISPLAY: `legal_actions_compact['move']` and
    `candidate_goals` both show ONLY the two opaque ids from `id_pair` -
    never a real object id, never a door (spec section 9's "strategic-
    goal abstraction from M5F.3A.5 remains intact"). Both structures show
    the SAME ids in the SAME order, giving the model one consistent
    picture. The world-target mapping is never reflected in the display
    at all - it only matters at response-resolution time
    (`goal_identity_model_response_transform`), so the model has no way
    to infer which real target an id currently represents."""

    def transform(model_payload: dict) -> dict:
        payload = dict(model_payload)
        ordered_ids = _permute_list(list(id_pair), order_policy, trial_index)

        legal_compact = dict(payload.get("legal_actions_compact", {}))
        legal_compact["move"] = list(ordered_ids)
        payload["legal_actions_compact"] = legal_compact
        payload["candidate_goals"] = [{"goal_id": gid} for gid in ordered_ids]

        metadata_sink["metadata"] = {
            "id_pair": list(id_pair),
            "goal_order": {"policy": order_policy.value, "presented_order": ordered_ids},
        }
        return payload

    return transform


def goal_identity_model_response_transform(
    *,
    goal_to_target: dict[str, str],
    metadata_sink: dict[str, Any],
) -> Callable[[dict], dict]:
    """Rewrites `raw_response['target_id']` from whichever opaque goal id
    the model selected into the real object id `goal_to_target` currently
    assigns it - the one place this milestone's mapping actually takes
    effect. The ORIGINAL opaque id is preserved under
    `raw_response['_original_goal_id']` (spec section 19's
    `selected_goal_id` telemetry requirement - lost otherwise, since
    telemetry captures this same dict after this rewrite). An unresolvable
    goal id (not in `goal_to_target`) is left untouched - it will then
    fail the EXISTING validator's own legality check exactly like any
    other illegal target_id (spec section 11's "explicit resolution
    failure", never a silently invented replacement)."""

    def transform(raw_response: dict) -> dict:
        response = dict(raw_response)
        original_goal_id = response.get("target_id")
        resolved_target = goal_to_target.get(original_goal_id) if original_goal_id is not None else None
        if resolved_target is not None:
            response["target_id"] = resolved_target
            response["_original_goal_id"] = original_goal_id
            metadata_sink["resolution"] = {"goal_id": original_goal_id, "resolved_target": resolved_target, "success": True}
        else:
            metadata_sink["resolution"] = {"goal_id": original_goal_id, "resolved_target": None, "success": False}
        return response

    return transform
