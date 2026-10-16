"""M5F.3A.4: candidate-representation control for the symmetric strategic-
choice harness - DECOUPLED (independent order control per serialized
structure) and SINGLE_CANDIDATE_VIEW (one compact entry per candidate,
never repeated elsewhere) representations, plus CURRENT (the historical,
unmodified control).

Every transform here operates on `model_payload` - the DISPLAY structure
`compact_observation_for_model` builds, injected via
`LLMControllerConfig.model_payload_transform` (a new, optional,
default-`None` hook this milestone added to `src/llm_controller.py`) -
never on `observation_payload` (the separate, untouched structure real
legality validation is drawn from; see that hook's own docstring). This
is what satisfies spec section 10/29: model output still resolves through
the existing validator/ActionExecutor against the TRUE, unmodified
Observation - these transforms only ever change what the model SEES, not
what the engine allows.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from src.evaluation.order_control import OrderPolicy


class RepresentationMode(Enum):
    CURRENT = "CURRENT"
    DECOUPLED = "DECOUPLED"
    SINGLE_CANDIDATE_VIEW = "SINGLE_CANDIDATE_VIEW"


def _permute_list(items: list[Any], policy: OrderPolicy, trial_index: int) -> list[Any]:
    if policy is OrderPolicy.CANONICAL:
        return list(items)
    if policy is OrderPolicy.REVERSED:
        return list(reversed(items))
    if policy is OrderPolicy.COUNTERBALANCED:
        return list(items) if trial_index % 2 == 0 else list(reversed(items))
    raise ValueError(f"unknown OrderPolicy: {policy!r}")


def decoupled_model_payload_transform(
    *,
    legal_actions_policy: OrderPolicy,
    room_objects_policy: OrderPolicy,
    trial_index: int,
    candidate_ids: tuple[str, str],
    metadata_sink: dict[str, Any],
) -> Callable[[dict], dict]:
    """Returns a callable for `LLMControllerConfig.model_payload_transform`
    implementing DECOUPLED mode (spec sections 5-7). `legal_actions_policy`
    reorders ONLY the two candidate ids within
    `model_payload['legal_actions_compact']['move']` - non-candidate move
    targets (doors) keep their existing position, since spec section 28
    requires non-candidate context stay stable. `room_objects_policy`
    reorders `model_payload['room']['objects']` in full - for the
    symmetric harness specifically, this list never contains the
    candidates at all (see `serialization_audit.md`: they fail
    `room.objects`' own inclusion filter from the neutral start), so this
    axis concretely tests whether DOOR display order affects candidate
    selection, not candidate order itself. `metadata_sink` (a plain dict
    the caller owns) receives the exact presented order for telemetry
    (spec section 24) - the transform itself must return only the new
    payload, so metadata can't ride along in the return value."""

    def transform(model_payload: dict) -> dict:
        payload = dict(model_payload)

        legal_compact = dict(payload.get("legal_actions_compact", {}))
        move_targets = list(legal_compact.get("move", []))
        candidate_positions = [i for i, t in enumerate(move_targets) if t in candidate_ids]
        candidate_subset = [move_targets[i] for i in candidate_positions]
        reordered_candidates = _permute_list(candidate_subset, legal_actions_policy, trial_index)
        for pos, val in zip(candidate_positions, reordered_candidates):
            move_targets[pos] = val
        legal_compact["move"] = move_targets
        payload["legal_actions_compact"] = legal_compact

        room = dict(payload.get("room", {}))
        room_objects = list(room.get("objects", []))
        reordered_room_objects = _permute_list(room_objects, room_objects_policy, trial_index)
        room["objects"] = reordered_room_objects
        payload["room"] = room

        metadata_sink["metadata"] = {
            "representation_mode": RepresentationMode.DECOUPLED.value,
            "legal_actions_order": {"policy": legal_actions_policy.value, "candidate_order": reordered_candidates},
            "room_objects_order": {"policy": room_objects_policy.value, "presented_order": [o.get("id") for o in reordered_room_objects]},
        }
        return payload

    return transform


def single_candidate_model_payload_transform(
    *,
    candidate_ids: tuple[str, str],
    order_policy: OrderPolicy,
    trial_index: int,
    metadata_sink: dict[str, Any],
) -> Callable[[dict], dict]:
    """Returns a callable implementing SINGLE_CANDIDATE_VIEW (spec
    sections 8-11): removes `candidate_ids` from
    `legal_actions_compact['move']` and `room.objects` (spec section 9 -
    "do not repeat those candidate IDs in separate model-facing... for the
    same evaluation decision"), then adds a NEW `candidate_targets` list,
    one entry per candidate, each still carrying the SAME real object id
    the harness already uses as `target_id` for MOVE (spec section 11's
    "candidate identity must remain stable" - no separate id-mapping
    layer is needed, since this harness's own `neutral_ids=True` ids are
    already the opaque tokens spec section 11 asks for). Removing a
    candidate from the DISPLAY never removes it from what's actually
    legal - validation reads `observation_payload["legal_actions"]`
    (computed straight from the real, untouched Observation), a
    completely separate structure this function never touches (spec
    section 10's "no validation bypass")."""

    def transform(model_payload: dict) -> dict:
        payload = dict(model_payload)

        legal_compact = dict(payload.get("legal_actions_compact", {}))
        move_targets = [t for t in legal_compact.get("move", []) if t not in candidate_ids]
        legal_compact["move"] = move_targets
        payload["legal_actions_compact"] = legal_compact

        room = dict(payload.get("room", {}))
        room["objects"] = [o for o in room.get("objects", []) if o.get("id") not in candidate_ids]
        payload["room"] = room

        ordered_candidates = _permute_list(list(candidate_ids), order_policy, trial_index)
        payload["candidate_targets"] = [{"candidate_id": cid, "legal_action": "move"} for cid in ordered_candidates]

        metadata_sink["metadata"] = {
            "representation_mode": RepresentationMode.SINGLE_CANDIDATE_VIEW.value,
            "candidate_order": {"policy": order_policy.value, "presented_order": ordered_candidates},
        }
        return payload

    return transform


def count_candidate_occurrences(model_payload: dict, candidate_id: str) -> int:
    """Spec section 27's prompt-repetition count - descriptive evidence
    only, never treated as the sole cause of any observed bias. Counts
    `candidate_id` in `legal_actions_compact['move']`, `room.objects`
    (by id), and `candidate_targets` (by id) - the three places a
    candidate could legitimately appear across CURRENT/DECOUPLED/
    SINGLE_CANDIDATE_VIEW."""
    count = 0
    count += model_payload.get("legal_actions_compact", {}).get("move", []).count(candidate_id)
    count += sum(1 for o in model_payload.get("room", {}).get("objects", []) if o.get("id") == candidate_id)
    count += sum(1 for c in model_payload.get("candidate_targets", []) if c.get("candidate_id") == candidate_id)
    return count
