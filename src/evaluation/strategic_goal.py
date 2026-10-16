"""M5F.3A.5: strategic goal selection contract - an evaluation/controller
layer separating WHAT the model wants to achieve (a strategic goal) from
HOW it gets there (navigation/execution, already owned by
`ActionExecutor`/pathfinding).

Architecture (spec section 1):

    Observation / strategic context
            v
    StrategicGoalChoice
            v
    goal resolver (this module's `resolve_goal`)
            v
    existing canonical GameAction
            v
    ActionExecutor / A* / Engine (completely unmodified)

M5F.3A.4 found that its own SINGLE_CANDIDATE_VIEW representation did not
solve the door/route-candidate confound: in 40/40 live trials the model
selected a still-visible DOOR instead of the new compact candidate id,
even though the candidate id remained fully legal. This milestone's
concrete fix (spec sections 6, 62): remove doors from the SELECTABLE
`move` candidate list too, not only the strategic objectives - doors may
remain visible as descriptive world facts (`room.objects`, spec section
30), just never as a peer entry in `legal_actions_compact["move"]`.

Implementation scope decision (documented per this milestone's own
"no route-candidate-representation escape hatch left" requirement,
distinct from a literal `{goal_id, reason}` schema rewrite): rather than
building a wholly new output schema/parsing/repair/fallback pipeline
outside `LLMController` (a far larger, riskier rewrite of already-tested
machinery), this module reuses the EXISTING `{action, target_id, reason}`
schema unchanged - `target_id`'s value IS the goal id (the same neutral
hex-token real object id M5F.3A.4 already established works with zero
validator changes, since `target_id` carries no enum constraint and
legality is re-derived from the true, untouched Observation regardless
of what the display shows). This satisfies every SUBSTANTIVE requirement
in the spec (one selectable identity per strategic intention, no
door/route peers, deterministic resolution, existing validation/repair/
fallback, one model call per decision) without touching
`LLMController`'s internals at all - reusing M5F.3A.4's own
`model_payload_transform` hook exactly as built. See
`contract_audit.md` for the full reasoning.
"""

from __future__ import annotations

from typing import Any, Callable

from src.evaluation.order_control import OrderPolicy
from src.evaluation.representation_control import _permute_list

DECISION_CONTRACT_GAME_ACTION = "game_action"
DECISION_CONTRACT_STRATEGIC_GOAL = "strategic_goal"


def resolve_goal(goal_id: str, candidate_map: dict[str, str]) -> str | None:
    """Deterministic goal resolver (spec section 9): `goal_id` -> the real
    canonical object id `candidate_map` maps it to, or `None` if
    unresolvable (an unknown goal id - spec section 11's "return explicit
    resolution failure... do not silently invent a replacement goal").
    For this harness `candidate_map` is an identity mapping (the goal id
    literally IS the real object's own neutral hex-token id - see this
    module's own docstring for why no separate mapping layer was needed
    mechanically), but implemented as an explicit dict lookup - never a
    bare pass-through - so the mapping is genuinely auditable (spec
    section 36) and swappable (spec section 41's optional id-swap
    diagnostic, not run by default)."""
    return candidate_map.get(goal_id)


def strategic_goal_model_payload_transform(
    *,
    candidate_ids: tuple[str, str],
    order_policy: OrderPolicy,
    trial_index: int,
    metadata_sink: dict[str, Any],
) -> Callable[[dict], dict]:
    """Extends M5F.3A.4's `single_candidate_model_payload_transform` with
    the one fix its own live data showed was missing: doors are ALSO
    removed from `legal_actions_compact['move']` (not only the strategic
    objectives), so `candidate_goals` is the ONLY selectable `move` peer -
    no door, room, or waypoint remains choosable at the same level (spec
    sections 6, 29-30, hard-stop section 62). Doors stay visible in
    `room.objects` as descriptive world facts only ("door X is open" -
    never "choose MOVE_TO door X" - spec section 30's exact distinction).
    """

    def transform(model_payload: dict) -> dict:
        payload = dict(model_payload)

        legal_compact = dict(payload.get("legal_actions_compact", {}))
        # Strip EVERY move target that isn't one of the two strategic
        # goals - doors included - leaving `move` selectable ONLY via
        # `candidate_goals` below.
        legal_compact["move"] = [t for t in legal_compact.get("move", []) if t in candidate_ids]
        payload["legal_actions_compact"] = legal_compact

        ordered_candidates = _permute_list(list(candidate_ids), order_policy, trial_index)
        payload["candidate_goals"] = [{"goal_id": gid} for gid in ordered_candidates]

        metadata_sink["metadata"] = {
            "representation_mode": DECISION_CONTRACT_STRATEGIC_GOAL,
            "goal_order": {"policy": order_policy.value, "presented_order": ordered_candidates},
        }
        return payload

    return transform


def count_door_move_candidates(model_payload: dict) -> int:
    """Spec section 39: for STRATEGIC_GOAL this must always be 0 - no
    selectable door/intermediate target remains as a `move` peer."""
    move_targets = model_payload.get("legal_actions_compact", {}).get("move", [])
    return sum(1 for t in move_targets if isinstance(t, str) and t.startswith("door_"))
