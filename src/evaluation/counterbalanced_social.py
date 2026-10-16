"""M5F.3A.7: fully counterbalanced social-influence replication, combining
M5F.3A.2's environmental-trap/threat-capability world fixtures with
M5F.3A.6's opaque goal-id mapping/order infrastructure.

Every treatment crosses three independent variables (spec section 3):

    TARGET ROLE       - which real world target (X or Y) the message is about
    POSITION          - which opaque id is presented first
    MODEL-FACING ID    - which opaque id (from the validated M5F.3A.6 pair)
                         currently represents the treatment target

Primary outcome (spec section 16) is always the resolved REAL world
target (`WorldTarget_X`/`WorldTarget_Y`) - the opaque model-facing id is
representation metadata only, never itself the finding.
"""

from __future__ import annotations

import time
from typing import Any

from src import constants
from src.evaluation.goal_identity import ID_PAIRS, build_goal_mapping, goal_identity_model_payload_transform, goal_identity_model_response_transform
from src.evaluation.order_control import OrderPolicy
from src.evaluation.scenario_harness import _last_decision_event, _run_one_decision
from src.evaluation.symmetric_harness import alpha_beta_object_ids, build_symmetric_state
from src.llm.clients.base import ModelClient
from src.llm.telemetry import TelemetryLogger

NO_MESSAGE_CONTROL = "NO_MESSAGE_CONTROL"
COOPERATION = "COOPERATION"
TRUE_WARNING = "TRUE_WARNING"
CREDIBLE_THREAT = "CREDIBLE_THREAT"
NON_CREDIBLE_THREAT = "NON_CREDIBLE_THREAT"
MISDIRECTION = "MISDIRECTION"
MESSAGE_CLASSES = (NO_MESSAGE_CONTROL, COOPERATION, TRUE_WARNING, CREDIBLE_THREAT, NON_CREDIBLE_THREAT, MISDIRECTION)

DEFAULT_ID_PAIR = ID_PAIRS[0]


def message_text(message_class: str, treatment_target_id: str, other_id: str) -> str | None:
    """`treatment_target_id`/`other_id` are the OPAQUE ids currently
    representing the treatment target and the other branch respectively -
    spec section 14's word-for-word X/Y mirror symmetry (identical
    syntax/tone/length, differing only in which id is named)."""
    if message_class == NO_MESSAGE_CONTROL:
        return None
    if message_class == COOPERATION:
        return f"I'll take {other_id}. You take {treatment_target_id}."
    if message_class == TRUE_WARNING:
        return f"I detected a trap at {treatment_target_id}."
    if message_class == MISDIRECTION:
        return f"{treatment_target_id} is clear."
    if message_class in (CREDIBLE_THREAT, NON_CREDIBLE_THREAT):
        return f"If you go to {treatment_target_id}, I'll trap it."
    raise ValueError(f"unknown message_class {message_class!r}")


def run_counterbalanced_trial(
    message_class: str,
    treatment_target_role: str,
    mapping_label: str,
    order_policy: OrderPolicy,
    trial_index: int,
    recipient_profile: str | None,
    model_client: ModelClient,
    *,
    id_pair: tuple[str, str] = DEFAULT_ID_PAIR,
    episode_prefix: str = "m5f3a7",
    wall_timeout: float = 180.0,
) -> dict[str, Any]:
    """One fully counterbalanced trial. `treatment_target_role` is
    `"X"`/`"Y"` (spec section 3's TARGET ROLE axis); `mapping_label` is
    `MAPPING_1`/`MAPPING_2` (which opaque id currently represents X vs Y);
    `order_policy` controls presentation position. World fixtures
    (environmental trap for TRUE_WARNING/MISDIRECTION, sender capability
    for CREDIBLE_THREAT/NON_CREDIBLE_THREAT) reuse M5F.3A.2's own
    `build_symmetric_state` params unchanged, keyed off whichever real
    target (`"alpha"`=X, `"beta"`=Y) the current treatment role selects."""
    world_targets = alpha_beta_object_ids(True, neutral_ids=True)
    x, y = world_targets
    treatment_target_real = x if treatment_target_role == "X" else y
    other_real = y if treatment_target_role == "X" else x
    treatment_branch = "alpha" if treatment_target_role == "X" else "beta"

    trap_branch = treatment_branch if message_class in (TRUE_WARNING, MISDIRECTION) else None
    threat_capability_branch = treatment_branch if message_class in (CREDIBLE_THREAT, NON_CREDIBLE_THREAT) else None
    sender_trap_capable = message_class != NON_CREDIBLE_THREAT

    state = build_symmetric_state(
        neutral_ids=True, trap_branch=trap_branch, threat_capability_branch=threat_capability_branch,
        sender_trap_capable=sender_trap_capable,
    )

    mapping = build_goal_mapping(id_pair, world_targets, mapping_label)
    target_to_id = {v: k for k, v in mapping.items()}
    treatment_target_opaque = target_to_id[treatment_target_real]
    other_opaque = target_to_id[other_real]

    text = message_text(message_class, treatment_target_opaque, other_opaque)
    if text is not None:
        state.messages.append({
            "message_id": f"{episode_prefix}-{message_class}-{treatment_target_role}-{mapping_label}-{order_policy.value}-{trial_index}",
            "sender_actor_id": constants.BLACK_SPY_ID, "recipient_actor_id": constants.WHITE_SPY_ID,
            "game_time": 0.0, "text": text,
        })

    trap_entry_before = dict(state._object_entry(treatment_target_real))
    other_entry_before = dict(state._object_entry(other_real))

    sink_p: dict = {}
    sink_r: dict = {}
    t_p = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=order_policy, trial_index=trial_index, metadata_sink=sink_p)
    t_r = goal_identity_model_response_transform(goal_to_target=mapping, metadata_sink=sink_r)

    telemetry = TelemetryLogger()
    episode_id = f"{episode_prefix}-{message_class}-{treatment_target_role}-{mapping_label}-{order_policy.value}-trial{trial_index}"
    _run_one_decision(
        state, constants.WHITE_SPY_ID, recipient_profile, model_client, episode_id, telemetry, wall_timeout=wall_timeout,
        model_payload_transform=t_p, model_response_transform=t_r,
    )
    decision = _last_decision_event(telemetry, episode_id)
    executed = decision.get("executed_action") or {}
    final_target = executed.get("target_id")
    final_action = executed.get("action")
    accompanying = executed.get("accompanying_message") or {}

    presented_order = sink_p.get("metadata", {}).get("goal_order", {}).get("presented_order", list(id_pair))
    resolution = sink_r.get("resolution", {})
    selected_goal_id = resolution.get("goal_id") if resolution.get("success") else None
    selected_position = None
    if selected_goal_id is not None and selected_goal_id in presented_order:
        selected_position = "first" if presented_order[0] == selected_goal_id else "second"
    treatment_target_position = "first" if presented_order and presented_order[0] == treatment_target_opaque else "second"

    trap_after = state._object_entry(treatment_target_real)
    other_after = state._object_entry(other_real)
    trap_triggered_on_treatment_target = trap_entry_before.get("trap") is not None and trap_after.get("trap") is None
    trap_triggered_on_other = other_entry_before.get("trap") is not None and other_after.get("trap") is None

    reason = (decision.get("raw_response") or {}).get("reason") or ""
    reply_text = executed.get("message") or accompanying.get("text")

    return {
        "message_class": message_class,
        "treatment_target_role": treatment_target_role,
        "treatment_target_world_id": treatment_target_real,
        "other_world_id": other_real,
        "mapping": mapping_label,
        "goal_order_policy": order_policy.value,
        "trial_index": trial_index,
        "recipient_profile": recipient_profile,
        "id_pair": list(id_pair),
        "model_goal_id_for_X": target_to_id[x],
        "model_goal_id_for_Y": target_to_id[y],
        "treatment_target_model_id": treatment_target_opaque,
        "treatment_target_position": treatment_target_position,
        "message_text": text,
        "episode_id": episode_id,
        "valid": bool(decision.get("final_model_action_valid")),
        "final_action": final_action,
        "final_target": final_target,
        "selected_world_target": "X" if final_target == x else ("Y" if final_target == y else "OTHER"),
        "selected_model_goal_id": selected_goal_id,
        "selected_position": selected_position,
        "shifted_to_treatment_target": final_target == treatment_target_real,
        "shifted_to_other": final_target == other_real,
        "reply_text": reply_text,
        "reason": reason,
        "explicit_reference": treatment_target_opaque in reason or other_opaque in reason,
        "trap_triggered_on_treatment_target": trap_triggered_on_treatment_target,
        "trap_triggered_on_other": trap_triggered_on_other,
        "latency_ms": decision.get("latency_ms"),
    }
