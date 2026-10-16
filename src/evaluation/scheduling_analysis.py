"""M6G spec sections 23-24, 37-43: pure, deterministic analysis helpers.

Everything here works on the JSON rows `SchedulingResult.to_row()` writes -
hashes, decisions, apply outcomes and event logs only (spec section 50: no
LLM-as-a-Judge). The statistics are deliberately small and fixed before any
live data existed (recorded in call_plan.json): total-variation distance
between two decision distributions, and an exact-seed permutation p-value
for it. An effect counts as supported only when it is both material
(TVD >= MATERIAL_TVD) and unlikely under exchangeability (p < ALPHA).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any, Iterable

from src.evaluation.duel_snapshot import ACTORS, json_hash
from src.llm_controller import _normalize_action_name

MATERIAL_TVD = 0.3
ALPHA = 0.05
N_PERMUTATIONS = 20000
PERMUTATION_SEED = 6001

# spec section 42
NO_SCHEDULING_EFFECT = "NO_SCHEDULING_EFFECT"
MODEL_CALL_ORDER_EFFECT = "MODEL_CALL_ORDER_EFFECT"
APPLICATION_ORDER_EFFECT = "APPLICATION_ORDER_EFFECT"
SEQUENTIAL_INFORMATION_EFFECT = "SEQUENTIAL_INFORMATION_EFFECT"
APPLY_TIME_INVALIDATION_EFFECT = "APPLY_TIME_INVALIDATION_EFFECT"
ACTOR_ORDER_EFFECT = "ACTOR_ORDER_EFFECT"
MIXED = "MIXED"
UNINTERPRETABLE_REQUESTS_DIFFER = "UNINTERPRETABLE_REQUESTS_DIFFER"

# spec section 41, earliest first
LAYERS = (
    "snapshot",
    "agent_request",
    "raw_model_output",
    "normalized_action",
    "application_order",
    "apply_time_legality",
    "world_result",
)

# Shared-state effects whose owner is decided by who applies first.
CONTESTED_EVENT_TYPES = frozenset({"DOOR_UNLOCKED", "TRAP_PLACED", "TRAP_REMOVED", "TRAP_DISARMED"})


def decision_label(record: dict[str, Any]) -> str:
    d = record["decision"]
    if d["action"] == "wait":
        return "wait"
    if d["action"] == "place_trap":
        return f"place_trap({d['target_id']},{d['trap_type']})"
    return f"{d['action']}({d['target_id']})"


def raw_decision_fields(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    """The decision exactly as the model emitted it (case, aliases and
    all), without the free-text rationale - the raw_model_output layer."""
    raw = record.get("raw_model_output") or {}
    return raw.get("action"), raw.get("target_id"), raw.get("trap_type")


def normalized_raw_action(record: dict[str, Any]) -> str | None:
    action = (record.get("raw_model_output") or {}).get("action")
    return _normalize_action_name(action) if action is not None else None


def distribution(labels: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(labels).items(), key=lambda kv: (-kv[1], kv[0])))


def modal(labels: list[str]) -> str | None:
    if not labels:
        return None
    counts = Counter(labels)
    top = max(counts.values())
    return sorted(k for k, v in counts.items() if v == top)[0]


def tvd(xs: list[str], ys: list[str]) -> float:
    if not xs or not ys:
        return 0.0
    cx, cy = Counter(xs), Counter(ys)
    return 0.5 * sum(abs(cx[k] / len(xs) - cy[k] / len(ys)) for k in set(cx) | set(cy))


def permutation_p_value(xs: list[str], ys: list[str], n_permutations: int = N_PERMUTATIONS, seed: int = PERMUTATION_SEED) -> float:
    """Two-sample permutation test on TVD (the null: both samples are
    exchangeable draws from one distribution). Seeded, so reproducible."""
    if not xs or not ys:
        return 1.0
    observed = tvd(xs, ys)
    if observed == 0.0:
        return 1.0
    pooled = list(xs) + list(ys)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        if tvd(pooled[: len(xs)], pooled[len(xs):]) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_permutations + 1)


def compare_distributions(xs: list[str], ys: list[str]) -> dict[str, Any]:
    distance = tvd(xs, ys)
    p = permutation_p_value(xs, ys)
    return {
        "n_x": len(xs),
        "n_y": len(ys),
        "distribution_x": distribution(xs),
        "distribution_y": distribution(ys),
        "modal_x": modal(xs),
        "modal_y": modal(ys),
        "tvd": round(distance, 4),
        "p_value": round(p, 6),
        "material": distance >= MATERIAL_TVD,
        "significant": p < ALPHA,
        "supported": distance >= MATERIAL_TVD and p < ALPHA,
    }


def event_multiset_hash(row: dict[str, Any]) -> str:
    """Order-insensitive companion to `event_log_hash`: which effects
    happened, regardless of the order they were logged in. Two actions
    applied in either order always log in that order, so an order-only
    event-log difference is a restatement of the schedule, not an outcome
    difference."""
    return json_hash(sorted(json.dumps(e, sort_keys=True, default=str) for e in row["events"]))


def world_outcome(row: dict[str, Any]) -> tuple[str, str]:
    return row["world_state_hash_after"], event_multiset_hash(row)


def _layer_values(row: dict[str, Any], layer: str) -> Any:
    recs = row["records"]
    if layer == "snapshot":
        return tuple(recs[a]["world_state_hash_before_snapshot"] for a in ACTORS)
    if layer == "agent_request":
        return tuple((recs[a]["agent_request_semantic_hash"], recs[a]["provider_request_hash"]) for a in ACTORS)
    if layer == "raw_model_output":
        return tuple(raw_decision_fields(recs[a]) for a in ACTORS)
    if layer == "normalized_action":
        return tuple(tuple(sorted(recs[a]["decision"].items())) for a in ACTORS)
    if layer == "application_order":
        return tuple(recs[a]["apply_index"] for a in ACTORS)
    if layer == "apply_time_legality":
        return tuple((recs[a]["legal_at_apply_time"], recs[a]["apply_outcome"]) for a in ACTORS)
    if layer == "world_result":
        return world_outcome(row)
    raise ValueError(layer)


def differing_layers(row_x: dict[str, Any], row_y: dict[str, Any]) -> list[str]:
    return [layer for layer in LAYERS if _layer_values(row_x, layer) != _layer_values(row_y, layer)]


def first_difference(row_x: dict[str, Any], row_y: dict[str, Any]) -> str:
    """Spec section 41: the earliest layer at which two paired trials
    differ, or "none". Rationale text is excluded from the raw layer - it
    varies between otherwise identical decisions and is not a decision -
    and the world layer ignores pure event-log ordering (see
    `event_multiset_hash`)."""
    layers = differing_layers(row_x, row_y)
    return layers[0] if layers else "none"


def contested_effects(row: dict[str, Any]) -> list[tuple[str, str]]:
    """(event type, actor) for every shared-state effect in the trial whose
    owner depends on who applied first."""
    return [(e["type"], e.get("actor_id")) for e in row["events"] if e["type"] in CONTESTED_EVENT_TYPES]


def contested_label(row: dict[str, Any]) -> str:
    effects = contested_effects(row)
    return ", ".join(f"{t}:{a}" for t, a in effects) if effects else "none"


def intent_label(record: dict[str, Any]) -> str:
    """What the model asked for, before validation - descriptive only (the
    predeclared statistics use the executed decision). Separates a
    deliberate WAIT from a fallback WAIT."""
    raw = record.get("raw_model_output") or {}
    action = normalized_raw_action(record)
    if action is None:
        return "no_action"
    if action == "wait":
        label = "wait"
    else:
        label = f"{action}({raw.get('target_id')}" + (f",{raw.get('trap_type')}" if action == "place_trap" else "") + ")"
    return label if record.get("first_pass_valid") else f"{label}[invalid:{record.get('validation_error')}]"


def request_identity(rows: Iterable[dict[str, Any]], actor_id: str) -> dict[str, Any]:
    rows = list(rows)
    semantic = {r["records"][actor_id]["agent_request_semantic_hash"] for r in rows}
    provider = {r["records"][actor_id]["provider_request_hash"] for r in rows}
    full = {r["records"][actor_id]["agent_request_hash"] for r in rows}
    return {
        "identical": len(semantic) == 1 and len(provider) == 1 and len(full) == 1,
        "agent_request_hash": sorted(full),
        "agent_request_semantic_hash": sorted(semantic),
        "provider_request_hash": sorted(provider),
    }


def apply_time_invalidated(record: dict[str, Any]) -> bool:
    """A first-pass-valid decision that was legal at its snapshot but not
    valid when the Engine applied it (spec sections 17-18)."""
    return bool(record["first_pass_valid"] and record["legal_at_snapshot"] and not record["legal_at_apply_time"])


def classify_apply_order_pair(a_first: dict[str, Any], b_first: dict[str, Any]) -> list[str]:
    """Spec section 23: same pre-action state, same decisions, A->B vs B->A."""
    if a_first["world_state_hash_pre_action"] != b_first["world_state_hash_pre_action"]:
        raise ValueError("apply-order pair must share the pre-action state")
    for actor_id in ACTORS:
        if a_first["records"][actor_id]["decision"] != b_first["records"][actor_id]["decision"]:
            raise ValueError("apply-order pair must share the decisions")
    labels = []
    if world_outcome(a_first) != world_outcome(b_first):
        labels.append(APPLICATION_ORDER_EFFECT)
    if any(
        a_first["records"][a]["legal_at_apply_time"] != b_first["records"][a]["legal_at_apply_time"]
        and a_first["records"][a]["first_pass_valid"]
        for a in ACTORS
    ):
        labels.append(APPLY_TIME_INVALIDATION_EFFECT)
    return labels or [NO_SCHEDULING_EFFECT]


def classify_call_order(s0_rows: list[dict[str, Any]], s1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec sections 12, 42: a call-order effect is only interpretable when
    each actor's request is byte-identical across S0 and S1."""
    per_actor = {}
    identical = True
    for actor_id in ACTORS:
        ident = request_identity(s0_rows + s1_rows, actor_id)
        identical = identical and ident["identical"]
        comp = compare_distributions(
            [decision_label(r["records"][actor_id]) for r in s0_rows],
            [decision_label(r["records"][actor_id]) for r in s1_rows],
        )
        comp["request_identity"] = ident
        per_actor[actor_id] = comp
    if not identical:
        label = UNINTERPRETABLE_REQUESTS_DIFFER
    elif any(c["supported"] for c in per_actor.values()):
        label = MODEL_CALL_ORDER_EFFECT
    else:
        label = NO_SCHEDULING_EFFECT
    return {"requests_identical_across_conditions": identical, "per_actor": per_actor, "classification": label}


def classify_snapshot_timing(s0_rows: list[dict[str, Any]], seq_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec sections 9, 24, 28: compare the second actor of a sequential
    schedule with the same actor in the frozen baseline."""
    second = seq_rows[0]["second_actor"]
    first = seq_rows[0]["first_actor"]
    s0_second = [decision_label(r["records"][second]) for r in s0_rows]
    seq_second = [decision_label(r["records"][second]) for r in seq_rows]
    comp = compare_distributions(s0_second, seq_second)
    s0_modal = modal(s0_second)
    saw = [r["second_actor_saw_first_action"] for r in seq_rows]
    world_changed = [r["world_changed_before_second_snapshot"] for r in seq_rows]
    invalidated = [apply_time_invalidated(r["records"][second]) for r in seq_rows]
    labels = []
    if any(saw) and comp["supported"]:
        labels.append(SEQUENTIAL_INFORMATION_EFFECT)
    if any(invalidated):
        labels.append(APPLY_TIME_INVALIDATION_EFFECT)
    return {
        "first_actor": first,
        "second_actor": second,
        "n": len(seq_rows),
        "same_snapshot_rate": _rate(r["same_snapshot"] for r in seq_rows),
        "world_changed_before_second_snapshot_rate": _rate(world_changed),
        "second_actor_saw_first_action_rate": _rate(saw),
        "second_actor_decision_vs_s0": comp,
        "second_agent_action_change_rate": _rate(label != s0_modal for label in seq_second),
        "second_actor_apply_time_invalidation_rate": _rate(invalidated),
        "classification": labels or [NO_SCHEDULING_EFFECT],
    }


def classify_actor_order(s3_rows: list[dict[str, Any]], s4_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec section 6 (S3 vs S4) and 30: does swapping which identity goes
    first change the outcome, and is the effect positional (the same for
    whichever actor holds the first slot) or identity-specific?"""
    def by_role(rows: list[dict[str, Any]], role: str) -> list[str]:
        return [decision_label(r["records"][r[role]]) for r in rows]

    first_role = compare_distributions(by_role(s3_rows, "first_actor"), by_role(s4_rows, "first_actor"))
    second_role = compare_distributions(by_role(s3_rows, "second_actor"), by_role(s4_rows, "second_actor"))
    holders_s3 = {actor for r in s3_rows for _, actor in contested_effects(r)}
    holders_s4 = {actor for r in s4_rows for _, actor in contested_effects(r)}
    contested_follows_order = bool(holders_s3 or holders_s4) and holders_s3 != holders_s4
    positional_symmetry = not first_role["supported"] and not second_role["supported"]
    labels = [ACTOR_ORDER_EFFECT] if contested_follows_order else [NO_SCHEDULING_EFFECT]
    return {
        "first_role_decisions": first_role,
        "second_role_decisions": second_role,
        "contested_effects_s3": distribution(contested_label(r) for r in s3_rows),
        "contested_effects_s4": distribution(contested_label(r) for r in s4_rows),
        "same_starting_state": len({r["world_state_hash_pre_action"] for r in s3_rows + s4_rows}) == 1,
        "contested_effect_follows_actor_order": contested_follows_order,
        "positional_symmetry": positional_symmetry,
        "classification": labels,
    }


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return round(sum(1 for v in values if v) / len(values), 4) if values else 0.0


def rate(values: Iterable[bool]) -> float:
    return _rate(values)
