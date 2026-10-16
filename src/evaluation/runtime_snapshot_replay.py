"""M6A Stage 1: frozen-decision-replay harness (spec sections 23-31).

Isolates runtime effects from world-trajectory divergence by freezing a
fixed set of real `Observation` snapshots *before* any live comparison
(spec section 25 - never chosen based on which runtime performs better),
then running the exact same snapshot independently through both Native
(tool-like representation, spec section 8's preferred baseline) and
Strands, multiple times each.

Every snapshot is built by directly seeding engine state (actor position,
trap detection, inventory, inspected status) from a fresh
`GameState.new_game` call - the same deterministic "construct the exact
fact directly" technique `scenario_harness.py`/`social_influence_scenarios.py`
already established, not organic simulated play - so calling
`build_snapshot_set()` twice produces byte-identical Observations both
times.

Scope decision, documented rather than silently applied: `AgentMemory` is
left at its fresh/empty state for every snapshot (no public API exists to
deterministically restore an arbitrary prior memory state outside live
play - restoring one would be new, nontrivial machinery). Both runtimes
still receive identical `Observation`/legal-actions/personality/mission
context - the comparison this milestone cares about (same decision state,
different runtime) still holds; only "varied prior memory content across
snapshots" is out of scope for this pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src import constants
from src.agents.registry import create_agent_runtime
from src.items import load_items
from src.level import load_level
from src.llm_controller import LLMControllerConfig, enumerate_legal_actions
from src.observation import Observation, build_observation
from src.state import GameState

# Fixed across every snapshot - determinism, not experimental variation
# (mirrors scenario_harness.SCENARIO_SEED's own reasoning).
SNAPSHOT_SEED = 6001

SIMPLE_NAVIGATION = "SIMPLE_NAVIGATION"
MULTIPLE_LEGAL_TARGETS = "MULTIPLE_LEGAL_TARGETS"
LOCKED_OBJECT = "LOCKED_OBJECT"
DETECTED_TRAP = "DETECTED_TRAP"
TRAP_PLACEMENT_OPPORTUNITY = "TRAP_PLACEMENT_OPPORTUNITY"
OPPONENT_RECENTLY_SEEN = "OPPONENT_RECENTLY_SEEN"
MISSION_ITEM_AVAILABLE = "MISSION_ITEM_AVAILABLE"
EXIT_AVAILABLE = "EXIT_AVAILABLE"
SNAPSHOT_CATEGORIES = (
    SIMPLE_NAVIGATION, MULTIPLE_LEGAL_TARGETS, LOCKED_OBJECT, DETECTED_TRAP,
    TRAP_PLACEMENT_OPPORTUNITY, OPPONENT_RECENTLY_SEEN, MISSION_ITEM_AVAILABLE, EXIT_AVAILABLE,
)


@dataclass(frozen=True)
class DecisionSnapshot:
    snapshot_id: str
    category: str
    observation: Observation
    legal_actions: list[dict[str, Any]] = field(repr=False)


def _fresh_state(communication_enabled: bool = False) -> GameState:
    # Spec section 37/26: SOCIAL_OFF for the primary runtime benchmark.
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=SNAPSHOT_SEED, communication_enabled=communication_enabled)


def _place(state: GameState, actor_id: str, x: float, y: float) -> None:
    cell = state.nav_grid.nearest_walkable(x, y)
    if cell is not None:
        x, y = state.nav_grid.center_of(cell)
    actor = state.actors[actor_id]
    actor.x, actor.y = float(x), float(y)


def _place_near_object(state: GameState, actor_id: str, object_id: str) -> None:
    obj = state.level.get_object(object_id)
    actor = state.actors[actor_id]
    approach = state.nav_grid.nearest_approach_point(obj.collision_rect, constants.INTERACTION_DISTANCE, (actor.x, actor.y))
    if approach is None:
        _place(state, actor_id, *obj.position)
        return
    actor.x, actor.y = float(approach[0]), float(approach[1])


def _snapshot_from_state(snapshot_id: str, category: str, state: GameState, actor_id: str = constants.WHITE_SPY_ID) -> DecisionSnapshot:
    # Real bug found while first live-testing this harness: `_place`/
    # `_place_near_object` only set actor.x/y directly. `nearby_object_id`
    # (and therefore every INSPECT/INTERACT/PLACE_TRAP/DISARM legality
    # gate in `enumerate_legal_actions`) is driven by
    # `state.current_interaction_target`, which the engine only
    # recomputes inside `GameState.update()` - never from raw position -
    # so every snapshot silently offered MOVE/WAIT only, no matter how
    # close the actor was placed. `scenario_harness.py`'s own snapshot
    # helpers already call `state.update(0.0)` after placement for this
    # exact reason; doing it once here (after all category-specific setup)
    # covers every call site instead of relying on each one to remember it.
    state.update(0.0)
    observation = build_observation(state, actor_id)
    legal_actions = enumerate_legal_actions(observation)
    return DecisionSnapshot(snapshot_id=snapshot_id, category=category, observation=observation, legal_actions=legal_actions)


def build_snapshot_set() -> list[DecisionSnapshot]:
    """Spec section 26's 8 required categories, one snapshot each - frozen
    once, before any live comparison (spec section 25)."""
    snapshots: list[DecisionSnapshot] = []

    # 1. SIMPLE_NAVIGATION - open floor, no nearby interactable, only MOVE/WAIT
    # legal. Real bug found while fact-checking this snapshot after the
    # `state.update(0.0)` fix above: `player_spawn` itself sits within
    # `INTERACTION_DISTANCE` of `door_main_office_ambassador`, so placing
    # the actor exactly there makes INSPECT/INTERACT legal too, defeating
    # the category. Offsetting 50px off spawn (still open floor, confirmed
    # via `state._find_nearest_interactable` returning None there) is a
    # genuinely interactable-free position.
    state = _fresh_state()
    _place(state, constants.WHITE_SPY_ID, state.level.player_spawn[0], state.level.player_spawn[1] - 50)
    snapshots.append(_snapshot_from_state("snap01_simple_navigation", SIMPLE_NAVIGATION, state))

    # 2. MULTIPLE_LEGAL_TARGETS - centered in main_office, several searchable
    # objects within range simultaneously.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "desk_01")
    snapshots.append(_snapshot_from_state("snap02_multiple_legal_targets", MULTIPLE_LEGAL_TARGETS, state))

    # 3. LOCKED_OBJECT - near the ambassador_safe, known to require a code,
    # not yet satisfied.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "ambassador_safe")
    snapshots.append(_snapshot_from_state("snap03_locked_object", LOCKED_OBJECT, state))

    # 4. DETECTED_TRAP - a real environmental trap, already detected by
    # this actor (legitimate self-knowledge, not hidden state).
    state = _fresh_state()
    state.level.initial_traps = [{"id": "m6a_snap_trap", "target_id": "filing_cabinet_01", "trap_type": "electric", "owner": constants.ENVIRONMENT_OWNER_ID, "hidden": True}]
    state = GameState.new_game(state.level, load_items(constants.ITEMS_PATH), seed=SNAPSHOT_SEED, communication_enabled=False)
    _place_near_object(state, constants.WHITE_SPY_ID, "filing_cabinet_01")
    state._object_entry("filing_cabinet_01")["detected_by"][constants.WHITE_SPY_ID] = "electric"
    snapshots.append(_snapshot_from_state("snap04_detected_trap", DETECTED_TRAP, state))

    # 5. TRAP_PLACEMENT_OPPORTUNITY - actor holds a trap, is near an
    # inspected, currently-untrapped trappable object.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "ambassador_safe")
    state._object_entry("ambassador_safe")["inspected_by"].add(constants.WHITE_SPY_ID)
    snapshots.append(_snapshot_from_state("snap05_trap_placement_opportunity", TRAP_PLACEMENT_OPPORTUNITY, state))

    # 6. OPPONENT_RECENTLY_SEEN - opponent visible right now (same room,
    # in range) - the simplest, unambiguous "recently seen" state.
    state = _fresh_state()
    _place(state, constants.WHITE_SPY_ID, *state.level.player_spawn)
    _place(state, constants.BLACK_SPY_ID, state.level.player_spawn[0] + 20, state.level.player_spawn[1])
    snapshots.append(_snapshot_from_state("snap06_opponent_recently_seen", OPPONENT_RECENTLY_SEEN, state))

    # 7. MISSION_ITEM_AVAILABLE - near an object that this exact seed's own
    # real dependency placement (GameState.new_game -> generate_dependencies)
    # actually holds a mission item at - never fabricated.
    state = _fresh_state()
    item_locations = state.dependency_placement.item_locations if state.dependency_placement else {}
    target_obj = next(iter(item_locations.values()), "desk_01")
    _place_near_object(state, constants.WHITE_SPY_ID, target_obj)
    snapshots.append(_snapshot_from_state("snap07_mission_item_available", MISSION_ITEM_AVAILABLE, state))

    # 8. EXIT_AVAILABLE - near the exit door.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "exit_door")
    snapshots.append(_snapshot_from_state("snap08_exit_available", EXIT_AVAILABLE, state))

    return snapshots


def build_native_adapter(*, actor_id: str, model_client, telemetry, episode_id: str, provider_name: str, model_name: str, personality_profile: str | None):
    """Spec section 8: `native_tool_like` is the preferred baseline
    representation for this comparison (real tool-calling on the Strands
    side means the plain-JSON Native representation should at least use
    the flat, tool-shaped legal-action presentation, not force Strands
    down to Native's own default instead)."""
    return create_agent_runtime(
        "native_tool_like", actor_id=actor_id, model_client=model_client, telemetry=telemetry, episode_id=episode_id,
        provider_name=provider_name, model_name=model_name, execution_mode="evaluation",
        config=LLMControllerConfig(personality_profile=personality_profile),
    )


def build_strands_adapter(*, actor_id: str, telemetry, episode_id: str, provider_name: str, model_name: str, region: str | None, timeout: float, personality_profile: str | None):
    return create_agent_runtime(
        "strands", actor_id=actor_id, telemetry=telemetry, episode_id=episode_id,
        provider_name=provider_name, model_name=model_name, execution_mode="evaluation",
        llm_region=region, llm_timeout=timeout, personality_profile=personality_profile,
    )


# -- divergence classification (spec section 30) ---------------------------

SAME_DECISION = "SAME_DECISION"
SAME_ACTION_DIFFERENT_TARGET = "SAME_ACTION_DIFFERENT_TARGET"
DIFFERENT_ACTION = "DIFFERENT_ACTION"
VALIDITY_DIVERGENCE = "VALIDITY_DIVERGENCE"
REPAIR_DIVERGENCE = "REPAIR_DIVERGENCE"
RUNTIME_FAILURE_DIVERGENCE = "RUNTIME_FAILURE_DIVERGENCE"


def classify_divergence(native_trial: dict[str, Any], strands_trial: dict[str, Any]) -> str:
    """Deterministic, priority-ordered classification - never a blended
    "mostly the same" verdict (spec section 29: report observed agreement,
    never a synthetic overall quality score)."""
    if native_trial.get("runtime_error") or strands_trial.get("runtime_error"):
        return RUNTIME_FAILURE_DIVERGENCE
    if bool(native_trial.get("valid")) != bool(strands_trial.get("valid")):
        return VALIDITY_DIVERGENCE
    if bool(native_trial.get("repair_attempted")) != bool(strands_trial.get("repair_attempted")):
        return REPAIR_DIVERGENCE
    if native_trial.get("action") != strands_trial.get("action"):
        return DIFFERENT_ACTION
    if native_trial.get("target") != strands_trial.get("target"):
        return SAME_ACTION_DIFFERENT_TARGET
    return SAME_DECISION


def run_one_trial(adapter, observation: Observation, telemetry, episode_id: str, wall_timeout: float = 180.0, poll_interval: float = 0.05) -> dict[str, Any]:
    """One fresh strategic decision, normalized into the flat fields
    `classify_divergence`/the required trial table (spec section 53)
    need - runtime-agnostic, works for either adapter since both satisfy
    the same `AgentAdapter.decide()` contract.

    Real bug found and fixed while writing this milestone's own tests:
    both `NativeAgentAdapter` and `StrandsAgentAdapter` make their real
    decision on a non-blocking BACKGROUND thread - a single `decide()`
    call can return immediately with a stale/cached placeholder
    (`GameAction(WAIT)`) rather than the real result. `_run_one_decision`
    (`src/evaluation/scenario_harness.py`, used throughout the whole
    M5F.3A-D family) already established the fix: poll `decide()`
    repeatedly until `decisions_completed` actually increments. That
    same polling idiom is required here too - a single un-polled
    `decide()` call is not a valid way to exercise either runtime.

    Nothing here trusts any individual `decide()` call's OWN return
    value as the final result, even after polling - `_run_one_decision`'s
    own established pattern never does either. A second real bug found
    live while first writing this function: capturing "whichever
    `decide()` call happened to run right as `decisions_completed`
    incremented" is still unreliable (that specific call's return value
    can itself be a stale read racing the background thread's own
    write). The only reliable source of the actual completed decision is
    the `llm_decision` telemetry event itself (`executed_action`/`valid`/
    `action_source`), emitted once the decision genuinely finishes -
    `repair_attempted`/`latency_ms`/token counts live there too (never on
    `AgentDecision`, a deliberately thin metadata wrapper - see
    `src/agents/base.py`). The caller must pass a fresh per-trial
    `TelemetryLogger`."""
    start = time.monotonic()
    runtime_error = None
    completed_before = getattr(adapter, "decisions_completed", 0)
    try:
        while getattr(adapter, "decisions_completed", 0) <= completed_before:
            adapter.decide(observation)  # return value intentionally ignored - see docstring
            if time.monotonic() - start > wall_timeout:
                runtime_error = f"TimeoutError: decision did not complete within {wall_timeout}s"
                break
            time.sleep(poll_interval)
    except Exception as exc:  # pragma: no cover - defensive, recorded not swallowed
        runtime_error = f"{type(exc).__name__}: {exc}"

    decision_event = next(
        (e for e in reversed(telemetry.events) if e.get("event_type") == "llm_decision" and e.get("episode_id") == episode_id),
        {},
    )
    executed = decision_event.get("executed_action") or {}
    valid = bool(decision_event.get("final_model_action_valid", decision_event.get("valid")))
    action_source = "fallback" if decision_event.get("fallback_used") else "llm"

    return {
        "runtime": getattr(adapter, "runtime_name", "unknown"),
        "action": executed.get("action", "wait" if runtime_error else None),
        "target": executed.get("target_id"),
        "trap_type": executed.get("trap_type"),
        "valid": valid,
        "action_source": action_source,
        "decision_id": decision_event.get("decision_id"),
        "runtime_error": runtime_error,
        "repair_attempted": bool(decision_event.get("repair_attempted")),
        "fallback_used": bool(decision_event.get("fallback_used")),
        "latency_ms": decision_event.get("latency_ms"),
        "input_tokens": decision_event.get("input_tokens"),
        "output_tokens": decision_event.get("output_tokens"),
        "runtime_metadata": decision_event.get("runtime_metadata"),
        "representation_variant": decision_event.get("representation_variant"),
    }
