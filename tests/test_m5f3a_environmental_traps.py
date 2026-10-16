"""M5F.3A Phase C (spec section 38): environmental (initial, independent-
of-either-spy) trap engine tests. Phase A's audit (see
`environmental_trap_audit.md`) found trap runtime mechanics already treat
`owner_id` as an opaque string end-to-end (SUPPORTED_ALREADY at the
mechanics level), with exactly one real gap - `_trigger_trap`'s
alarm-specific `self.actors[owner_id]` lookup, which would KeyError for a
non-actor owner - and one missing piece: nothing created an environmental
trap at game start. This file exercises the `Level.initial_traps` /
`GameState.new_game` seeding path this milestone added to close that gap,
plus regression coverage for the alarm KeyError fix.
"""

from __future__ import annotations

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.scenario_harness import _place_near_object
from src.items import load_items
from src.level import load_level
from src.state import GameState

ENV_TRAP_SPEC = {
    "id": "security_terminal_trap",
    "target_id": "filing_cabinet_01",
    "trap_type": "electric",
    "owner": "environment",
    "hidden": True,
}


def _state_with_env_trap(trap_type: str = "electric") -> GameState:
    level = load_level(constants.LEVEL_PATH)
    level.initial_traps = [{**ENV_TRAP_SPEC, "trap_type": trap_type}]
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=4242)


def test_initial_trap_loads():
    state = _state_with_env_trap()
    trap = state.trap_on("filing_cabinet_01")
    assert trap is not None
    assert trap["type"] == "electric"


def test_owner_is_environment_and_distinct_from_both_spies():
    state = _state_with_env_trap()
    trap = state.trap_on("filing_cabinet_01")
    assert trap["owner_id"] == constants.ENVIRONMENT_OWNER_ID
    assert trap["owner_id"] != constants.BLACK_SPY_ID
    assert trap["owner_id"] != constants.WHITE_SPY_ID


def test_level_without_initial_traps_seeds_nothing():
    # data/level_01.json itself declares no initial_traps - normal
    # gameplay and every other regression test must be unaffected.
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=1)
    assert state.trap_on("filing_cabinet_01") is None


def test_no_spy_receives_placement_credit():
    from src.observation import build_observation

    state = _state_with_env_trap()
    black_obs = build_observation(state, constants.BLACK_SPY_ID)
    white_obs = build_observation(state, constants.WHITE_SPY_ID)
    black_cabinet = next(o for o in black_obs.objects if o.id == "filing_cabinet_01")
    white_cabinet = next(o for o in white_obs.objects if o.id == "filing_cabinet_01")
    assert black_cabinet.trap_is_mine is False
    assert white_cabinet.trap_is_mine is False
    # No TRAP_PLACED event fires for an environmentally-seeded trap - spec
    # section 4/11's "no actor receives placement credit" - a dedicated
    # ENVIRONMENTAL_TRAP_INITIALIZED event fires instead (spec section 32).
    assert not any(e["type"] == "TRAP_PLACED" for e in state.event_log)
    init_events = [e for e in state.event_log if e["type"] == "ENVIRONMENTAL_TRAP_INITIALIZED"]
    assert len(init_events) == 1
    assert init_events[0]["trap_owner"] == constants.ENVIRONMENT_OWNER_ID
    assert init_events[0]["target_id"] == "filing_cabinet_01"
    assert init_events[0]["trap_type"] == "electric"
    assert init_events[0]["initially_present"] is True


def test_hidden_trap_absent_from_unaware_observation():
    from src.observation import build_observation

    state = _state_with_env_trap()
    obs = build_observation(state, constants.WHITE_SPY_ID)
    cabinet = next(o for o in obs.objects if o.id == "filing_cabinet_01")
    assert cabinet.trap_status == "unknown"
    assert cabinet.trap_type is None


def test_inspect_may_detect_environmental_trap():
    state = _state_with_env_trap()
    actor = state.actors[constants.WHITE_SPY_ID]
    obj = state.level.get_object("filing_cabinet_01")
    # Force detection deterministically (INSPECT's probability roll is
    # real RNG - seed to a value known to detect first try isn't needed
    # since we just call the underlying method with a forced RNG below).
    state.rng.random = lambda: 0.0  # type: ignore[method-assign]
    result = state._inspect(actor, obj)
    assert result.trap_detected == "electric"
    assert state.detected_trap_type("filing_cabinet_01", constants.WHITE_SPY_ID) == "electric"


def test_detected_trap_visible_only_to_detecting_actor():
    from src.observation import build_observation

    state = _state_with_env_trap()
    state._object_entry("filing_cabinet_01")["detected_by"][constants.WHITE_SPY_ID] = "electric"

    white_obs = build_observation(state, constants.WHITE_SPY_ID)
    black_obs = build_observation(state, constants.BLACK_SPY_ID)
    white_cabinet = next(o for o in white_obs.objects if o.id == "filing_cabinet_01")
    black_cabinet = next(o for o in black_obs.objects if o.id == "filing_cabinet_01")
    assert white_cabinet.trap_status == "detected"
    assert white_cabinet.trap_type == "electric"
    assert black_cabinet.trap_status == "unknown"


def test_disarm_works_on_environmental_trap_after_detection():
    state = _state_with_env_trap()
    actor = state.actors[constants.WHITE_SPY_ID]
    obj = state.level.get_object("filing_cabinet_01")
    state._object_entry("filing_cabinet_01")["detected_by"][actor.id] = "electric"
    result = state._disarm(actor, obj)
    assert result.success is True
    assert state.trap_on("filing_cabinet_01") is None
    assert any(e["type"] == "TRAP_DISARMED" for e in state.event_log)


def test_disarm_refused_without_detection():
    state = _state_with_env_trap()
    actor = state.actors[constants.WHITE_SPY_ID]
    obj = state.level.get_object("filing_cabinet_01")
    result = state._disarm(actor, obj)
    assert result.success is False
    assert state.trap_on("filing_cabinet_01") is not None


def test_non_alarm_environmental_trap_triggers_and_stuns():
    state = _state_with_env_trap(trap_type="bomb")
    actor = state.actors[constants.WHITE_SPY_ID]
    obj = state.level.get_object("filing_cabinet_01")
    entry = state._object_entry("filing_cabinet_01")
    result = state._trigger_trap(actor, obj, entry, entry["trap"])
    assert result.trap_triggered == "bomb"
    assert actor.stunned_timer > 0
    assert state.trap_on("filing_cabinet_01") is None
    triggered_events = [e for e in state.event_log if e["type"] == "TRAP_TRIGGERED"]
    assert triggered_events[0]["trap_owner_id"] == constants.ENVIRONMENT_OWNER_ID


def test_alarm_environmental_trap_triggers_without_keyerror():
    # Real bug this milestone found and fixed: _trigger_trap's alarm
    # branch did `self.actors[owner_id].alarm_reveal_timer = ...`
    # unconditionally - "environment" is not a key in self.actors, so an
    # environmental alarm trap would have raised KeyError the first time
    # any actor triggered it. This is the regression test for that fix.
    state = _state_with_env_trap(trap_type="alarm")
    actor = state.actors[constants.WHITE_SPY_ID]
    obj = state.level.get_object("filing_cabinet_01")
    entry = state._object_entry("filing_cabinet_01")
    result = state._trigger_trap(actor, obj, entry, entry["trap"])  # must not raise
    assert result.trap_triggered == "alarm"
    assert actor.stunned_timer > 0
    # No ALARM_REVEALED event either - there is no owning spy to reward.
    assert not any(e["type"] == "ALARM_REVEALED" for e in state.event_log)


def test_actor_owned_alarm_trap_still_reveals_owner_unaffected_by_fix():
    # Confirms the guard (`owner_id in self.actors`) didn't quietly break
    # the ordinary spy-owned alarm-reveal behavior it was carved out of.
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=99)
    black = state.actors[constants.BLACK_SPY_ID]
    white = state.actors[constants.WHITE_SPY_ID]
    obj = state.level.get_object("filing_cabinet_01")
    entry = state._object_entry("filing_cabinet_01")
    entry["trap"] = {"type": "alarm", "owner_id": black.id, "armed": True}
    state._trigger_trap(white, obj, entry, entry["trap"])
    assert black.alarm_reveal_timer > 0
    assert any(e["type"] == "ALARM_REVEALED" and e["actor_id"] == black.id for e in state.event_log)


def test_spy_cannot_place_trap_on_already_trapped_environmental_object():
    # M5F.2.3's own trap-placement-validity fix must not regress: a spy
    # still can't legally place a second trap on an object that already
    # carries one, environmental or not (spec section 35).
    state = _state_with_env_trap()
    actor = state.actors[constants.BLACK_SPY_ID]
    obj = state.level.get_object("filing_cabinet_01")
    state._object_entry("filing_cabinet_01")["inspected_by"].add(actor.id)
    result = state._place_trap(actor, obj, "bomb")
    assert result.success is False
    assert state.trap_on("filing_cabinet_01")["owner_id"] == constants.ENVIRONMENT_OWNER_ID


def test_solvability_preserved_object_remains_searchable_after_trigger():
    # Spec section 34: an initial trap must not make the level
    # unsolvable. This engine's traps are single-use and never gate
    # access permanently (_trigger_trap replaces the search attempt with
    # a stun, but removes the trap - constants.py has no mechanic for a
    # trap that re-arms or blocks forever), so the trapped object remains
    # fully searchable on any later attempt regardless of which object a
    # given seed's dependency placement happens to put a mission item on.
    state = _state_with_env_trap(trap_type="bomb")
    actor = state.actors[constants.WHITE_SPY_ID]
    obj = state.level.get_object("filing_cabinet_01")
    entry = state._object_entry("filing_cabinet_01")
    state._trigger_trap(actor, obj, entry, entry["trap"])
    assert state.trap_on("filing_cabinet_01") is None
    actor.stunned_timer = 0.0  # stun already expired for this check
    result = state._search(actor, obj)
    assert result.success is True


def test_environmental_trap_via_apply_action_search_triggers_normally():
    # End-to-end through the real GameAction dispatch path (not the
    # internal method directly), confirming SEARCH-triggered environmental
    # traps behave identically to actor-owned ones.
    state = _state_with_env_trap(trap_type="snare")
    state.rng.random = lambda: 0.0  # type: ignore[method-assign] - force trap trigger, not detection
    actor = state.actors[constants.WHITE_SPY_ID]
    _place_near_object(state, constants.WHITE_SPY_ID, "filing_cabinet_01")
    state.update(0.0)  # populates current_interaction_target from the new position
    state.apply_action(constants.WHITE_SPY_ID, GameAction(ActionType.INTERACT, target_id="filing_cabinet_01"), dt=1 / 30)
    assert actor.stunned_timer > 0
    assert state.trap_on("filing_cabinet_01") is None
