"""M5E: active deterministic opponent (spec sections 42-46). No live LLM
required anywhere in this file - deterministic fixtures and a real
GameState for the tests that need genuine pathfinding/engine interaction.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src import constants
from src.actions import ActionType, GameAction
from src.bot_controller import (
    OPPORTUNIST,
    RACER,
    REASON_DISARM_DETECTED_TRAP,
    REASON_INSPECT_NEAR_TARGET,
    REASON_MISSION_READY_EXIT,
    REASON_MOVE_TO_EXPLORATION_TARGET,
    REASON_MOVE_TO_MISSION_TARGET,
    REASON_PLACE_TRAP_OPPORTUNITY,
    REASON_SEARCH_NEAR_TARGET,
    REASON_WAIT_STUNNED_OR_COMPLETE,
    ScriptedBotController,
    infer_reason_code,
)
from src.items import load_items
from src.level import load_level
from src.llm.telemetry import TelemetryLogger
from src.observation import Observation, ObservedObject, build_observation
from src.state import GameState

WHITE = constants.WHITE_SPY_ID
BLACK = constants.BLACK_SPY_ID


class _FakeNavGrid:
    """Minimal NavGrid stand-in (see tests/test_action_lifecycle.py) - MOVE-
    target-selection tests care about *which* target gets chosen and that
    a real GameAction results, not genuine pathfinding geometry."""

    def nearest_approach_point(self, rect, max_distance, from_xy):
        return (from_xy[0] + 80.0, from_xy[1])

    def find_path(self, start_xy, goal_xy):
        return [goal_xy]


def _obj(id, **overrides):
    base = dict(
        id=id, type="desk", label=id, position=(10.0, 10.0), collision_size=(20.0, 20.0),
        interactable=True, trappable=True, searched=False, inspected=False, trap_status="unknown",
        trap_type=None, trap_is_mine=False, door_open=None, lock_type=None, locked=False, can_unlock=False,
    )
    base.update(overrides)
    return ObservedObject(**base)


def _make_observation(**overrides):
    base = dict(
        actor_id=BLACK, seed=1, position=(5.0, 5.0), facing="east", stunned=False, stunned_seconds=0.0,
        busy=False, busy_seconds=0.0, mission_time_remaining=100.0, mission_complete=False,
        inventory=frozenset(), knowledge=frozenset(), required_items=("passport",), trap_inventory={"bomb": 1},
        nearby_object_id=None, items_found_here=frozenset(), objects=(), nav_grid=_FakeNavGrid(),
        log_event=lambda *a, **k: None, current_room="office", known_rooms=frozenset({"office"}),
        known_doors=frozenset(), known_locked_objects={}, known_item_locations={}, visible_other_actor=False,
        opponent=None, last_seen_opponent=None, revealed_opponent=None, revealed_opponent_seconds=0.0,
    )
    base.update(overrides)
    return Observation(**base)


def _new_state(seed: int) -> GameState:
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=seed)


# ---------------------------------------------------------------------------
# Opponent policy tests (spec section 42)
# ---------------------------------------------------------------------------


def test_mission_ready_moves_toward_exit():
    exit_door = _obj("exit_door", type="door", position=(9000.0, 9000.0))
    observation = _make_observation(objects=(exit_door,), inventory=frozenset({"passport"}), nearby_object_id=None)
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
    action = opponent.get_action(observation)
    assert infer_reason_code(observation, action) == REASON_MOVE_TO_MISSION_TARGET


def test_mission_ready_and_near_exit_attempts_it():
    exit_door = _obj("exit_door", type="door")
    observation = _make_observation(objects=(exit_door,), inventory=frozenset({"passport"}), nearby_object_id="exit_door")
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
    action = opponent.get_action(observation)
    assert action.type == ActionType.INTERACT
    assert infer_reason_code(observation, action) == REASON_MISSION_READY_EXIT


def test_detected_enemy_trap_takes_priority_disarm():
    trapped = _obj("desk_01", trap_status="detected", trap_type="bomb", trap_is_mine=False, inspected=True)
    observation = _make_observation(objects=(trapped,), nearby_object_id="desk_01")
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
    action = opponent.get_action(observation)
    assert action.type == ActionType.DISARM
    assert infer_reason_code(observation, action) == REASON_DISARM_DETECTED_TRAP


def test_near_uninspected_object_inspects_first():
    desk = _obj("desk_01", inspected=False, searched=False)
    observation = _make_observation(objects=(desk,), nearby_object_id="desk_01")
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)  # inspect_before_search=True
    action = opponent.get_action(observation)
    assert action.type == ActionType.INSPECT
    assert infer_reason_code(observation, action) == REASON_INSPECT_NEAR_TARGET


def test_near_inspected_unsearched_object_searches():
    desk = _obj("desk_01", inspected=True, searched=False, trap_status="clear_after_inspection")
    observation = _make_observation(objects=(desk,), nearby_object_id="desk_01", trap_inventory={})
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
    action = opponent.get_action(observation)
    assert action.type == ActionType.INTERACT
    assert infer_reason_code(observation, action) == REASON_SEARCH_NEAR_TARGET


def test_racer_profile_skips_inspection_and_searches_directly():
    desk = _obj("desk_01", inspected=False, searched=False)
    observation = _make_observation(objects=(desk,), nearby_object_id="desk_01")
    opponent = ScriptedBotController(BLACK, profile=RACER)  # inspect_before_search=False
    action = opponent.get_action(observation)
    assert action.type == ActionType.INTERACT
    assert infer_reason_code(observation, action) == REASON_SEARCH_NEAR_TARGET


def test_legal_trap_opportunity_places_trap():
    # Object just searched, believed clear, handed this actor an item
    # (items_found_here) and sits near the exit - two relevance signals
    # per _trap_value, enough to clear the placement-cost bar; actor
    # holds trap inventory.
    exit_door = _obj("exit_door", type="door", position=(0.0, 0.0))
    desk = _obj("desk_01", inspected=True, searched=True, trap_status="clear_after_inspection", position=(10.0, 10.0))
    observation = _make_observation(
        objects=(desk, exit_door), nearby_object_id="desk_01", trap_inventory={"bomb": 3}, position=(10.0, 10.0),
        items_found_here=frozenset({"desk_01"}),
    )
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
    action = opponent.get_action(observation)
    assert action.type == ActionType.PLACE_TRAP
    assert infer_reason_code(observation, action) == REASON_PLACE_TRAP_OPPORTUNITY


def test_otherwise_deterministic_move_target_toward_nearest_unsearched():
    state = _new_state(seed=1301)
    observation = build_observation(state, BLACK)
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
    action = opponent.get_action(observation)
    assert action.type == ActionType.MOVE
    assert infer_reason_code(observation, action) == REASON_MOVE_TO_EXPLORATION_TARGET


def test_stunned_or_complete_waits_deterministically():
    observation = _make_observation(objects=(), stunned=True)
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
    action = opponent.get_action(observation)
    assert action.type == ActionType.WAIT
    assert infer_reason_code(observation, action) == REASON_WAIT_STUNNED_OR_COMPLETE


def test_stable_target_tie_break_is_deterministic_across_runs():
    """Two objects equidistant from the actor - target selection must pick
    the same one every time for the same input (spec section 5)."""
    a = _obj("desk_a", position=(20.0, 0.0))
    b = _obj("desk_b", position=(-20.0, 0.0))
    observation = _make_observation(objects=(a, b), position=(0.0, 0.0))

    picks = set()
    for _ in range(5):
        opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
        target = opponent._select_target(observation)
        picks.add(target.id if target else None)
    assert len(picks) == 1  # always the same choice, never flip-flopping


# ---------------------------------------------------------------------------
# Information boundary (spec section 43)
# ---------------------------------------------------------------------------


def test_opponent_cannot_see_undetected_enemy_trap_type_or_owner():
    # trap_status stays "unknown"/"clear_after_inspection" for anything not
    # legitimately detected - ObservedObject never carries owner/type
    # unless trap_status == "detected" (observation.py's own boundary).
    hidden_trap_desk = _obj("desk_01", trap_status="unknown", inspected=False)
    observation = _make_observation(objects=(hidden_trap_desk,), nearby_object_id="desk_01")
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
    action = opponent.get_action(observation)
    # Inspects rather than disarming something it cannot legitimately know exists.
    assert action.type == ActionType.INSPECT


def test_opponent_has_no_private_opponent_memory_or_hidden_item_access():
    state = _new_state(seed=1302)
    observation = build_observation(state, BLACK)
    # The only channel to any state at all is this Observation - assert it
    # carries no reference to the raw GameState/level/items objects.
    assert not hasattr(observation, "level")
    assert not hasattr(observation, "items")
    assert not hasattr(observation, "generator")
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
    action = opponent.get_action(observation)
    assert isinstance(action, GameAction)


def test_opponent_observation_hides_agent_when_not_visible():
    state = _new_state(seed=1303)
    # Both spies spawn in main_office by level design - move White away to
    # a different room to exercise the "not currently visible" boundary.
    white = state.actors[WHITE]
    white.x, white.y = 700.0, 500.0  # security_room, per level_01.json's room rects
    state.update(0.0)
    observation = build_observation(state, BLACK)
    assert state.level.room_id_at(white.x, white.y) != state.level.room_id_at(state.actors[BLACK].x, state.actors[BLACK].y)
    assert observation.visible_other_actor is False
    assert observation.opponent is None


# ---------------------------------------------------------------------------
# Shared time / fairness (spec sections 44, mandatory)
# ---------------------------------------------------------------------------


class _FakeModelClient:
    def __init__(self, response, delay=0.0):
        self.response = response
        self.delay = delay

    def generate_action(self, *args, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        return self.response


def _run_fairness_scenario(delay: float, ticks: int = 40) -> tuple[float, float]:
    from src.game import Game

    game = Game(
        seed=1401, white_controller="llm", llm_provider="mock", llm_model="mock-model",
        llm_execution_mode="evaluation", black_controller="scripted",
    )
    game.bot_controller.controller.model_client = _FakeModelClient({"action": "wait"}, delay=delay)
    black_start = (game.state.actors[BLACK].x, game.state.actors[BLACK].y)
    mission_time_before = game.state.mission_time_remaining

    for _ in range(ticks):
        game._step(1 / 30)

    black_end = (game.state.actors[BLACK].x, game.state.actors[BLACK].y)
    moved = ((black_end[0] - black_start[0]) ** 2 + (black_end[1] - black_start[1]) ** 2) ** 0.5
    return moved, mission_time_before - game.state.mission_time_remaining


def test_slow_inference_does_not_give_opponent_free_progress():
    """Mandatory fairness regression (spec sections 9, 25, 44): a slow fake
    model must not let the scripted opponent move/progress further than a
    fast one over the same number of ticks - both are governed by the same
    shared effective_dt, frozen while White's LLM is thinking."""
    fast_moved, fast_time_used = _run_fairness_scenario(delay=0.0)
    slow_moved, slow_time_used = _run_fairness_scenario(delay=0.3)

    # The slow model keeps White "thinking" (effective_dt=0) for far more
    # of these 40 ticks than the fast one - if Black were on its own clock,
    # it would show *more* movement/time-used under the slow condition.
    assert slow_moved <= fast_moved + 1e-6
    assert slow_time_used <= fast_time_used + 1e-6


def test_opponent_progresses_during_evaluated_agents_multi_tick_move():
    """Once White has an accepted multi-tick MOVE in flight, Black must
    keep progressing on the same simulation ticks (spec section 10) - not
    frozen merely because White is mid-execution (thinking_in_progress is
    False once a decision is accepted, so effective_dt is normal dt)."""
    from src.game import Game

    game = Game(
        seed=1402, white_controller="llm", llm_provider="mock", llm_model="mock-model",
        llm_execution_mode="evaluation", black_controller="scripted",
    )
    game.bot_controller.controller.model_client = _FakeModelClient({"action": "move", "target_id": "door_reception_corridor"})
    black_start = (game.state.actors[BLACK].x, game.state.actors[BLACK].y)

    for _ in range(150):
        game._step(1 / 30)

    black_end = (game.state.actors[BLACK].x, game.state.actors[BLACK].y)
    moved = ((black_end[0] - black_start[0]) ** 2 + (black_end[1] - black_start[1]) ** 2) ** 0.5
    assert moved > 5.0  # Black genuinely moved while White's own MOVE was executing


# ---------------------------------------------------------------------------
# Trap competition (spec section 45)
# ---------------------------------------------------------------------------


def test_opponent_placed_trap_triggers_on_evaluated_actor_interaction():
    """PLACE_TRAP/INTERACT/DISARM never carry an explicit target_id from
    either ScriptedBotController or (matching it here) this test - both
    rely on the engine's own `current_interaction_target` resolution
    (target_id=None means "whatever's nearest", preserved for exactly
    this reason by the M5D.4 engine fix - see GameState._target_object)."""
    state = _new_state(seed=1304)
    desk = state.level.get_object("desk_01")
    black = state.actors[BLACK]
    black.x, black.y = desk.position[0], desk.position[1]
    black.trap_inventory["bomb"] = 1
    state.update(0.0)
    black_obs = build_observation(state, BLACK)
    assert black_obs.nearby_object_id == "desk_01"

    state.apply_action(BLACK, GameAction(ActionType.INSPECT), 1 / 30)  # PLACE_TRAP requires prior inspection (M5D.4)
    black.action_cooldown = 0.0  # clear INSPECT's own cooldown so the next action isn't dropped as "busy"
    state.apply_action(BLACK, GameAction(ActionType.PLACE_TRAP, trap_type="bomb"), 1 / 30)
    assert any(e["type"] == "TRAP_PLACED" and e["actor_id"] == BLACK for e in state.event_log)

    white = state.actors[WHITE]
    white.x, white.y = desk.position[0], desk.position[1]
    state.update(0.0)
    state.apply_action(WHITE, GameAction(ActionType.INTERACT), 1 / 30)

    triggered = [e for e in state.event_log if e["type"] == "TRAP_TRIGGERED" and e["actor_id"] == WHITE]
    assert len(triggered) == 1
    assert triggered[0]["trap_owner_id"] == BLACK
    assert white.is_stunned


def test_detected_trap_can_be_disarmed_by_evaluated_actor():
    state = _new_state(seed=1305)
    desk = state.level.get_object("desk_01")
    black = state.actors[BLACK]
    black.x, black.y = desk.position[0], desk.position[1]
    black.trap_inventory["bomb"] = 1
    state.update(0.0)
    state.apply_action(BLACK, GameAction(ActionType.INSPECT), 1 / 30)  # PLACE_TRAP requires prior inspection (M5D.4)
    black.action_cooldown = 0.0
    state.apply_action(BLACK, GameAction(ActionType.PLACE_TRAP, trap_type="bomb"), 1 / 30)

    entry = state._object_entry("desk_01")
    entry["detected_by"][WHITE] = "bomb"  # simulate a prior successful INSPECT detection

    white = state.actors[WHITE]
    white.x, white.y = desk.position[0], desk.position[1]
    state.update(0.0)
    state.apply_action(WHITE, GameAction(ActionType.DISARM), 1 / 30)

    assert any(e["type"] == "TRAP_DISARMED" and e["actor_id"] == WHITE for e in state.event_log)
    assert entry["trap"] is None


# ---------------------------------------------------------------------------
# Competitive result semantics (spec section 46)
# ---------------------------------------------------------------------------


def test_competitive_outcomes_via_episode_runner():
    from src.evaluation.compatibility import CompatibilityStatus
    from src.evaluation.episode_runner import EpisodeRunner, EpisodeStatus
    from src.evaluation.spec import ExperimentSpec, VariantSpec

    spec = ExperimentSpec(
        experiment_id="m5e_outcome_smoke",
        variants=(
            VariantSpec(
                variant_id="scripted_ref", agent_runtime="scripted", provider="mock", model="mock-model",
                opponent_type="scripted_active",
            ),
        ),
        seeds=(1306,),
        max_episode_game_time=180.0,
        max_episode_wall_time=60.0,
        max_decisions=10000,
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner(frame_dt=1 / 30, poll_sleep=0.0)
    result = runner.run(episode_spec, telemetry_path="/dev/null")

    assert result.compatibility_status == CompatibilityStatus.SUPPORTED
    assert result.status in {EpisodeStatus.COMPLETED, EpisodeStatus.TIMED_OUT, EpisodeStatus.ABORTED}
    assert result.competitive_facts is not None
    assert result.opponent_game_facts is not None
    # Technical failure vs competitive loss: a scripted controller has no
    # provider/runtime failures, so this should never resolve as a
    # disguised "opponent victory" via a technical error path.
    assert result.error_type != "max_runtime_failures_exceeded"


def test_active_opponent_integrity_reconciles(tmp_path):
    import json

    from src.evaluation.experiment_runner import ExperimentRunner
    from src.evaluation.integrity import verify_integrity
    from src.evaluation.spec import ExperimentSpec, VariantSpec

    spec = ExperimentSpec(
        experiment_id="m5e_integrity_smoke",
        variants=(
            VariantSpec(
                variant_id="scripted_ref", agent_runtime="scripted", provider="mock", model="mock-model",
                opponent_type="scripted_active",
            ),
        ),
        seeds=(1309,),
        max_episode_game_time=20.0,
        max_episode_wall_time=30.0,
        max_decisions=10000,
    )
    from src.evaluation.episode_runner import EpisodeRunner

    runner = ExperimentRunner(episode_runner=EpisodeRunner(poll_sleep=0.0), output_root=str(tmp_path))
    result = runner.run(spec, write_outputs=True)

    decisions_path = os.path.join(str(tmp_path), spec.experiment_id, "decisions.jsonl")
    with open(decisions_path, "r", encoding="utf-8") as f:
        raw_events = [json.loads(line) for line in f if line.strip()]
    assert any(e.get("event_type") == "opponent_decision" for e in raw_events)

    report = verify_integrity(result, decisions_jsonl_path=decisions_path)
    assert report.checks.get("opponent_decision_telemetry_present") is True
    assert report.ok, report.problems


def test_technical_failure_is_not_reported_as_opponent_victory():
    """If the evaluated agent aborts on a technical failure, that must be
    recorded as a technical episode failure - never silently as the
    opponent having 'won' (spec section 38).

    M6A.2 real bug found and fixed live: this test's own
    `_AlwaysInvalidClient` returned a malformed dict rather than raising -
    that's actually a content-shape MODEL VALIDATION failure (confirmed:
    it increments `validation_failure_count`, never `provider_failure_count`),
    not the genuine provider/infrastructure failure this test's own name
    and docstring claim to simulate. It only passed before because
    `EpisodeRunner`'s stop-rule read the combined `fallback_count`, which
    conflated the two (see M5F.3B.1's own identical fix in
    `duel_runner.py`, never backported here until this milestone). Fixed
    by actually raising, so this test now exercises what it always meant
    to."""
    import src.game as game_module
    from src.evaluation.episode_runner import EpisodeRunner, EpisodeStatus
    from src.evaluation.spec import ExperimentSpec, VariantSpec

    class _AlwaysInvalidClient:
        def generate_action(self, *args, **kwargs):
            raise RuntimeError("simulated provider outage")

    def build_broken_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return _AlwaysInvalidClient()

    monkeypatch_target = game_module.Game._build_llm_client
    game_module.Game._build_llm_client = build_broken_mock
    try:
        spec = ExperimentSpec(
            experiment_id="m5e_technical_failure_smoke",
            variants=(
                VariantSpec(
                    variant_id="native_mock", agent_runtime="native", provider="mock", model="mock-model",
                    opponent_type="scripted_active",
                ),
            ),
            seeds=(1307,),
            max_runtime_failures=2,
            max_episode_wall_time=30.0,
        )
        episode_spec = spec.expand()[0]
        runner = EpisodeRunner(frame_dt=1 / 30, poll_sleep=0.0)
        result = runner.run(episode_spec, telemetry_path="/dev/null")
    finally:
        game_module.Game._build_llm_client = monkeypatch_target

    assert result.status == EpisodeStatus.ABORTED
    assert result.error_type == "max_runtime_failures_exceeded"
    # A technical failure is not itself a mission_success=False *gameplay*
    # claim about the opponent - the fact is recorded distinctly as an
    # error_type, and mission_success reflects White's own authoritative
    # state, never inferred from "opponent must have won."
    assert result.mission_success is False  # White's own authoritative state - never escaped
    assert result.opponent_game_facts is not None  # Black's own progress is still recorded independently


# ---------------------------------------------------------------------------
# Opponent determinism / replay (spec section 24, STOP condition 17)
# ---------------------------------------------------------------------------


def test_opponent_decision_sequence_is_deterministic_across_replays():
    def replay(seed: int) -> list[tuple[str, str | None]]:
        state = _new_state(seed=seed)
        opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST)
        sequence = []
        for _ in range(80):
            observation = build_observation(state, BLACK)
            action = opponent.get_action(observation)
            sequence.append((action.type.value, action.target_id))
            state.apply_action(BLACK, action, 1 / 30)
            state.update(1 / 30)
        return sequence

    first = replay(1308)
    second = replay(1308)
    assert first == second


# ---------------------------------------------------------------------------
# Recorded opponent-decision telemetry (spec sections 26-28)
# ---------------------------------------------------------------------------


def test_opponent_decision_telemetry_is_actor_tagged_and_uses_reason_codes():
    telemetry = TelemetryLogger()
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST, telemetry=telemetry, episode_id="ep-1")
    desk = _obj("desk_01", inspected=False, searched=False)
    observation = _make_observation(objects=(desk,), nearby_object_id="desk_01")

    opponent.get_action(observation)

    events = [e for e in telemetry.events if e["event_type"] == "opponent_decision"]
    assert len(events) == 1
    event = events[0]
    assert event["actor_id"] == BLACK
    assert event["controller"] == "scripted_active_v1"
    assert event["episode_id"] == "ep-1"
    assert event["selected_action"] == "inspect"
    assert event["target_id"] == "desk_01"
    assert event["reason_code"] == REASON_INSPECT_NEAR_TARGET
    assert "decision_id" in event and event["decision_id"]


def test_opponent_decision_telemetry_collapses_repeated_frame_calls():
    """Bug found in the M5E full 3-seed benchmark: `Game._step` calls
    `get_action` once per render frame (60Hz) regardless of whether
    simulation time is actually advancing - unconditionally emitting
    telemetry on every one of those calls turned a handful of real
    decisions into tens of thousands of byte-identical
    `opponent_decision` rows per episode (up to 54k, ballooning
    decisions.jsonl to 112MB for 9 episodes), most of it logged while the
    evaluated agent's LLM call was pending and game time was frozen. Only
    an actual change in the chosen (action, target, reason) should count
    as a new decision."""
    telemetry = TelemetryLogger()
    opponent = ScriptedBotController(BLACK, profile=OPPORTUNIST, telemetry=telemetry, episode_id="ep-1")
    desk = _obj("desk_01", inspected=False, searched=False)
    observation = _make_observation(objects=(desk,), nearby_object_id="desk_01")

    for _ in range(500):
        opponent.get_action(observation)

    events = [e for e in telemetry.events if e["event_type"] == "opponent_decision"]
    assert len(events) == 1
    assert opponent.decisions_made == 1

    searched_desk = _obj("desk_01", inspected=True, searched=False)
    opponent.get_action(_make_observation(objects=(searched_desk,), nearby_object_id="desk_01"))

    events = [e for e in telemetry.events if e["event_type"] == "opponent_decision"]
    assert len(events) == 2
    assert opponent.decisions_made == 2
