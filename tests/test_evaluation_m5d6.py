"""M5D.6 spec sections 1-3, 32: pre-benchmark navigation sanity +
benchmark-specific correctness tests. No live LLM required anywhere.

Section 1's mandatory pre-benchmark check: confirm M5D.4's proximity fix
gated INTERACTION legality (INSPECT/INTERACT/PLACE_TRAP/DISARM) on
`nearby_object_id`, never MOVE_TO - MOVE_TO stays legal for any known,
reachable interactable object regardless of distance (spec section 2).
"""

from __future__ import annotations

import os

from src import constants
from src.evaluation.compatibility import CompatibilityStatus
from src.evaluation.episode_runner import EpisodeRunner, EpisodeStatus
from src.evaluation.integrity import verify_integrity
from src.evaluation.metrics import compute_metrics
from src.evaluation.paired import paired_seed_rows
from src.evaluation.spec import ExperimentSpec, VariantSpec
from src.items import load_items
from src.level import load_level
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController, enumerate_legal_actions
from src.observation import Observation, ObservedObject, build_observation
from src.state import GameState

WHITE = constants.WHITE_SPY_ID
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "m5d6_corrected_mission_v2.json")


def _load_m5d6_spec() -> ExperimentSpec:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return ExperimentSpec.from_json(f.read())


def _obj(id, **overrides):
    base = dict(
        id=id,
        type="desk",
        label=id,
        position=(500.0, 500.0),
        collision_size=(20.0, 20.0),
        interactable=True,
        trappable=False,
        searched=False,
        inspected=False,
        trap_status="unknown",
        trap_type=None,
        trap_is_mine=False,
        door_open=None,
        lock_type=None,
        locked=False,
        can_unlock=False,
    )
    base.update(overrides)
    return ObservedObject(**base)


def _make_observation(**overrides):
    base = dict(
        actor_id="white",
        seed=1,
        position=(5.0, 5.0),
        facing="east",
        stunned=False,
        stunned_seconds=0.0,
        busy=False,
        busy_seconds=0.0,
        mission_time_remaining=100.0,
        mission_complete=False,
        inventory=frozenset(),
        knowledge=frozenset(),
        required_items=(),
        trap_inventory={},
        nearby_object_id=None,
        items_found_here=frozenset(),
        objects=(),
        nav_grid=None,
        log_event=lambda *a, **k: None,
        current_room="office",
        known_rooms=frozenset({"office"}),
        known_doors=frozenset(),
        known_locked_objects={},
        known_item_locations={},
        visible_other_actor=False,
        opponent=None,
        last_seen_opponent=None,
        revealed_opponent=None,
        revealed_opponent_seconds=0.0,
    )
    base.update(overrides)
    return Observation(**base)


def _legal(observation):
    return {(e["action"], e.get("target_id")) for e in enumerate_legal_actions(observation)}


# ---------------------------------------------------------------------------
# Section 1-3: mandatory pre-benchmark navigation sanity
# ---------------------------------------------------------------------------


def test_far_known_object_move_legal_interaction_illegal():
    far_desk = _obj("desk_01", position=(9000.0, 9000.0))
    observation = _make_observation(objects=(far_desk,), nearby_object_id=None)
    legal = _legal(observation)

    assert ("move", "desk_01") in legal
    assert ("inspect", "desk_01") not in legal
    assert ("interact", "desk_01") not in legal


def test_far_known_door_move_remains_legal():
    far_door = _obj("archives_door", type="door", position=(9000.0, 9000.0))
    observation = _make_observation(objects=(far_door,), nearby_object_id=None)
    assert ("move", "archives_door") in _legal(observation)


def test_move_to_is_not_over_restricted_to_near_object_id_only():
    """The specific regression M5D.6 section 1 asks to rule out: MOVE_TO
    must not have accidentally become gated the same way INTERACT/INSPECT
    are. Multiple far objects must all remain legal MOVE_TO targets at
    once - proving MOVE_TO isn't limited to a single nearby target."""
    a = _obj("desk_01", position=(100.0, 100.0))
    b = _obj("cabinet_01", type="filing_cabinet", position=(2000.0, 50.0))
    c = _obj("safe_01", type="safe", position=(50.0, 3000.0))
    observation = _make_observation(objects=(a, b, c), nearby_object_id=None)
    legal = _legal(observation)

    assert ("move", "desk_01") in legal
    assert ("move", "cabinet_01") in legal
    assert ("move", "safe_01") in legal
    # None of them are near - none should be interaction-legal.
    assert not any(action in {"inspect", "interact"} for action, target in legal if target in {"desk_01", "cabinet_01", "safe_01"})


def test_real_level_has_no_object_fog_of_war_document_the_rule():
    """Section 3's 'unknown object' / 'newly discovered target' scenarios
    assume a discovery mechanic this game doesn't have: `build_observation`
    has always built `objects` from the level's *entire* object list
    (src/observation.py), not a per-room/discovered subset - object
    geometry was never hidden, since M3A (see src/observation.py's own
    module docstring). There is no 'secret_object' to test appearing after
    discovery. What partial observability this game does implement is
    tested elsewhere and confirmed unaffected: opponent position outside
    the shared room, undetected trap existence/type, and unmet lock
    requirement values. This test locks in the actual, documented rule
    rather than assuming the spec's hypothetical discovery mechanic."""
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=601)
    observation = build_observation(state, WHITE)

    # The actor starts in main_office (per level_01.json) but objects from
    # every other room (e.g. the archives' bookshelf, the security room's
    # locker) are already present in Observation - not filtered by
    # visited_rooms.
    known_ids = {obj.id for obj in observation.objects}
    all_level_object_ids = {obj.id for obj in state.level.objects}
    assert known_ids == all_level_object_ids

    legal = enumerate_legal_actions(observation)
    move_targets = {e["target_id"] for e in legal if e["action"] == "move"}
    # Every interactable/door object is a legal MOVE_TO target immediately,
    # confirming section 3's "far known object" rule is the actual,
    # intended rule here (reachability, not discovery).
    for obj in state.level.objects:
        if obj.interactable or obj.type == "door":
            assert obj.id in move_targets


def test_no_hidden_actor_state_leaks_through_legal_actions_or_move():
    """The partial-observability guarantees M5D.6 section 24 asks to
    re-verify: opponent position, undetected traps, and unmet lock
    requirement *values* never appear in legal_actions - only object ids
    and the actor's own legitimately-known booleans (locked/can_unlock)."""
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=602)
    observation = build_observation(state, WHITE)
    legal = enumerate_legal_actions(observation)

    serialized = str(legal)
    other_actor_id = next(a for a in state.actors if a != WHITE)
    assert other_actor_id not in serialized
    # No raw lock "requires" item id ever appears in legal_actions itself -
    # only target ids and action names.
    for entry in legal:
        assert set(entry.keys()) <= {"action", "target_id"}


# ---------------------------------------------------------------------------
# Interaction proximity remains strict (regression guard)
# ---------------------------------------------------------------------------


def test_interaction_proximity_remains_strict_after_all_prior_fixes():
    near = _obj("desk_01")
    far = _obj("desk_02", position=(9000.0, 9000.0))
    observation = _make_observation(objects=(near, far), nearby_object_id="desk_01")
    legal = _legal(observation)

    assert ("inspect", "desk_01") in legal
    assert ("inspect", "desk_02") not in legal
    assert ("move", "desk_02") in legal  # still reachable


# ---------------------------------------------------------------------------
# repair-v2 telemetry required / provider timeout classified separately
# ---------------------------------------------------------------------------


class _TimeoutModelClient:
    def generate_action(self, *args, **kwargs):
        raise TimeoutError("timed out")


def test_provider_timeout_classified_separately_from_other_fallbacks():
    desk = _obj("desk_01")
    observation = _make_observation(objects=(desk,), nearby_object_id="desk_01")
    telemetry = TelemetryLogger()
    controller = LLMController(WHITE, _TimeoutModelClient(), telemetry=telemetry)

    controller.get_action(observation)
    controller._thinking_thread.join(timeout=2.0)

    event = telemetry.events[-1]
    assert event["error_type"] == "provider_error"
    assert "timed out" in (event["error_message"] or "")
    assert event["recovery_outcome"] == "RUNTIME_FAILURE"

    metrics = compute_metrics([event], {}, {})
    assert metrics["provider_timeouts"] == 1
    assert metrics["runtime_exceptions"] == 0
    assert metrics["invalid_model_output_fallbacks"] == 0


def test_runtime_failure_budget_breakdown_sums_to_total_fallbacks():
    decision_events = [
        {"fallback_used": True, "error_type": "provider_error", "error_message": "timed out"},
        {"fallback_used": True, "error_type": "strands_runtime_error", "error_message": "tool crashed"},
        {"fallback_used": True, "error_type": "out_of_range", "error_message": "not legal"},
        {"fallback_used": False, "error_type": None},
    ]
    metrics = compute_metrics(decision_events, {}, {})
    total_fallbacks = sum(1 for e in decision_events if e["fallback_used"])
    assert metrics["provider_timeouts"] + metrics["runtime_exceptions"] + metrics["invalid_model_output_fallbacks"] == total_fallbacks
    assert metrics["provider_timeouts"] == 1
    assert metrics["runtime_exceptions"] == 1
    assert metrics["invalid_model_output_fallbacks"] == 1


# ---------------------------------------------------------------------------
# The frozen M5D.6 config
# ---------------------------------------------------------------------------


def test_m5d6_config_has_new_immutable_id_and_shared_seeds():
    spec = _load_m5d6_spec()
    assert spec.experiment_id == "m5d6-corrected-mission-v2"
    assert spec.experiment_id not in {
        "m5d1-local-baseline-v1",
        "m5d3-corrected-mission-v1",
        "m5d4-live-validation",
        "m5d5-live-validation",
    }
    variant_ids = [v.variant_id for v in spec.variants]
    assert len(variant_ids) == 4
    episodes = spec.expand()
    seeds_by_variant = {}
    for e in episodes:
        seeds_by_variant.setdefault(e.variant.variant_id, set()).add(e.seed)
    assert len({frozenset(s) for s in seeds_by_variant.values()}) == 1


def test_m5d6_strands_qwen_stays_compatibility_only():
    from src.evaluation.compatibility import check_compatibility

    spec = _load_m5d6_spec()
    d_variant = next(v for v in spec.variants if v.agent_runtime == "strands" and v.model == "qwen:latest")
    compatibility = check_compatibility(d_variant)
    assert compatibility.status == CompatibilityStatus.MODEL_CAPABILITY_MISMATCH

    runner = EpisodeRunner()
    episode_spec = next(e for e in spec.expand() if e.variant.variant_id == d_variant.variant_id)
    result = runner.run(episode_spec, telemetry_path="/dev/null")
    assert result.status == EpisodeStatus.INCOMPATIBLE
    assert result.mission_success is None


# ---------------------------------------------------------------------------
# Integrity: recovery_telemetry_present, using a real (mocked) harness run
# ---------------------------------------------------------------------------


def _mock_variant(variant_id: str) -> VariantSpec:
    return VariantSpec(variant_id=variant_id, agent_runtime="native", provider="mock", model="mock-model")


def test_recovery_telemetry_present_and_integrity_reconciles(tmp_path):
    import json

    from src.evaluation.experiment_runner import ExperimentRunner

    spec = ExperimentSpec(
        experiment_id="m5d6_recovery_smoke",
        variants=(_mock_variant("native_mock"),),
        seeds=(9101,),
        max_decisions=2,
        max_episode_wall_time=30.0,
    )
    runner = ExperimentRunner(output_root=str(tmp_path))
    result = runner.run(spec, write_outputs=True)

    decisions_path = os.path.join(str(tmp_path), spec.experiment_id, "decisions.jsonl")
    with open(decisions_path, "r", encoding="utf-8") as f:
        raw_events = [json.loads(line) for line in f if line.strip()]
    decision_events = [e for e in raw_events if e.get("event_type") == "llm_decision"]
    assert decision_events
    assert all(e.get("recovery_outcome") is not None for e in decision_events)

    report = verify_integrity(result, decisions_jsonl_path=decisions_path)
    assert report.checks.get("recovery_telemetry_present") is True
    assert report.ok, report.problems


# ---------------------------------------------------------------------------
# Per-seed pairing with M5D.6's recovery/navigation fields
# ---------------------------------------------------------------------------


def test_m5d6_paired_rows_include_recovery_fields(tmp_path):
    from src.evaluation.experiment_runner import ExperimentRunner

    spec = ExperimentSpec(
        experiment_id="m5d6_pairing_smoke",
        variants=(_mock_variant("native_a"), _mock_variant("native_b")),
        seeds=(9102,),
        max_decisions=2,
        max_episode_wall_time=30.0,
    )
    runner = ExperimentRunner(output_root=str(tmp_path))
    result = runner.run(spec, write_outputs=False)

    fields = ("repair_success_rate", "same_illegal_action_repeated", "provider_timeouts", "unique_rooms_visited")
    rows = paired_seed_rows(result, "native_a", "native_b", fields=fields)
    assert len(rows) == 1
    for field_name in fields:
        assert f"native_a.{field_name}" in rows[0]
        assert f"native_b.{field_name}" in rows[0]
