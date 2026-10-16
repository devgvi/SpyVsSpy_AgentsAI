"""M6A.2 Phase A (spec section 41) + Phase B dry pair (spec section 42):
network-free tests for pairing/first-divergence/trajectory-trace/
classification logic, plus one real (mocked) `EpisodeRunner` episode to
confirm the pipeline works against genuine telemetry event shape, not
just hand-built fixtures. No network calls.

Also includes a direct regression test for the real bug found and fixed
while starting this milestone: `Game._build_llm_agent` never passed
`personality_profile` through for `agent_runtime="strands"`, even though
`StrandsAgentAdapter` has supported it since M6A - meaning a Strands
episode's personality would have silently differed from its paired
Native episode's, an M6A.2 STOP CONDITION violation (item 3) before this
fix.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json

from src.evaluation.episode_trajectory_analysis import (
    DETECTED_TRAP,
    DIVERGENCE_FOUND,
    EARLY_DIVERGENCE_HIGH_IMPACT,
    EARLY_DIVERGENCE_LOW_IMPACT,
    INTERACTION_AVAILABLE,
    INTRINSIC_VARIANCE_DOMINATES,
    INTRINSIC_VARIANCE_POSSIBLE,
    LATE_DIVERGENCE,
    LOCKED_OBJECT,
    MULTIPLE_LEGAL_TARGETS,
    NO_DIVERGENCE,
    NO_DIVERGENCE_CLASS,
    PAIR_INTEGRITY_BROKEN,
    RECONVERGED,
    RUNTIME_EFFECT_CONTEXT_DEPENDENT,
    RUNTIME_EFFECT_LOCAL_BUT_TRAJECTORY_SMALL,
    RUNTIME_EFFECT_SUPPORTED_CONTEXT,
    RUNTIME_TRAJECTORY_EFFECT_SUPPORTED,
    SIMPLE_NAVIGATION,
    TECHNICAL_COST_DIFFERENCE_ONLY,
    TECHNICALLY_INVALID,
    TRAP_PLACEMENT_OPPORTUNITY,
    build_trajectory_trace,
    categorize_decision_context,
    classify_overall_trajectory,
    classify_pair,
    decision_state_key,
    event_legal_actions_hash,
    event_observation_hash,
    find_first_divergence,
    memory_summary_hash,
    tag_provider_variance_context,
)

# -- synthetic event fixtures (schema-accurate, matching the real
# `llm_decision` event probed live via a mocked EpisodeRunner run) --------


def _event(decision_index, action, target=None, trap_type=None, legal_actions=None, near_object_id=None,
           locked=False, can_unlock=True, room="main_office", extra_memory=0):
    legal_actions = legal_actions if legal_actions is not None else [{"action": "wait"}, {"action": "move", "target_id": "desk_01"}]
    objects = [{"id": near_object_id, "locked": locked, "can_unlock": can_unlock}] if near_object_id else []
    obs = {
        "self": {"near_object_id": near_object_id, "room": room},
        "room": {"id": room, "objects": objects},
    }
    obs_payload_for_hash = {"decision_index": decision_index, "room": room, "near_object_id": near_object_id}
    from src.evaluation.runtime_variance_analysis import stable_hash

    return {
        "event_type": "llm_decision", "decision_index": decision_index,
        "executed_action": {"action": action, "target_id": target, "trap_type": trap_type},
        "legal_actions": legal_actions,
        "observation": obs,
        "observation_hash": stable_hash(obs_payload_for_hash),
        "visited_rooms_count": 1 + extra_memory, "known_requirements_count": 0, "known_empty_objects_count": 0,
        "recent_events_count": 0, "recent_failures_count": 0, "recent_decisions_count": decision_index, "recent_repetitions_count": 0,
    }


def _same_state_pair(decision_index, native_action, strands_action, **kwargs):
    """Two events sharing identical state (observation/legal_actions/
    memory) but possibly different executed actions - the exact shape a
    genuine same-seed pair produces before/at a divergence point."""
    n = _event(decision_index, native_action, **kwargs)
    s = _event(decision_index, strands_action, **kwargs)
    s["observation_hash"] = n["observation_hash"]  # same state by construction
    return n, s


# -- categorize_decision_context (spec section 27) ---------------------------


def test_categorize_simple_navigation():
    e = _event(0, "move", "desk_01", legal_actions=[{"action": "wait"}, {"action": "move", "target_id": "desk_01"}])
    assert categorize_decision_context(e) == SIMPLE_NAVIGATION


def test_categorize_trap_placement_opportunity():
    e = _event(0, "place_trap", "safe", trap_type="electric", legal_actions=[{"action": "wait"}, {"action": "place_trap", "target_id": "safe", "trap_type": "electric"}])
    assert categorize_decision_context(e) == TRAP_PLACEMENT_OPPORTUNITY


def test_categorize_detected_trap():
    e = _event(0, "disarm", "cabinet", legal_actions=[{"action": "wait"}, {"action": "disarm", "target_id": "cabinet"}])
    assert categorize_decision_context(e) == DETECTED_TRAP


def test_categorize_locked_object():
    e = _event(0, "inspect", "safe", near_object_id="safe", locked=True, can_unlock=False,
               legal_actions=[{"action": "wait"}, {"action": "inspect", "target_id": "safe"}])
    assert categorize_decision_context(e) == LOCKED_OBJECT


def test_categorize_multiple_legal_targets():
    e = _event(0, "interact", "desk_01", near_object_id="desk_01",
               legal_actions=[{"action": "wait"}, {"action": "interact", "target_id": "desk_01"},
                               {"action": "move", "target_id": "desk_01"}, {"action": "move", "target_id": "door_01"}])
    assert categorize_decision_context(e) == MULTIPLE_LEGAL_TARGETS


def test_categorize_interaction_available_single_move_target():
    e = _event(0, "interact", "desk_01", near_object_id="desk_01",
               legal_actions=[{"action": "wait"}, {"action": "interact", "target_id": "desk_01"}, {"action": "move", "target_id": "desk_01"}])
    assert categorize_decision_context(e) == INTERACTION_AVAILABLE


def test_tag_provider_variance_context():
    assert tag_provider_variance_context(MULTIPLE_LEGAL_TARGETS) == RUNTIME_EFFECT_SUPPORTED_CONTEXT
    assert tag_provider_variance_context(TRAP_PLACEMENT_OPPORTUNITY) == INTRINSIC_VARIANCE_POSSIBLE
    assert tag_provider_variance_context(SIMPLE_NAVIGATION) is None


# -- state key / hash stability (spec section 15-16) -------------------------


def test_decision_state_key_stable_for_identical_event():
    e = _event(0, "move", "desk_01")
    assert decision_state_key(e) == decision_state_key(dict(e))


def test_decision_state_key_differs_on_memory_drift():
    a = _event(0, "move", "desk_01", extra_memory=0)
    b = _event(0, "move", "desk_01", extra_memory=5)
    assert memory_summary_hash(a) != memory_summary_hash(b)
    assert decision_state_key(a) != decision_state_key(b)


def test_event_legal_actions_hash_matches_reused_function():
    e = _event(0, "move", "desk_01")
    assert event_legal_actions_hash(e) == event_legal_actions_hash(dict(e))


def test_event_observation_hash_reads_the_real_field():
    e = _event(0, "move", "desk_01")
    assert event_observation_hash(e) == e["observation_hash"]


# -- first-divergence detection (spec sections 14, 17) -----------------------


def test_no_divergence_when_every_paired_decision_matches():
    native = [_event(i, "move", "desk_01") for i in range(3)]
    strands = [_event(i, "move", "desk_01") for i in range(3)]
    for n, s in zip(native, strands):
        s["observation_hash"] = n["observation_hash"]
    result = find_first_divergence(native, strands)
    assert result["status"] == NO_DIVERGENCE


def test_pair_integrity_broken_at_decision_zero():
    native = [_event(0, "move", "desk_01", room="main_office")]
    strands = [_event(0, "move", "desk_01", room="archives")]  # different room = different world state, decision 0
    result = find_first_divergence(native, strands)
    assert result["status"] == PAIR_INTEGRITY_BROKEN
    assert result["decision_index"] == 0


def test_divergence_found_at_action_mismatch():
    e0n, e0s = _same_state_pair(0, "move", "move", target="desk_01")
    e0s["executed_action"]["target_id"] = "desk_01"
    e1n, e1s = _same_state_pair(1, "interact", "inspect", target="desk_01", near_object_id="desk_01")
    native = [e0n, e1n]
    strands = [e0s, e1s]
    result = find_first_divergence(native, strands)
    assert result["status"] == DIVERGENCE_FOUND
    assert result["decision_index"] == 1
    assert result["native_action"] == "INTERACT|desk_01"
    assert result["strands_action"] == "INSPECT|desk_01"


def test_divergence_context_and_provider_variance_tag_recorded():
    e0n, e0s = _same_state_pair(0, "interact", "inspect", target="desk_01", near_object_id="desk_01",
                                 legal_actions=[{"action": "wait"}, {"action": "interact", "target_id": "desk_01"},
                                                {"action": "move", "target_id": "desk_01"}, {"action": "move", "target_id": "door_01"}])
    result = find_first_divergence([e0n], [e0s])
    assert result["status"] == DIVERGENCE_FOUND
    assert result["context_category"] == MULTIPLE_LEGAL_TARGETS
    assert result["provider_variance_tag"] == RUNTIME_EFFECT_SUPPORTED_CONTEXT


def test_divergence_stops_at_first_mismatch_not_a_later_one():
    e0n, e0s = _same_state_pair(0, "move", "move", target="desk_01")
    e1n, e1s = _same_state_pair(1, "interact", "inspect", target="desk_01", near_object_id="desk_01")
    e2n, e2s = _same_state_pair(2, "move", "wait", target="exit_door")  # would also differ, never reached
    result = find_first_divergence([e0n, e1n, e2n], [e0s, e1s, e2s])
    assert result["decision_index"] == 1


# -- trajectory trace (spec section 50) --------------------------------------


def test_trajectory_trace_marks_first_divergence():
    events = [_event(i, "move", "desk_01", room=f"room{i}") for i in range(3)]
    trace = build_trajectory_trace(events, "native", first_divergence_index=1)
    assert [t["decision_index"] for t in trace] == [0, 1, 2]
    assert [t["is_first_divergence"] for t in trace] == [False, True, False]
    assert trace[0]["room"] == "room0"
    assert all(t["runtime"] == "native" for t in trace)


def test_trajectory_trace_no_marker_when_no_divergence():
    events = [_event(i, "move", "desk_01") for i in range(2)]
    trace = build_trajectory_trace(events, "strands", first_divergence_index=None)
    assert all(not t["is_first_divergence"] for t in trace)


# -- pair classification (spec section 53) -----------------------------------


_SAME_OUTCOME = {"mission_success": True, "escape_success": True, "mission_items_acquired": 3, "rooms_visited": ["a", "b"]}
_DIFFERENT_OUTCOME = {"mission_success": False, "escape_success": False, "mission_items_acquired": 1, "rooms_visited": ["a"]}
# Completion/escape match (the "major" outcomes) but item count/room
# coverage differs slightly - the genuine EARLY_DIVERGENCE_LOW_IMPACT case.
_MINOR_DIFFERENT_OUTCOME = {"mission_success": True, "escape_success": True, "mission_items_acquired": 2, "rooms_visited": ["a", "b"]}


def test_classify_pair_no_divergence():
    div = {"status": NO_DIVERGENCE, "decision_index": None}
    assert classify_pair(div, _SAME_OUTCOME, _SAME_OUTCOME, 10, 10) == NO_DIVERGENCE_CLASS


def test_classify_pair_technically_invalid_on_integrity_break():
    div = {"status": PAIR_INTEGRITY_BROKEN, "decision_index": 0}
    assert classify_pair(div, _SAME_OUTCOME, _DIFFERENT_OUTCOME, 10, 10) == TECHNICALLY_INVALID


def test_classify_pair_technically_invalid_flag():
    div = {"status": DIVERGENCE_FOUND, "decision_index": 5}
    assert classify_pair(div, _SAME_OUTCOME, _DIFFERENT_OUTCOME, 20, 20, technically_valid=False) == TECHNICALLY_INVALID


def test_classify_pair_early_high_impact():
    div = {"status": DIVERGENCE_FOUND, "decision_index": 2}
    assert classify_pair(div, _SAME_OUTCOME, _DIFFERENT_OUTCOME, 20, 20) == EARLY_DIVERGENCE_HIGH_IMPACT


def test_classify_pair_reconverged_when_outcomes_match_despite_divergence():
    div = {"status": DIVERGENCE_FOUND, "decision_index": 2}
    assert classify_pair(div, _SAME_OUTCOME, dict(_SAME_OUTCOME), 20, 20) == RECONVERGED


def test_classify_pair_late_divergence():
    div = {"status": DIVERGENCE_FOUND, "decision_index": 18}
    assert classify_pair(div, _SAME_OUTCOME, _DIFFERENT_OUTCOME, 20, 20) == LATE_DIVERGENCE


def test_classify_pair_early_low_impact():
    # Regression test for a real bug found while writing this test: an
    # earlier single-boolean impact check made this branch unreachable
    # dead code (see `_outcome_impact`'s own docstring). Completion/escape
    # match, only item count differs, divergence is early -> genuinely
    # low, not zero, impact.
    div = {"status": DIVERGENCE_FOUND, "decision_index": 1}
    assert classify_pair(div, _SAME_OUTCOME, _MINOR_DIFFERENT_OUTCOME, 10, 10) == EARLY_DIVERGENCE_LOW_IMPACT


def test_classify_pair_reconverged_takes_priority_over_early_when_outcomes_fully_match():
    # Spec section 22: do not assume every divergence is high impact -
    # an early divergence with fully matching final outcomes is
    # RECONVERGED, not EARLY_DIVERGENCE_LOW_IMPACT.
    div = {"status": DIVERGENCE_FOUND, "decision_index": 1}
    assert classify_pair(div, _SAME_OUTCOME, dict(_SAME_OUTCOME), 10, 10) == RECONVERGED


# -- overall classification (spec section 54) --------------------------------


def test_overall_runtime_trajectory_effect_supported():
    labels = [EARLY_DIVERGENCE_HIGH_IMPACT, EARLY_DIVERGENCE_HIGH_IMPACT, NO_DIVERGENCE_CLASS]
    tags = [RUNTIME_EFFECT_SUPPORTED_CONTEXT, RUNTIME_EFFECT_SUPPORTED_CONTEXT, None]
    assert classify_overall_trajectory(labels, tags, cost_differs=True) == RUNTIME_TRAJECTORY_EFFECT_SUPPORTED


def test_overall_local_but_trajectory_small():
    labels = [RECONVERGED, LATE_DIVERGENCE, NO_DIVERGENCE_CLASS]
    tags = [None, None, None]
    assert classify_overall_trajectory(labels, tags, cost_differs=True) == RUNTIME_EFFECT_LOCAL_BUT_TRAJECTORY_SMALL


def test_overall_intrinsic_variance_dominates_no_divergence_no_cost():
    labels = [NO_DIVERGENCE_CLASS, NO_DIVERGENCE_CLASS, NO_DIVERGENCE_CLASS]
    assert classify_overall_trajectory(labels, [None, None, None], cost_differs=False) == INTRINSIC_VARIANCE_DOMINATES


def test_overall_technical_cost_difference_only_no_divergence_but_cost_differs():
    labels = [NO_DIVERGENCE_CLASS, NO_DIVERGENCE_CLASS]
    assert classify_overall_trajectory(labels, [None, None], cost_differs=True) == TECHNICAL_COST_DIFFERENCE_ONLY


def test_overall_context_dependent_fallback():
    labels = [EARLY_DIVERGENCE_HIGH_IMPACT, NO_DIVERGENCE_CLASS, NO_DIVERGENCE_CLASS]
    tags = [RUNTIME_EFFECT_SUPPORTED_CONTEXT, None, None]
    result = classify_overall_trajectory(labels, tags, cost_differs=True)
    assert result == RUNTIME_EFFECT_CONTEXT_DEPENDENT


def test_overall_empty_is_mixed():
    assert classify_overall_trajectory([], [], cost_differs=False) == "MIXED"


# -- Game -> Strands personality-propagation bug fix (found this milestone) --


def test_game_strands_personality_reaches_the_adapter():
    from src.game import Game
    from src.llm.personality import SABOTEUR, render_personality_block

    game = Game(seed=1, white_controller="llm", agent_runtime="strands", white_personality=SABOTEUR)
    controller = game.bot_controller
    assert controller.personality_profile == SABOTEUR
    assert render_personality_block(SABOTEUR) in controller._effective_system_prompt


def test_game_strands_default_personality_is_byte_identical_to_no_personality():
    from src.game import Game
    from src.agents.strands_agent import STRANDS_SYSTEM_PROMPT

    game = Game(seed=1, white_controller="llm", agent_runtime="strands")
    assert game.bot_controller._effective_system_prompt == STRANDS_SYSTEM_PROMPT


# -- Phase B: one real (mocked) EpisodeRunner episode, genuine telemetry ----


def test_phase_b_dry_pair_no_false_divergence_on_real_telemetry(tmp_path):
    """Spec section 42: runs one real, network-free `EpisodeRunner`
    episode (MockModelClient, scripted_active_v1/OPPORTUNIST opponent),
    then verifies `find_first_divergence` reports NO_DIVERGENCE when the
    same real telemetry is compared against itself - proving the
    analysis pipeline doesn't manufacture a false divergence from genuine
    event shape/ordering quirks that a hand-built fixture might miss."""
    import src.game as game_module
    from src.llm.clients.mock_client import MockModelClient
    from src.evaluation.episode_runner import EpisodeRunner
    from src.evaluation.spec import EpisodeSpec, VariantSpec, MAP_VERSION
    from src.llm.agent_memory import AgentMemory
    from src.llm_controller import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, PROMPT_VERSION

    def build_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return MockModelClient(response=[
            {"action": "move", "target_id": "door_corridor_main_office", "reason": "x"},
            {"action": "move", "target_id": "desk_01", "reason": "x"},
            {"action": "inspect", "target_id": "desk_01", "reason": "x"},
        ])

    orig = game_module.Game._build_llm_client
    game_module.Game._build_llm_client = build_mock
    try:
        variant = VariantSpec(
            variant_id="A", agent_runtime="native_tool_like", provider="mock", model="mock-model",
            opponent_type="scripted_active", opponent_profile="OPPORTUNIST", personality_profile="OPPORTUNIST",
        )
        spec = EpisodeSpec(
            experiment_id="m6a2_phase_b", episode_id="m6a2_phase_b_ep1", variant=variant, seed=4242, execution_mode="evaluation",
            max_episode_game_time=60.0, max_episode_wall_time=30.0, max_decisions=3, max_runtime_failures=5,
            prompt_version=PROMPT_VERSION, observation_schema_version=OBSERVATION_SCHEMA_VERSION,
            memory_schema_version=AgentMemory.MEMORY_SCHEMA_VERSION, action_schema_version=ACTION_SCHEMA_VERSION,
            map_version=MAP_VERSION,
        )
        telemetry_path = str(tmp_path / "decisions.jsonl")
        runner = EpisodeRunner()
        result = runner.run(spec, telemetry_path=telemetry_path)
        assert result.decision_count == 3
    finally:
        game_module.Game._build_llm_client = orig

    with open(telemetry_path) as f:
        events = [json.loads(l) for l in f if l.strip()]
    decision_events = [e for e in events if e.get("event_type") == "llm_decision"]
    assert len(decision_events) == 3

    outcome = find_first_divergence(decision_events, decision_events)
    assert outcome["status"] == NO_DIVERGENCE

    trace = build_trajectory_trace(decision_events, "native", first_divergence_index=None)
    assert len(trace) == 3
    assert trace[0]["action"] == "move" and trace[0]["target"] == "door_corridor_main_office"
    assert trace[2]["action"] == "inspect" and trace[2]["target"] == "desk_01"
