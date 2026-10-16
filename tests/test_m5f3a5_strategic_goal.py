"""M5F.3A.5: network-free tests for the strategic goal selection contract
(`src/evaluation/strategic_goal.py`) - spec section 53's required
coverage. No live calls."""

from __future__ import annotations

from src import constants
from src.evaluation.order_control import OrderPolicy
from src.evaluation.strategic_goal import (
    DECISION_CONTRACT_GAME_ACTION,
    DECISION_CONTRACT_STRATEGIC_GOAL,
    count_door_move_candidates,
    resolve_goal,
    strategic_goal_model_payload_transform,
)
from src.evaluation.symmetric_harness import alpha_beta_object_ids, build_symmetric_state
from src.llm_controller import LLMControllerConfig, compact_observation_for_model, enumerate_legal_actions, serialize_observation
from src.observation import build_observation


def _base_payload():
    state = build_symmetric_state(neutral_ids=True)
    obs = build_observation(state, constants.WHITE_SPY_ID)
    payload = serialize_observation(obs)
    model_payload = compact_observation_for_model(payload)
    candidates = alpha_beta_object_ids(True, neutral_ids=True)
    return state, obs, payload, model_payload, candidates


def test_strategic_goal_mode_default_off():
    config = LLMControllerConfig()
    assert config.model_payload_transform is None
    assert DECISION_CONTRACT_GAME_ACTION != DECISION_CONTRACT_STRATEGIC_GOAL


def test_existing_callers_unchanged_without_transform():
    state, obs, payload, model_payload, candidates = _base_payload()
    # No transform applied - CURRENT behavior, doors present as before.
    assert count_door_move_candidates(model_payload) == 2


def test_goal_candidates_generated_deterministically():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink1, sink2 = {}, {}
    t1 = strategic_goal_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink1)
    t2 = strategic_goal_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink2)
    r1, r2 = t1(model_payload), t2(model_payload)
    assert r1["candidate_goals"] == r2["candidate_goals"]


def test_neutral_ids_stable():
    for gid in alpha_beta_object_ids(True, neutral_ids=True):
        assert "alpha" not in gid.lower()
        assert "beta" not in gid.lower()
        assert gid not in ("1", "2", "left", "right", "A", "B")


def test_order_reversal_changes_presentation_only():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink_a, sink_b = {}, {}
    t_a = strategic_goal_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink_a)
    t_b = strategic_goal_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.REVERSED, trial_index=0, metadata_sink=sink_b)
    r_a, r_b = t_a(model_payload), t_b(model_payload)
    order_a = [c["goal_id"] for c in r_a["candidate_goals"]]
    order_b = [c["goal_id"] for c in r_b["candidate_goals"]]
    assert set(order_a) == set(order_b) == set(candidates)
    assert order_a == list(reversed(order_b))
    # Non-candidate context (self/time/memory/opponent/mission/communication) unchanged.
    for key in ("self", "time", "memory", "opponent", "mission", "communication"):
        assert r_a[key] == r_b[key] == model_payload[key]


def test_goal_resolver_maps_correctly():
    state, obs, payload, model_payload, candidates = _base_payload()
    candidate_map = {c: c for c in candidates}
    for gid in candidates:
        assert resolve_goal(gid, candidate_map) == gid


def test_resolver_output_uses_existing_gameaction_target_id():
    # The resolved id must be a real, legal target_id under the true
    # (unmodified) Observation - confirmed via enumerate_legal_actions,
    # never a synthetic identifier the engine wouldn't recognize.
    state, obs, payload, model_payload, candidates = _base_payload()
    legal = enumerate_legal_actions(obs)
    legal_move_targets = {a.get("target_id") for a in legal if a["action"] == "move"}
    for gid in candidates:
        resolved = resolve_goal(gid, {c: c for c in candidates})
        assert resolved in legal_move_targets


def test_invalid_goal_rejected():
    candidate_map = {c: c for c in alpha_beta_object_ids(True, neutral_ids=True)}
    assert resolve_goal("not_a_real_goal_id", candidate_map) is None


def test_fallback_does_not_pick_first_candidate():
    # The critical anti-bias rule (spec section 34): an unresolvable goal
    # must return None, never candidate_map's first value.
    candidates = alpha_beta_object_ids(True, neutral_ids=True)
    candidate_map = {c: c for c in candidates}
    result = resolve_goal("garbage", candidate_map)
    assert result is None
    assert result != list(candidate_map.values())[0]


def test_door_intermediate_targets_absent_from_selectable_goal_list():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    t = strategic_goal_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink)
    result = t(model_payload)
    assert count_door_move_candidates(result) == 0
    move_targets = result["legal_actions_compact"]["move"]
    assert set(move_targets) == set(candidates)
    # Doors remain visible as descriptive world facts (spec section 30),
    # just not as selectable move peers.
    assert any(o["id"].startswith("door_") for o in result["room"]["objects"])


def test_gamestate_unchanged_by_strategic_goal_transform():
    state, obs, payload, model_payload, candidates = _base_payload()
    original_level_objects = list(state.level.objects)
    sink = {}
    t = strategic_goal_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.REVERSED, trial_index=0, metadata_sink=sink)
    t(model_payload)
    assert state.level.objects == original_level_objects


def test_no_engine_bypass_legality_unaffected():
    # Legality (what enumerate_legal_actions reports against the real
    # Observation) is completely untouched by the display transform.
    state, obs, payload, model_payload, candidates = _base_payload()
    canonical_legal = enumerate_legal_actions(obs)
    sink = {}
    t = strategic_goal_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.REVERSED, trial_index=0, metadata_sink=sink)
    t(model_payload)
    assert enumerate_legal_actions(obs) == canonical_legal


def test_telemetry_metadata_emitted():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    t = strategic_goal_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.COUNTERBALANCED, trial_index=1, metadata_sink=sink)
    t(model_payload)
    assert sink["metadata"]["representation_mode"] == DECISION_CONTRACT_STRATEGIC_GOAL
    assert "goal_order" in sink["metadata"]
    assert sink["metadata"]["goal_order"]["presented_order"] == list(reversed(candidates))


def test_one_model_call_semantics_preserved():
    # This module never introduces a second output schema or a second
    # LLM call path - it reuses run_symmetric_trial/_run_one_decision's
    # existing single-decision-per-trial contract unchanged (verified by
    # source inspection: strategic_goal.py imports no model client, makes
    # no generate_action call of its own).
    import inspect

    import src.evaluation.strategic_goal as mod

    source = inspect.getsource(mod)
    assert "generate_action" not in source
    assert "ModelClient(" not in source


def test_prompt_payload_secret_safe():
    import json

    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    t = strategic_goal_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink)
    result = t(model_payload)
    serialized = json.dumps(result, default=str)
    assert "AWS_BEARER_TOKEN_BEDROCK" not in serialized
    assert "bedrock-api-key" not in serialized
