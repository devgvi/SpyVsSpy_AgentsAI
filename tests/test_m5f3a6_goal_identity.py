"""M5F.3A.6: network-free tests for goal-identity/world-target cross-
mapping isolation (`src/evaluation/goal_identity.py`) - spec section 21's
required coverage. No live calls."""

from __future__ import annotations

from src import constants
from src.evaluation.goal_identity import (
    ID_PAIRS,
    MAPPING_1,
    MAPPING_2,
    build_goal_mapping,
    goal_identity_model_payload_transform,
    goal_identity_model_response_transform,
    token_shape_audit,
)
from src.evaluation.order_control import OrderPolicy
from src.evaluation.symmetric_harness import alpha_beta_object_ids, build_symmetric_state
from src.llm_controller import LLMControllerConfig, compact_observation_for_model, enumerate_legal_actions, serialize_observation
from src.observation import build_observation


def _world_targets():
    return alpha_beta_object_ids(True, neutral_ids=True)


def _base_payload():
    state = build_symmetric_state(neutral_ids=True)
    obs = build_observation(state, constants.WHITE_SPY_ID)
    payload = serialize_observation(obs)
    model_payload = compact_observation_for_model(payload)
    return state, obs, payload, model_payload


def test_at_least_three_id_pairs_defined():
    assert len(ID_PAIRS) >= 3
    for pair in ID_PAIRS:
        assert len(pair) == 2
        assert pair[0] != pair[1]


def test_ids_neutral_non_semantic():
    forbidden = ("alpha", "beta", "left", "right", "security", "archives", "exit", "safe")
    for pair in ID_PAIRS:
        for gid in pair:
            assert gid.lower() not in forbidden
            assert not gid.isdigit()


def test_id_a_maps_to_x_under_mapping_1():
    world_targets = _world_targets()
    mapping = build_goal_mapping(ID_PAIRS[0], world_targets, MAPPING_1)
    assert mapping[ID_PAIRS[0][0]] == world_targets[0]
    assert mapping[ID_PAIRS[0][1]] == world_targets[1]


def test_id_a_maps_to_y_after_swap():
    world_targets = _world_targets()
    mapping = build_goal_mapping(ID_PAIRS[0], world_targets, MAPPING_2)
    assert mapping[ID_PAIRS[0][0]] == world_targets[1]
    assert mapping[ID_PAIRS[0][1]] == world_targets[0]


def test_id_b_mirrors_correctly():
    world_targets = _world_targets()
    id_a, id_b = ID_PAIRS[1]
    m1 = build_goal_mapping((id_a, id_b), world_targets, MAPPING_1)
    m2 = build_goal_mapping((id_a, id_b), world_targets, MAPPING_2)
    assert m1[id_b] != m2[id_b]
    assert {m1[id_b], m2[id_b]} == set(world_targets)


def test_order_changes_presentation_only():
    state, obs, payload, model_payload = _base_payload()
    id_pair = ID_PAIRS[0]
    sink_a, sink_b = {}, {}
    t_a = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink_a)
    t_b = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=OrderPolicy.REVERSED, trial_index=0, metadata_sink=sink_b)
    r_a, r_b = t_a(model_payload), t_b(model_payload)
    order_a = [c["goal_id"] for c in r_a["candidate_goals"]]
    order_b = [c["goal_id"] for c in r_b["candidate_goals"]]
    assert order_a == list(reversed(order_b))
    assert set(order_a) == set(id_pair)
    for key in ("self", "time", "memory", "opponent", "mission", "communication"):
        assert r_a[key] == r_b[key] == model_payload[key]


def test_mapping_changes_resolver_target_only_not_presentation():
    state, obs, payload, model_payload = _base_payload()
    id_pair = ID_PAIRS[0]
    world_targets = _world_targets()
    sink = {}
    t = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink)
    result = t(model_payload)
    # The display never reveals which real target an id maps to.
    for c in result["candidate_goals"]:
        assert c["goal_id"] not in world_targets
    assert not any(t in str(result) for t in world_targets)


def test_world_geometry_unchanged():
    state, obs, payload, model_payload = _base_payload()
    original_objects = list(state.level.objects)
    id_pair = ID_PAIRS[0]
    world_targets = _world_targets()
    mapping = build_goal_mapping(id_pair, world_targets, MAPPING_1)
    sink_p, sink_r = {}, {}
    tp = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink_p)
    tr = goal_identity_model_response_transform(goal_to_target=mapping, metadata_sink=sink_r)
    tp(model_payload)
    tr({"action": "move", "target_id": id_pair[0]})
    assert state.level.objects == original_objects


def test_gamestate_unchanged():
    state, obs, payload, model_payload = _base_payload()
    original_object_states = {k: dict(v) for k, v in state.object_states.items()}
    id_pair = ID_PAIRS[0]
    sink = {}
    t = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=OrderPolicy.REVERSED, trial_index=1, metadata_sink=sink)
    t(model_payload)
    assert {k: dict(v) for k, v in state.object_states.items()} == original_object_states


def test_resolver_deterministic():
    world_targets = _world_targets()
    mapping = build_goal_mapping(ID_PAIRS[0], world_targets, MAPPING_1)
    sink1, sink2 = {}, {}
    t1 = goal_identity_model_response_transform(goal_to_target=mapping, metadata_sink=sink1)
    t2 = goal_identity_model_response_transform(goal_to_target=mapping, metadata_sink=sink2)
    r1 = t1({"action": "move", "target_id": ID_PAIRS[0][0], "reason": "x"})
    r2 = t2({"action": "move", "target_id": ID_PAIRS[0][0], "reason": "x"})
    assert r1["target_id"] == r2["target_id"]


def test_resolved_target_is_real_legal_id():
    state, obs, payload, model_payload = _base_payload()
    world_targets = _world_targets()
    legal_move_targets = {a.get("target_id") for a in enumerate_legal_actions(obs) if a["action"] == "move"}
    mapping = build_goal_mapping(ID_PAIRS[0], world_targets, MAPPING_1)
    sink = {}
    t = goal_identity_model_response_transform(goal_to_target=mapping, metadata_sink=sink)
    result = t({"action": "move", "target_id": ID_PAIRS[0][0], "reason": "x"})
    assert result["target_id"] in legal_move_targets


def test_fallback_neutral_unresolvable_id_untouched():
    world_targets = _world_targets()
    mapping = build_goal_mapping(ID_PAIRS[0], world_targets, MAPPING_1)
    sink = {}
    t = goal_identity_model_response_transform(goal_to_target=mapping, metadata_sink=sink)
    result = t({"action": "move", "target_id": "not_a_real_goal_id", "reason": "x"})
    # Left untouched - not silently remapped to any candidate.
    assert result["target_id"] == "not_a_real_goal_id"
    assert sink["resolution"]["success"] is False


def test_original_goal_id_preserved_in_response():
    world_targets = _world_targets()
    mapping = build_goal_mapping(ID_PAIRS[0], world_targets, MAPPING_1)
    sink = {}
    t = goal_identity_model_response_transform(goal_to_target=mapping, metadata_sink=sink)
    result = t({"action": "move", "target_id": ID_PAIRS[0][0], "reason": "x"})
    assert result["_original_goal_id"] == ID_PAIRS[0][0]
    assert result["target_id"] == mapping[ID_PAIRS[0][0]]


def test_telemetry_exact():
    state, obs, payload, model_payload = _base_payload()
    id_pair = ID_PAIRS[0]
    sink = {}
    t = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=OrderPolicy.COUNTERBALANCED, trial_index=1, metadata_sink=sink)
    t(model_payload)
    assert sink["metadata"]["id_pair"] == list(id_pair)
    assert sink["metadata"]["goal_order"]["presented_order"] == list(reversed(id_pair))


def test_no_door_intermediate_candidates_reappear():
    state, obs, payload, model_payload = _base_payload()
    id_pair = ID_PAIRS[0]
    sink = {}
    t = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink)
    result = t(model_payload)
    move_targets = result["legal_actions_compact"]["move"]
    assert not any(m.startswith("door_") for m in move_targets)
    assert set(move_targets) == set(id_pair)


def test_model_response_transform_hook_defaults_to_none():
    config = LLMControllerConfig()
    assert config.model_response_transform is None


def test_token_shape_audit_reports_equal_length():
    audit = token_shape_audit()
    assert len(audit["ids"]) == len(ID_PAIRS) * 2
    assert audit["all_same_length"] is True
