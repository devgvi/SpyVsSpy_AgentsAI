"""M5F.3A.4: network-free tests for candidate-representation control
(`src/evaluation/representation_control.py`) - spec section 42's required
coverage. No live calls."""

from __future__ import annotations

from src import constants
from src.evaluation.order_control import OrderPolicy
from src.evaluation.representation_control import (
    count_candidate_occurrences,
    decoupled_model_payload_transform,
    single_candidate_model_payload_transform,
)
from src.evaluation.symmetric_harness import alpha_beta_object_ids, build_symmetric_state
from src.llm_controller import compact_observation_for_model, enumerate_legal_actions, serialize_observation
from src.observation import build_observation


def _base_payload():
    state = build_symmetric_state(neutral_ids=True)
    obs = build_observation(state, constants.WHITE_SPY_ID)
    payload = serialize_observation(obs)
    model_payload = compact_observation_for_model(payload)
    candidates = alpha_beta_object_ids(True, neutral_ids=True)
    return state, obs, payload, model_payload, candidates


def test_current_preserves_historical_coupling():
    # CURRENT (no transform) never changes - the historical control.
    state, obs, payload, model_payload, candidates = _base_payload()
    assert model_payload["legal_actions_compact"]["move"] == list(payload["legal_actions_compact"]["move"]) if "legal_actions_compact" in payload else True
    # Candidates present in legal_actions_compact['move'] untouched.
    assert all(c in model_payload["legal_actions_compact"]["move"] for c in candidates)


def test_decoupled_objects_order_independent_of_legal_actions():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    transform = decoupled_model_payload_transform(
        legal_actions_policy=OrderPolicy.CANONICAL, room_objects_policy=OrderPolicy.REVERSED,
        trial_index=0, candidate_ids=candidates, metadata_sink=sink,
    )
    result = transform(model_payload)
    # legal_actions candidate order unchanged (canonical)...
    move = result["legal_actions_compact"]["move"]
    candidate_positions_in_move = [t for t in move if t in candidates]
    assert candidate_positions_in_move == list(candidates)
    # ...while room.objects WAS reordered.
    assert [o["id"] for o in result["room"]["objects"]] == list(reversed([o["id"] for o in model_payload["room"]["objects"]]))


def test_decoupled_legal_actions_order_independent():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    transform = decoupled_model_payload_transform(
        legal_actions_policy=OrderPolicy.REVERSED, room_objects_policy=OrderPolicy.CANONICAL,
        trial_index=0, candidate_ids=candidates, metadata_sink=sink,
    )
    result = transform(model_payload)
    move = result["legal_actions_compact"]["move"]
    candidate_positions_in_move = [t for t in move if t in candidates]
    assert candidate_positions_in_move == list(reversed(candidates))
    assert [o["id"] for o in result["room"]["objects"]] == [o["id"] for o in model_payload["room"]["objects"]]


def test_decoupled_room_objects_order_independent():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    transform = decoupled_model_payload_transform(
        legal_actions_policy=OrderPolicy.CANONICAL, room_objects_policy=OrderPolicy.REVERSED,
        trial_index=0, candidate_ids=candidates, metadata_sink=sink,
    )
    result = transform(model_payload)
    # Non-candidate (door) entries in move stay in their original relative order.
    move = result["legal_actions_compact"]["move"]
    doors_in_move = [t for t in move if t not in candidates]
    original_doors_in_move = [t for t in model_payload["legal_actions_compact"]["move"] if t not in candidates]
    assert doors_in_move == original_doors_in_move


def test_changing_one_order_does_not_mutate_others():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    transform = decoupled_model_payload_transform(
        legal_actions_policy=OrderPolicy.REVERSED, room_objects_policy=OrderPolicy.CANONICAL,
        trial_index=0, candidate_ids=candidates, metadata_sink=sink,
    )
    result = transform(model_payload)
    # Everything outside legal_actions/room is untouched.
    for key in ("self", "time", "memory", "opponent", "mission", "communication"):
        assert result[key] == model_payload[key]


def test_single_candidate_view_contains_one_entry_per_candidate():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    transform = single_candidate_model_payload_transform(
        candidate_ids=candidates, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink,
    )
    result = transform(model_payload)
    assert len(result["candidate_targets"]) == 2
    assert {c["candidate_id"] for c in result["candidate_targets"]} == set(candidates)


def test_single_candidate_view_removes_duplicate_exposure():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    transform = single_candidate_model_payload_transform(
        candidate_ids=candidates, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink,
    )
    result = transform(model_payload)
    assert not any(t in candidates for t in result["legal_actions_compact"]["move"])
    assert not any(o.get("id") in candidates for o in result["room"]["objects"])
    for cid in candidates:
        assert count_candidate_occurrences(result, cid) == 1


def test_candidate_ids_stable_under_order_reversal():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink_a = {}
    transform_a = single_candidate_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink_a)
    sink_b = {}
    transform_b = single_candidate_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.REVERSED, trial_index=0, metadata_sink=sink_b)
    result_a = transform_a(model_payload)
    result_b = transform_b(model_payload)
    ids_a = {c["candidate_id"] for c in result_a["candidate_targets"]}
    ids_b = {c["candidate_id"] for c in result_b["candidate_targets"]}
    assert ids_a == ids_b == set(candidates)


def test_same_canonical_legal_action_set_across_representations():
    # The TRUE legal-action set (used for validation) never changes -
    # only the DISPLAY (model_payload) does. observation_payload's own
    # legal_actions field (what validation reads) is never touched by
    # either transform.
    state, obs, payload, model_payload, candidates = _base_payload()
    canonical_legal = enumerate_legal_actions(obs)
    sink = {}
    transform = single_candidate_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.REVERSED, trial_index=0, metadata_sink=sink)
    transform(model_payload)  # mutating the display copy only
    still_canonical_legal = enumerate_legal_actions(obs)
    assert canonical_legal == still_canonical_legal
    assert all((a["action"], a.get("target_id")) in [(x["action"], x.get("target_id")) for x in still_canonical_legal] for a in canonical_legal)


def test_gamestate_unchanged_by_representation_transforms():
    state, obs, payload, model_payload, candidates = _base_payload()
    original_level_objects = list(state.level.objects)
    sink = {}
    transform = decoupled_model_payload_transform(
        legal_actions_policy=OrderPolicy.REVERSED, room_objects_policy=OrderPolicy.REVERSED,
        trial_index=0, candidate_ids=candidates, metadata_sink=sink,
    )
    transform(model_payload)
    assert state.level.objects == original_level_objects


def test_counterbalancing_deterministic():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink_even = {}
    t_even = single_candidate_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.COUNTERBALANCED, trial_index=0, metadata_sink=sink_even)
    r_even = t_even(model_payload)
    sink_odd = {}
    t_odd = single_candidate_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.COUNTERBALANCED, trial_index=1, metadata_sink=sink_odd)
    r_odd = t_odd(model_payload)
    even_order = [c["candidate_id"] for c in r_even["candidate_targets"]]
    odd_order = [c["candidate_id"] for c in r_odd["candidate_targets"]]
    assert even_order == list(candidates)
    assert odd_order == list(reversed(candidates))


def test_order_telemetry_emitted():
    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    transform = decoupled_model_payload_transform(
        legal_actions_policy=OrderPolicy.REVERSED, room_objects_policy=OrderPolicy.CANONICAL,
        trial_index=2, candidate_ids=candidates, metadata_sink=sink,
    )
    transform(model_payload)
    assert "metadata" in sink
    assert sink["metadata"]["representation_mode"] == "DECOUPLED"
    assert "legal_actions_order" in sink["metadata"]
    assert "room_objects_order" in sink["metadata"]


def test_prompt_snapshots_secret_safe():
    import json

    state, obs, payload, model_payload, candidates = _base_payload()
    sink = {}
    transform = single_candidate_model_payload_transform(candidate_ids=candidates, order_policy=OrderPolicy.CANONICAL, trial_index=0, metadata_sink=sink)
    result = transform(model_payload)
    serialized = json.dumps(result, default=str)
    assert "AWS_BEARER_TOKEN_BEDROCK" not in serialized
    assert "bedrock-api-key" not in serialized


def test_model_payload_transform_hook_defaults_to_none_no_behavior_change():
    from src.llm_controller import LLMControllerConfig

    config = LLMControllerConfig()
    assert config.model_payload_transform is None
