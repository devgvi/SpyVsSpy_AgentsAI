"""M5F.3A.3: network-free tests for the evaluation-only ordering control
utility (`src/evaluation/order_control.py`) - spec section 60's required
coverage. No live calls."""

from __future__ import annotations

from src import constants
from src.evaluation.order_control import (
    OrderPolicy,
    permute_observation_messages,
    permute_observation_objects,
    semantically_equivalent,
)
from src.evaluation.symmetric_harness import build_symmetric_state
from src.observation import build_observation


def _base_observation():
    state = build_symmetric_state(neutral_ids=True)
    return state, build_observation(state, constants.WHITE_SPY_ID)


def test_canonical_ordering_preserved():
    state, obs = _base_observation()
    permuted, meta = permute_observation_objects(obs, OrderPolicy.CANONICAL, trial_index=0)
    assert [o.id for o in permuted.objects] == [o.id for o in obs.objects]
    assert meta["policy"] == "CANONICAL"


def test_reversed_object_order_changes_serialization_only():
    state, obs = _base_observation()
    permuted, meta = permute_observation_objects(obs, OrderPolicy.REVERSED, trial_index=0)
    assert [o.id for o in permuted.objects] == list(reversed([o.id for o in obs.objects]))
    # Every other field is untouched.
    assert permuted.position == obs.position
    assert permuted.current_room == obs.current_room
    assert permuted.required_items == obs.required_items


def test_counterbalanced_order_deterministic_by_trial_index():
    state, obs = _base_observation()
    even, _ = permute_observation_objects(obs, OrderPolicy.COUNTERBALANCED, trial_index=0)
    odd, _ = permute_observation_objects(obs, OrderPolicy.COUNTERBALANCED, trial_index=1)
    even_again, _ = permute_observation_objects(obs, OrderPolicy.COUNTERBALANCED, trial_index=2)
    assert [o.id for o in even.objects] == [o.id for o in obs.objects]
    assert [o.id for o in odd.objects] == list(reversed([o.id for o in obs.objects]))
    assert [o.id for o in even_again.objects] == [o.id for o in even.objects]


def test_semantic_facts_identical_under_permutation():
    state, obs = _base_observation()
    permuted, _ = permute_observation_objects(obs, OrderPolicy.REVERSED, trial_index=0)
    assert semantically_equivalent(obs, permuted)


def test_semantic_equivalence_rejects_a_real_difference():
    state, obs = _base_observation()
    _, other_obs = _base_observation()
    other_state = build_symmetric_state(neutral_ids=True, alpha_on_left=False)
    different_obs = build_observation(other_state, constants.WHITE_SPY_ID)
    assert semantically_equivalent(obs, different_obs) is False


def test_engine_state_unchanged_by_permutation():
    state, obs = _base_observation()
    original_level_objects = list(state.level.objects)
    original_object_states = {k: dict(v) for k, v in state.object_states.items()}
    permute_observation_objects(obs, OrderPolicy.REVERSED, trial_index=0)
    assert state.level.objects == original_level_objects
    assert {k: dict(v) for k, v in state.object_states.items()} == original_object_states


def test_legal_action_membership_unchanged_when_only_objects_permuted():
    from src.llm_controller import enumerate_legal_actions

    state, obs = _base_observation()
    permuted, _ = permute_observation_objects(obs, OrderPolicy.REVERSED, trial_index=0)
    canonical_legal = enumerate_legal_actions(obs)
    permuted_legal = enumerate_legal_actions(permuted)
    canonical_set = {(a["action"], a.get("target_id")) for a in canonical_legal}
    permuted_set = {(a["action"], a.get("target_id")) for a in permuted_legal}
    assert canonical_set == permuted_set  # same actions/targets available
    assert len(canonical_legal) == len(permuted_legal)  # same cardinality


def test_order_metadata_emitted():
    state, obs = _base_observation()
    _, meta = permute_observation_objects(obs, OrderPolicy.COUNTERBALANCED, trial_index=3)
    for key in ("structure_name", "policy", "trial_index", "canonical_order", "presented_order"):
        assert key in meta
    assert meta["structure_name"] == "objects"
    assert meta["trial_index"] == 3


def test_alpha_beta_each_first_equally_under_counterbalancing():
    state, obs = _base_observation()
    first_ids = []
    for trial_index in range(10):
        permuted, _ = permute_observation_objects(obs, OrderPolicy.COUNTERBALANCED, trial_index=trial_index)
        first_ids.append(permuted.objects[0].id)
    counts = {i: first_ids.count(i) for i in set(first_ids)}
    assert len(counts) == 2
    assert all(c == 5 for c in counts.values())


def test_message_order_reversal_changes_only_message_ordering():
    state, obs = _base_observation()
    # Two messages seeded directly into engine-truth state (matching this
    # project's established pattern for controlled fixtures) rather than
    # via two back-to-back apply_action(SEND_MESSAGE) calls, which would
    # need an intervening state.update() to clear the first send's own
    # action_cooldown before a second send could execute.
    state.messages.append(
        {"message_id": "msg-1", "sender_actor_id": constants.BLACK_SPY_ID, "recipient_actor_id": constants.WHITE_SPY_ID, "game_time": 0.0, "text": "first message"}
    )
    state.messages.append(
        {"message_id": "msg-2", "sender_actor_id": constants.BLACK_SPY_ID, "recipient_actor_id": constants.WHITE_SPY_ID, "game_time": 1.0, "text": "second message"}
    )
    obs_with_messages = build_observation(state, constants.WHITE_SPY_ID)
    assert len(obs_with_messages.messages_received_recently) == 2

    permuted, meta = permute_observation_messages(obs_with_messages, OrderPolicy.REVERSED, trial_index=0)
    assert [m["message_id"] for m in permuted.messages_received_recently] == list(
        reversed([m["message_id"] for m in obs_with_messages.messages_received_recently])
    )
    assert [o.id for o in permuted.objects] == [o.id for o in obs_with_messages.objects]  # objects untouched
    assert meta["structure_name"] == "messages_received_recently"


def test_no_secret_leakage_through_order_metadata():
    state, obs = _base_observation()
    _, meta = permute_observation_objects(obs, OrderPolicy.REVERSED, trial_index=0)
    serialized = str(meta)
    assert "AWS_BEARER_TOKEN_BEDROCK" not in serialized
    assert "bedrock-api-key" not in serialized
