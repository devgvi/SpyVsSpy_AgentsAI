"""M5F.3A.7: network-free tests for the fully counterbalanced social-
influence replication harness (`src/evaluation/counterbalanced_social.py`)
- spec section 33's required coverage, exercised via a MockModelClient
(no live Bedrock calls, but the real end-to-end trial-running pipeline)."""

from __future__ import annotations

from src import constants
from src.evaluation.counterbalanced_social import (
    COOPERATION,
    CREDIBLE_THREAT,
    MISDIRECTION,
    NON_CREDIBLE_THREAT,
    TRUE_WARNING,
    message_text,
    run_counterbalanced_trial,
)
from src.evaluation.goal_identity import ID_PAIRS, MAPPING_1, MAPPING_2, build_goal_mapping
from src.evaluation.order_control import OrderPolicy
from src.evaluation.symmetric_harness import alpha_beta_object_ids
from src.llm.clients.mock_client import MockModelClient
from src.llm.personality import OPPORTUNIST

ID_PAIR = ID_PAIRS[0]


def test_target_role_mirror_generation():
    x, y = alpha_beta_object_ids(True, neutral_ids=True)
    mapping = build_goal_mapping(ID_PAIR, (x, y), MAPPING_1)
    target_to_id = {v: k for k, v in mapping.items()}
    text_x = message_text(COOPERATION, target_to_id[x], target_to_id[y])
    text_y = message_text(COOPERATION, target_to_id[y], target_to_id[x])
    assert text_x != text_y
    assert len(text_x.split()) == len(text_y.split())


def test_id_mapping_balance():
    x, y = alpha_beta_object_ids(True, neutral_ids=True)
    m1 = build_goal_mapping(ID_PAIR, (x, y), MAPPING_1)
    m2 = build_goal_mapping(ID_PAIR, (x, y), MAPPING_2)
    assert m1[ID_PAIR[0]] != m2[ID_PAIR[0]]
    assert {m1[ID_PAIR[0]], m2[ID_PAIR[0]]} == {x, y}


def test_position_balance():
    client_first = MockModelClient(response={"action": "move", "target_id": ID_PAIR[0], "reason": "x"})
    trial_first = run_counterbalanced_trial(COOPERATION, "X", MAPPING_1, OrderPolicy.CANONICAL, 1, OPPORTUNIST, client_first, id_pair=ID_PAIR)
    client_second = MockModelClient(response={"action": "move", "target_id": ID_PAIR[0], "reason": "x"})
    trial_second = run_counterbalanced_trial(COOPERATION, "X", MAPPING_1, OrderPolicy.REVERSED, 1, OPPORTUNIST, client_second, id_pair=ID_PAIR)
    assert trial_first["treatment_target_position"] != trial_second["treatment_target_position"]


def test_message_x_y_symmetry_all_classes():
    x, y = alpha_beta_object_ids(True, neutral_ids=True)
    mapping = build_goal_mapping(ID_PAIR, (x, y), MAPPING_1)
    target_to_id = {v: k for k, v in mapping.items()}
    for cls in (COOPERATION, TRUE_WARNING, CREDIBLE_THREAT, NON_CREDIBLE_THREAT, MISDIRECTION):
        text_x = message_text(cls, target_to_id[x], target_to_id[y])
        text_y = message_text(cls, target_to_id[y], target_to_id[x])
        assert len(text_x.split()) == len(text_y.split())
        assert text_x != text_y


def test_credible_capability_present():
    client = MockModelClient(response={"action": "wait", "reason": "checking"})
    trial = run_counterbalanced_trial(CREDIBLE_THREAT, "X", MAPPING_1, OrderPolicy.CANONICAL, 1, OPPORTUNIST, client, id_pair=ID_PAIR)
    assert trial["message_text"] is not None
    # Build the world directly to check the underlying capability fact.
    from src.evaluation.symmetric_harness import build_symmetric_state

    state = build_symmetric_state(neutral_ids=True, threat_capability_branch="alpha", sender_trap_capable=True)
    assert any(v > 0 for v in state.actors[constants.BLACK_SPY_ID].trap_inventory.values())


def test_non_credible_capability_absent():
    from src.evaluation.symmetric_harness import build_symmetric_state

    state = build_symmetric_state(neutral_ids=True, threat_capability_branch="alpha", sender_trap_capable=False)
    assert all(v == 0 for v in state.actors[constants.BLACK_SPY_ID].trap_inventory.values())


def test_warning_trap_hidden_from_recipient():
    from src.evaluation.symmetric_harness import build_symmetric_state
    from src.observation import build_observation

    state = build_symmetric_state(neutral_ids=True, trap_branch="alpha")
    obs = build_observation(state, constants.WHITE_SPY_ID)
    x, _ = alpha_beta_object_ids(True, neutral_ids=True)
    obj = next(o for o in obs.objects if o.id == x)
    assert obj.trap_status == "unknown"
    assert state.detected_trap_type(x, constants.BLACK_SPY_ID) is not None
    assert state.detected_trap_type(x, constants.WHITE_SPY_ID) is None


def test_misdirection_contradicts_sender_known_trap_state():
    from src.evaluation.symmetric_harness import build_symmetric_state

    state = build_symmetric_state(neutral_ids=True, trap_branch="alpha")
    x, _ = alpha_beta_object_ids(True, neutral_ids=True)
    assert state.trap_on(x) is not None
    assert state.detected_trap_type(x, constants.BLACK_SPY_ID) is not None
    text = message_text(MISDIRECTION, "some_opaque_id", "other_id")
    assert "clear" in text.lower()


def test_goal_resolves_to_correct_world_target():
    x, y = alpha_beta_object_ids(True, neutral_ids=True)
    for role, expected in (("X", x), ("Y", y)):
        mapping = build_goal_mapping(ID_PAIR, (x, y), MAPPING_1)
        target_to_id = {v: k for k, v in mapping.items()}
        opaque = target_to_id[expected]
        client = MockModelClient(response={"action": "move", "target_id": opaque, "reason": "resolve test"})
        trial = run_counterbalanced_trial(COOPERATION, role, MAPPING_1, OrderPolicy.CANONICAL, 1, OPPORTUNIST, client, id_pair=ID_PAIR)
        assert trial["final_target"] == expected


def test_telemetry_captures_all_balancing_dimensions():
    client = MockModelClient(response={"action": "wait", "reason": "telemetry check"})
    trial = run_counterbalanced_trial(TRUE_WARNING, "Y", MAPPING_2, OrderPolicy.REVERSED, 3, OPPORTUNIST, client, id_pair=ID_PAIR)
    for key in (
        "treatment_target_role", "goal_order_policy", "model_goal_id_for_X", "model_goal_id_for_Y",
        "treatment_target_world_id", "treatment_target_model_id", "treatment_target_position",
        "selected_model_goal_id", "selected_world_target",
    ):
        assert key in trial


def test_no_message_control_has_no_text():
    client = MockModelClient(response={"action": "wait", "reason": "x"})
    trial = run_counterbalanced_trial("NO_MESSAGE_CONTROL", "X", MAPPING_1, OrderPolicy.CANONICAL, 1, OPPORTUNIST, client, id_pair=ID_PAIR)
    assert trial["message_text"] is None
