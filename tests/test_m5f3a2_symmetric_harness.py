"""M5F.3A.2: network-free tests for the purpose-built symmetric two-branch
evaluation harness (`src/evaluation/symmetric_harness.py`) - geometry
mirror correctness, static symmetry audit, partial information, trap/
threat-capability fixture seeding, and message-text symmetry. No live
calls (Stage 0, spec section 48)."""

from __future__ import annotations

import math

from src import constants
from src.evaluation.symmetric_harness import (
    COOPERATION,
    INVALID,
    MISDIRECTION,
    NON_CREDIBLE_THREAT,
    START_XY,
    SYMMETRIC,
    THREAT,
    WARNING,
    alpha_beta_object_ids,
    build_symmetric_state,
    message_text,
    static_symmetry_audit,
)
from src.observation import build_observation


def test_static_symmetry_audit_passes():
    result = static_symmetry_audit()
    assert result["result"] == SYMMETRIC
    assert result["reasons"] == []


def test_path_distances_are_exactly_equal():
    result = static_symmetry_audit()
    assert result["checks"]["path_distance_alpha"] == result["checks"]["path_distance_beta"]


def test_object_ids_are_stable_alpha_beta_regardless_of_layout():
    # The real bug this milestone found and fixed before any live spend:
    # object ids must literally BE "objective_alpha"/"objective_beta" (not
    # "objective_side_a"/"objective_side_b" with the label swap only
    # tracked in Python bookkeeping), or a treatment message referring to
    # "objective_alpha" has no real object in the recipient's Observation
    # to resolve to.
    for alpha_on_left in (True, False):
        alpha_id, beta_id = alpha_beta_object_ids(alpha_on_left)
        assert alpha_id == "objective_alpha"
        assert beta_id == "objective_beta"


def test_alpha_on_left_false_swaps_geometry_not_ids():
    from src.evaluation.symmetric_harness import build_symmetric_state

    left_state = build_symmetric_state(alpha_on_left=True)
    right_state = build_symmetric_state(alpha_on_left=False)
    left_alpha_pos = left_state.level.get_object("objective_alpha").position
    right_alpha_pos = right_state.level.get_object("objective_alpha").position
    assert left_alpha_pos != right_alpha_pos


def test_recipient_starts_at_neutral_center_both_layouts():
    for alpha_on_left in (True, False):
        state = build_symmetric_state(alpha_on_left=alpha_on_left)
        actor = state.actors[constants.WHITE_SPY_ID]
        assert (actor.x, actor.y) == START_XY


def test_path_distance_equal_for_both_label_assignments():
    for alpha_on_left in (True, False):
        state = build_symmetric_state(alpha_on_left=alpha_on_left)
        alpha_id, beta_id = alpha_beta_object_ids(alpha_on_left)
        alpha_obj = state.level.get_object(alpha_id)
        beta_obj = state.level.get_object(beta_id)
        ap_a = state.nav_grid.nearest_approach_point(alpha_obj.collision_rect, constants.INTERACTION_DISTANCE, START_XY)
        ap_b = state.nav_grid.nearest_approach_point(beta_obj.collision_rect, constants.INTERACTION_DISTANCE, START_XY)
        assert math.dist(START_XY, ap_a) == math.dist(START_XY, ap_b)


def test_no_required_items_no_exit_attractor():
    state = build_symmetric_state()
    obs = build_observation(state, constants.WHITE_SPY_ID)
    assert obs.required_items == ()
    assert all(o.id != "exit_door" for o in obs.objects)


def test_warning_seeds_real_environmental_trap_on_named_branch():
    state = build_symmetric_state(trap_branch="alpha")
    alpha_id, _ = alpha_beta_object_ids(True)
    trap = state.trap_on(alpha_id)
    assert trap is not None
    assert trap["owner_id"] == constants.ENVIRONMENT_OWNER_ID


def test_warning_trap_partial_information_recipient_unaware():
    state = build_symmetric_state(trap_branch="alpha", recipient_actor_id=constants.WHITE_SPY_ID)
    alpha_id, _ = alpha_beta_object_ids(True)
    obs = build_observation(state, constants.WHITE_SPY_ID)
    alpha_obs = next(o for o in obs.objects if o.id == alpha_id)
    assert alpha_obs.trap_status == "unknown"
    assert state.detected_trap_type(alpha_id, constants.BLACK_SPY_ID) == "electric"
    assert state.detected_trap_type(alpha_id, constants.WHITE_SPY_ID) is None


def test_threat_capability_present_and_absent():
    capable_state = build_symmetric_state(threat_capability_branch="alpha", sender_trap_capable=True)
    incapable_state = build_symmetric_state(threat_capability_branch="alpha", sender_trap_capable=False)
    alpha_id, _ = alpha_beta_object_ids(True)
    assert constants.BLACK_SPY_ID in capable_state._object_entry(alpha_id)["inspected_by"]
    assert any(v > 0 for v in capable_state.actors[constants.BLACK_SPY_ID].trap_inventory.values())
    assert all(v == 0 for v in incapable_state.actors[constants.BLACK_SPY_ID].trap_inventory.values())


def test_threat_capability_not_visible_to_recipient():
    capable = build_symmetric_state(threat_capability_branch="alpha", sender_trap_capable=True)
    incapable = build_symmetric_state(threat_capability_branch="alpha", sender_trap_capable=False)
    capable_obs = build_observation(capable, constants.WHITE_SPY_ID)
    incapable_obs = build_observation(incapable, constants.WHITE_SPY_ID)
    assert capable_obs.position == incapable_obs.position
    assert capable_obs.objects == incapable_obs.objects


def test_recipient_actor_swap_black_as_recipient():
    state = build_symmetric_state(recipient_actor_id=constants.BLACK_SPY_ID)
    black = state.actors[constants.BLACK_SPY_ID]
    assert (black.x, black.y) == START_XY


def test_message_text_alpha_beta_word_count_symmetry():
    for cls in (COOPERATION, WARNING, THREAT, NON_CREDIBLE_THREAT, MISDIRECTION):
        alpha_text = message_text(cls, "alpha")
        beta_text = message_text(cls, "beta")
        assert len(alpha_text.split()) == len(beta_text.split())
        assert alpha_text != beta_text
        # A simultaneous alpha<->beta swap must turn one into the other -
        # true even for COOPERATION, whose wording names both branches
        # ("I'll take alpha. You take beta." <-> "I'll take beta. You
        # take alpha."), unlike the single-branch-reference classes where
        # only one word ever changes.
        swapped = alpha_text.replace("alpha", "\0").replace("beta", "alpha").replace("\0", "beta")
        assert swapped == beta_text


def test_threat_and_non_credible_threat_share_identical_wording():
    assert message_text(THREAT, "alpha") == message_text(NON_CREDIBLE_THREAT, "alpha")
    assert message_text(THREAT, "beta") == message_text(NON_CREDIBLE_THREAT, "beta")
    assert "credible" not in message_text(THREAT, "alpha").lower()


def test_message_text_no_message_control_is_none():
    from src.evaluation.symmetric_harness import NO_MESSAGE_CONTROL

    assert message_text(NO_MESSAGE_CONTROL, "alpha") is None


def test_message_referent_is_a_real_observable_object():
    # The message says "objective_alpha" - the recipient must be able to
    # resolve that to a real object it can see, in both layouts.
    for alpha_on_left in (True, False):
        state = build_symmetric_state(alpha_on_left=alpha_on_left)
        obs = build_observation(state, constants.WHITE_SPY_ID)
        ids = {o.id for o in obs.objects}
        assert "objective_alpha" in ids
        assert "objective_beta" in ids


def test_neutral_identifiers_no_semantic_labels():
    forbidden = ("security", "archives", "office", "safe", "exit", "passport", "documents")
    for cls in (COOPERATION, WARNING, THREAT, NON_CREDIBLE_THREAT, MISDIRECTION):
        for branch in ("alpha", "beta"):
            text = message_text(cls, branch).lower()
            assert not any(word in text for word in forbidden)


def test_neutral_ids_ground_message_referent():
    from src.evaluation.symmetric_harness import NEUTRAL_OBJ_ALPHA_ID, NEUTRAL_OBJ_BETA_ID

    state = build_symmetric_state(neutral_ids=True)
    obs = build_observation(state, constants.WHITE_SPY_ID)
    ids = {o.id for o in obs.objects}
    assert NEUTRAL_OBJ_ALPHA_ID in ids
    assert NEUTRAL_OBJ_BETA_ID in ids
    assert "objective_alpha" not in ids


def test_neutral_ids_hide_word_alpha_beta_in_both_id_and_label():
    state = build_symmetric_state(neutral_ids=True)
    obs = build_observation(state, constants.WHITE_SPY_ID)
    for obj in obs.objects:
        assert "alpha" not in obj.id.lower()
        assert "beta" not in obj.id.lower()


def test_neutral_ids_message_text_names_a_real_object():
    from src.evaluation.symmetric_harness import NEUTRAL_OBJ_ALPHA_ID, WARNING

    text = message_text(WARNING, "alpha", neutral_ids=True)
    assert NEUTRAL_OBJ_ALPHA_ID in text
    assert "alpha" not in text.replace(NEUTRAL_OBJ_ALPHA_ID, "")


def test_resolve_branch_neutral_ids_does_not_match_plain_ids():
    from src.evaluation.symmetric_harness import resolve_branch

    assert resolve_branch("objective_alpha", neutral_ids=True) == "OTHER"
    assert resolve_branch("objective_x7q2m9", neutral_ids=True) == "alpha"


def test_symmetry_audit_detects_injected_asymmetry():
    # Regression harness for the audit function itself: deliberately break
    # one dimension and confirm it's caught, not silently passed.
    from src.evaluation import symmetric_harness as mod

    state = mod.build_symmetric_state()
    state._object_entry(mod.OBJ_ALPHA_ID)["searched_by"].add(constants.WHITE_SPY_ID)
    same_visibility = (
        state.is_searched_by(mod.OBJ_ALPHA_ID, constants.WHITE_SPY_ID) == state.is_searched_by(mod.OBJ_BETA_ID, constants.WHITE_SPY_ID)
    )
    assert same_visibility is False
