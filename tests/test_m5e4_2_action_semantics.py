"""M5E.4.2: door state/effect/traversal representation - the fix for the
M5E.4.1-traced defect (repeated INTERACT on an already-open door, because
nothing told the model INTERACT toggles the door itself while MOVE_TO is
what actually crosses it). Covers spec sections 36-37: field
serialization, closed/open-door affordances, Native/Strands semantic
equivalence, no hidden-state leakage, and that no existing engine/legal-
action behavior changed."""

from __future__ import annotations

from src.llm_controller import (
    NATIVE_CURRENT_REPRESENTATION,
    NATIVE_TOOL_LIKE_REPRESENTATION,
    _door_affordance,
    compact_observation_for_model,
    enumerate_legal_actions,
    format_legal_actions_tool_like,
    serialize_observation,
)
from src.observation import Observation, ObservedObject


def _obj(id, **overrides):
    base = dict(
        id=id, type="desk", label=id, position=(10.0, 10.0), collision_size=(20.0, 20.0),
        interactable=True, trappable=False, searched=False, inspected=False, trap_status="unknown",
        trap_type=None, trap_is_mine=False, door_open=None, connects=None, lock_type=None, locked=False, can_unlock=False,
    )
    base.update(overrides)
    return ObservedObject(**base)


def _door(id, door_open, connects=("corridor", "archives"), locked=False, can_unlock=False, **overrides):
    return _obj(
        id, type="door", interactable=False, door_open=door_open, connects=connects,
        lock_type=("keycard" if locked or can_unlock else None), locked=locked, can_unlock=can_unlock, **overrides,
    )


def _make_observation(**overrides):
    base = dict(
        actor_id="white", seed=1, position=(5.0, 5.0), facing="east", stunned=False, stunned_seconds=0.0,
        busy=False, busy_seconds=0.0, mission_time_remaining=100.0, mission_complete=False,
        inventory=frozenset(), knowledge=frozenset(), required_items=(), trap_inventory={},
        nearby_object_id=None, items_found_here=frozenset(), objects=(), nav_grid=None,
        log_event=lambda *a, **k: None, current_room="corridor", known_rooms=frozenset({"corridor"}),
        known_doors=frozenset(), known_locked_objects={}, known_item_locations={},
        visible_other_actor=False, opponent=None, last_seen_opponent=None,
        revealed_opponent=None, revealed_opponent_seconds=0.0,
    )
    base.update(overrides)
    return Observation(**base)


# -- _door_affordance (spec sections 8-11, 16) --------------------------------


def test_door_affordance_empty_for_non_door_objects():
    assert _door_affordance(_obj("desk_01")) == {}


def test_closed_unlocked_door_affordance():
    info = _door_affordance(_door("archives_door", door_open=False))
    assert info["traversable"] is False
    assert info["interact_effect"] == "open"
    assert info["connects"] == ["corridor", "archives"]


def test_open_door_affordance_interact_effect_is_close():
    """Spec section 9: if re-interacting with an open door would close
    it, the model must see that fact explicitly."""
    info = _door_affordance(_door("archives_door", door_open=True))
    assert info["traversable"] is True
    assert info["interact_effect"] == "close"


def test_locked_door_without_key_has_no_interact_effect():
    """INTERACT isn't legal here at all (see enumerate_legal_actions's
    lock gate) - no effect to advertise."""
    info = _door_affordance(_door("vault_door", door_open=False, locked=True, can_unlock=False))
    assert info["interact_effect"] is None
    assert info["traversable"] is False


def test_locked_door_with_key_effect_is_unlock_and_open():
    info = _door_affordance(_door("vault_door", door_open=False, locked=True, can_unlock=True))
    assert info["interact_effect"] == "unlock_and_open"


def test_door_without_known_connection_reports_none():
    info = _door_affordance(_door("mystery_door", door_open=False, connects=None))
    assert info["connects"] is None


# -- serialization (spec section 36) ------------------------------------------


def test_serialize_observation_includes_door_fields_in_room_objects():
    door = _door("archives_door", door_open=True)
    obs = _make_observation(objects=(door,), nearby_object_id="archives_door")
    payload = serialize_observation(obs)
    door_dict = next(o for o in payload["room"]["objects"] if o["id"] == "archives_door")
    assert door_dict["door_open"] is True
    assert door_dict["traversable"] is True
    assert door_dict["interact_effect"] == "close"
    assert door_dict["connects"] == ["corridor", "archives"]


def test_serialize_observation_never_adds_door_fields_to_non_door_objects():
    desk = _obj("desk_01")
    obs = _make_observation(objects=(desk,), nearby_object_id="desk_01")
    payload = serialize_observation(obs)
    desk_dict = next(o for o in payload["room"]["objects"] if o["id"] == "desk_01")
    assert "connects" not in desk_dict or desk_dict.get("connects") is None
    assert desk_dict.get("interact_effect") is None or "interact_effect" not in desk_dict


def test_compact_observation_carries_door_affordance_fields_through():
    door = _door("archives_door", door_open=False)
    obs = _make_observation(objects=(door,), nearby_object_id="archives_door")
    payload = serialize_observation(obs)
    compact = compact_observation_for_model(payload, NATIVE_CURRENT_REPRESENTATION)
    door_dict = next(o for o in compact["room"]["objects"] if o["id"] == "archives_door")
    assert door_dict["traversable"] is False
    assert door_dict["interact_effect"] == "open"
    assert door_dict["connects"] == ["corridor", "archives"]


# -- available_actions_text annotation (spec sections 11, 22) -----------------


def test_format_legal_actions_tool_like_annotates_door_targets():
    legal = [{"action": "interact", "target_id": "archives_door"}, {"action": "move", "target_id": "archives_door"}]
    door_info = {"archives_door": {"door_open": True, "traversable": True, "interact_effect": "close", "connects": ["corridor", "archives"]}}
    text = format_legal_actions_tool_like(legal, door_info=door_info)
    assert "interact_effect=close" in text
    assert "traversable" in text
    assert "corridor<->archives" in text


def test_format_legal_actions_tool_like_without_door_info_is_unchanged():
    """Regression: calling with no `door_info` (every pre-M5E.4.2 call
    site) must render exactly as before - bare target lines."""
    legal = [{"action": "move", "target_id": "desk_01"}]
    text = format_legal_actions_tool_like(legal)
    assert "- desk_01" in text
    assert "[" not in text


# -- Native/Strands semantic equivalence (spec section 13) -------------------


def test_native_current_and_tool_like_variants_expose_identical_door_affordance_facts():
    """Both representation variants (Native's plain compact JSON, and the
    tool-like text Strands/Native-tool-like both read from) must carry
    the *same* door_open/traversable/interact_effect/connects facts -
    formatting may differ, information content must not (spec section 13)."""
    door = _door("archives_door", door_open=True)
    obs = _make_observation(objects=(door,), nearby_object_id="archives_door")
    payload = serialize_observation(obs)

    variant_a = compact_observation_for_model(payload, NATIVE_CURRENT_REPRESENTATION)
    variant_b = compact_observation_for_model(payload, NATIVE_TOOL_LIKE_REPRESENTATION)

    door_a = next(o for o in variant_a["room"]["objects"] if o["id"] == "archives_door")
    door_b = next(o for o in variant_b["room"]["objects"] if o["id"] == "archives_door")
    # The room-object list itself (which both Strands and Native tool-like
    # read - Strands imports the identical serialize_observation/
    # compact_observation_for_model functions) is byte-identical regardless
    # of representation_variant; only available_actions_text differs.
    assert door_a == door_b
    assert "interact_effect=close" in variant_b["available_actions_text"]


# -- no hidden-state leakage (spec section 14) ---------------------------------


def test_door_affordance_never_exposes_a_room_the_actor_has_not_been_told_about():
    """`connects` only ever reflects `GameObject.connects`, the same
    static level geometry both actors could already see (this project has
    no fog-of-war on room geometry - see observation.py's own module
    docstring) - never something derived from the other actor's hidden
    state or an unvisited room's contents."""
    door = _door("archives_door", door_open=False, connects=("corridor", "archives"))
    obs = _make_observation(objects=(door,), known_rooms=frozenset({"corridor"}))
    payload = serialize_observation(obs)
    door_dict = next(o for o in payload["room"]["objects"] if o["id"] == "archives_door")
    # connects is present (static geometry) but nothing about *contents*
    # of the unvisited "archives" room leaks anywhere in this payload.
    assert door_dict["connects"] == ["corridor", "archives"]
    assert "archives" not in payload["memory"]["visited_rooms"]


# -- preserve existing engine/legal-action semantics (spec section 37) -------


def test_closed_unlocked_door_still_offers_interact_as_legal():
    door = _door("archives_door", door_open=False)
    obs = _make_observation(objects=(door,), nearby_object_id="archives_door")
    legal = enumerate_legal_actions(obs)
    assert ("interact", "archives_door") in {(e["action"], e.get("target_id")) for e in legal}


def test_open_door_remains_traversable_ie_move_still_legal():
    door = _door("archives_door", door_open=True)
    obs = _make_observation(objects=(door,), nearby_object_id="archives_door")
    legal = enumerate_legal_actions(obs)
    assert ("move", "archives_door") in {(e["action"], e.get("target_id")) for e in legal}


def test_interact_on_open_door_remains_legal_not_suppressed():
    """Spec section 21: closing an open door must remain a legitimate,
    legal action - this milestone only makes its *consequence* explicit,
    never forbids it."""
    door = _door("archives_door", door_open=True)
    obs = _make_observation(objects=(door,), nearby_object_id="archives_door")
    legal = enumerate_legal_actions(obs)
    assert ("interact", "archives_door") in {(e["action"], e.get("target_id")) for e in legal}


def test_locked_door_without_key_still_withholds_interact():
    door = _door("vault_door", door_open=False, locked=True, can_unlock=False)
    obs = _make_observation(objects=(door,), nearby_object_id="vault_door")
    legal = enumerate_legal_actions(obs)
    assert ("interact", "vault_door") not in {(e["action"], e.get("target_id")) for e in legal}


def test_no_automatic_traversal_is_introduced():
    """Spec section 19: legal_actions must still require the model to
    choose MOVE_TO separately - INTERACT never appears alongside an
    implicit move, and there is no new action type."""
    door = _door("archives_door", door_open=False)
    obs = _make_observation(objects=(door,), nearby_object_id="archives_door")
    legal = enumerate_legal_actions(obs)
    action_names = {e["action"] for e in legal}
    assert action_names <= {"wait", "move", "inspect", "interact", "place_trap", "disarm"}
