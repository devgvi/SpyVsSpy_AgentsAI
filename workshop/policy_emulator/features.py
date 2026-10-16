"""PolicyFeatureExtractorV1: compact, observable-only features of one decision.

Input: the annotated obs-v1 payload - exactly the dict the live LLM controller
builds (`serialize_observation(obs, live_state=True, live_workshop=True)` +
`annotate_live_payload`) and logs as `llm_decision.observation`. The same
function therefore runs on a logged Phase-A record, on a rebuilt Stage-2
snapshot and on a live Observation inside the SurrogateController.

Only payload fields the model itself sees are read (READ_PATHS). `self.knowledge`,
`communication`, `time` (world-clock artefacts under the barrier) and the raw
positions are never read; nothing here touches GameState.

`recent_actions` (most recent first) is the agent's own recent decisions, the
`strategy_memory.recent_decisions` the model sees. Phase-A telemetry does not
log strategy memory, so for full matches it is rebuilt from the same actor's
earlier decisions in the match.
"""
from __future__ import annotations

from typing import Any

from workshop.policy_emulator.schema import FEATURE_SCHEMA_VERSION, code_from_visible_puzzle, legal_targets, object_types

# (top-level key, sub-key) pairs the extractor reads; the leakage test pins this set
READ_PATHS = {
    ("time_budget", "time_remaining_s"), ("time_budget", "time_limit_s"), ("time_budget", "urgency"),
    ("time_budget", "required_items_missing"), ("time_budget", "exit"),
    ("mission", "ready_to_exit"), ("mission", "required_items"), ("mission", "have"),
    ("self", "inventory"), ("self", "lives"), ("self", "traps_triggered_on_you"), ("self", "room"),
    ("self", "trap_inventory"), ("self", "own_traps"), ("self", "near_object_id"),
    ("encounter", "alert"), ("encounter", "opponent_room"), ("encounter", "opponent_distance"),
    ("encounter", "opponent_lives"), ("encounter", "fight_possible_now"),
    ("opponent", "visible"), ("opponent", "last_seen"),
    ("legal_actions", None), ("room", "objects"), ("exploration", "current_room"), ("exploration", "rooms"),
    ("memory", "visited_rooms"), ("memory", "known_locked_objects"),
    ("puzzle", "safe_code_known"), ("puzzle", "clue"), ("puzzle", "inscriptions_you_have_read"),
}

# name, group, type, source, note
FEATURES: list[tuple[str, str, str, str]] = [
    ("time_remaining_seconds", "mission", "int", "time_budget.time_remaining_s"),
    ("time_remaining_fraction", "mission", "float", "time_budget.time_remaining_s / time_limit_s"),
    ("time_pressure", "mission", "category", "time_budget.urgency (relaxed / tight / critical)"),
    ("objectives_remaining_count", "mission", "int", "len(time_budget.required_items_missing)"),
    ("has_archive_key", "mission", "bool", "self.inventory"),
    ("has_security_keycard", "mission", "bool", "self.inventory"),
    ("has_safe_code", "mission", "bool", "puzzle.safe_code_known"),
    ("safe_code_derivable", "mission", "bool", "puzzle.clue + inscriptions_you_have_read"),
    ("has_secret_documents", "mission", "bool", "self.inventory"),
    ("has_passport", "mission", "bool", "self.inventory"),
    ("mission_complete", "mission", "bool", "mission.ready_to_exit"),
    ("exit_known", "mission", "bool", "time_budget.exit.known"),
    ("exit_reachable_if_observable", "mission", "bool", "time_budget.exit.eta_s is a number"),
    ("hearts", "self", "int", "self.lives"),
    ("health_band", "self", "category", "self.lives: 3 high, 2 mid, 1 low"),
    ("trap_hits", "self", "int", "self.traps_triggered_on_you"),
    ("current_room", "self", "category", "self.room"),
    ("trap_inventory_total", "self", "int", "sum(self.trap_inventory)"),
    ("own_traps_armed", "self", "int", "len(self.own_traps)"),
    ("near_object_type", "self", "category", "type of self.near_object_id (room.objects), or none"),
    ("last_action_type", "self", "category", "recent_actions[0]"),
    ("opponent_visible", "opponent", "bool", "opponent.visible"),
    ("opponent_alert_present", "opponent", "bool", "encounter.alert present"),
    ("opponent_same_room", "opponent", "bool", "encounter.opponent_room == self.room"),
    ("opponent_distance_bucket", "opponent", "category", "encounter.opponent_distance: close <=140 / near <=300 / far"),
    ("last_seen_opponent_room", "opponent", "category", "opponent.last_seen.room, or never"),
    ("opponent_hearts", "opponent", "int|null", "encounter.opponent_lives (only while visible)"),
    ("life_balance", "opponent", "category", "self.lives vs encounter.opponent_lives: lead / equal / trail / unknown"),
    ("fight_possible_now", "opponent", "bool", "encounter.fight_possible_now"),
    ("inspect_legal", "opportunity", "bool", "legal_actions"),
    ("search_legal", "opportunity", "bool", "legal interact on a non-door object"),
    ("door_interact_legal", "opportunity", "bool", "legal interact on a door"),
    ("place_trap_legal", "opportunity", "bool", "legal_actions"),
    ("disarm_legal", "opportunity", "bool", "legal_actions"),
    ("approach_opponent_legal", "opportunity", "bool", "legal_actions"),
    ("avoid_opponent_legal", "opportunity", "bool", "legal_actions"),
    ("enter_code_legal", "opportunity", "bool", "legal_actions"),
    ("exit_interact_legal", "opportunity", "bool", "legal interact on exit_door"),
    ("fight_related_option_present", "opportunity", "bool", "approach or avoid legal"),
    ("legal_action_count", "opportunity", "int", "len(legal_actions)"),
    ("move_target_count", "opportunity", "int", "legal move targets"),
    ("room_status", "opportunity", "category", "exploration.rooms[current].status"),
    ("room_unsearched_count", "opportunity", "int", "exploration.rooms[current].unsearched"),
    ("room_uninspected_count", "opportunity", "int", "exploration.rooms[current].uninspected"),
    ("room_inspected_unsearched_count", "opportunity", "int", "unsearched but already inspected, current room"),
    ("room_untraversed_exits_count", "opportunity", "int", "exploration.rooms[current].untraversed_exits"),
    ("known_unsearched_elsewhere_count", "opportunity", "int", "unsearched objects in other visited rooms"),
    ("visited_room_count", "memory", "int", "memory.visited_rooms"),
    ("known_locked_object_count", "memory", "int", "memory.known_locked_objects"),
    ("recent_action_1", "memory", "category", "recent_actions[0] (strategy_memory.recent_decisions)"),
    ("recent_action_2", "memory", "category", "recent_actions[1]"),
    ("recent_action_3", "memory", "category", "recent_actions[2]"),
    ("recent_trap_attempt", "memory", "bool", "place_trap among recent_actions[0:3]"),
    ("opponent_seen_before", "memory", "bool", "opponent.last_seen is set"),
]

# requested by the spec (section 8) but not observable in the logged payload
UNAVAILABLE = {
    "currently_moving": "a decision is requested only when no action is executing; the payload has no movement state",
    "currently_interacting": "same: the payload has no in-progress interaction state",
    "last_seen_opponent_age": "opponent.last_seen.game_time is world time, not mission time; the two differ under some clocks",
    "nearby_interactable_count": "the payload exposes one object in interaction range at most (near_object_type)",
    "nearby_searched_trappable_count": "room.objects carries no room membership for searched objects",
    "recent_fight": "strategy_memory.recent_events is not logged in Phase-A telemetry",
    "recent_life_loss": "strategy_memory.recent_events is not logged in Phase-A telemetry",
    "recent_objective_progress": "strategy_memory.recent_events is not logged in Phase-A telemetry",
    "exit_legal": "move exit_door is always legal; exit_interact_legal records standing at the exit",
}

FEATURE_NAMES = tuple(name for name, *_ in FEATURES)


def distance_bucket(distance: Any) -> str:
    """The Phase-A report's buckets (cross_model_report._bucket), short names."""
    if not isinstance(distance, int | float) or isinstance(distance, bool):
        return "not_visible"
    return "close" if distance <= 140 else "near" if distance <= 300 else "far"


def health_band(lives: Any) -> str:
    if not isinstance(lives, int):
        return "unknown"
    return "high" if lives >= 3 else "mid" if lives == 2 else "low"


class PolicyFeatureExtractorV1:
    """Deterministic: the same payload (and recent actions) always gives the same dict."""

    version = FEATURE_SCHEMA_VERSION

    def extract(self, payload: dict[str, Any], recent_actions: list[str] | None = None) -> dict[str, Any]:
        tb = payload.get("time_budget") or {}
        mission = payload.get("mission") or {}
        me = payload.get("self") or {}
        enc = payload.get("encounter") or {}
        opp = payload.get("opponent") or {}
        exploration = payload.get("exploration") or {}
        memory = payload.get("memory") or {}
        puzzle = payload.get("puzzle") or {}
        legal = legal_targets(payload)
        types = object_types(payload)
        inventory = set(me.get("inventory") or [])
        remaining, limit = tb.get("time_remaining_s"), tb.get("time_limit_s")
        missing = tb.get("required_items_missing")
        if missing is None:
            missing = [i for i in mission.get("required_items") or [] if i not in (mission.get("have") or [])]
        exit_facts = tb.get("exit") or {}
        room = me.get("room")
        here = exploration.get("current_room") or room
        rooms = exploration.get("rooms") or {}
        room_facts = rooms.get(here) or {}
        unsearched = list(room_facts.get("unsearched") or [])
        uninspected = set(room_facts.get("uninspected") or [])
        lives = me.get("lives")
        opp_lives = enc.get("opponent_lives") if opp.get("visible") else None
        interact = [t for t in legal.get("interact", []) if t]
        recent = [a for a in (recent_actions or []) if a][:3]
        last_seen = opp.get("last_seen") or {}
        near = me.get("near_object_id")
        return {
            "time_remaining_seconds": remaining,
            "time_remaining_fraction": round(remaining / limit, 3) if remaining is not None and limit else None,
            "time_pressure": str(tb.get("urgency") or "unknown").lower(),
            "objectives_remaining_count": len(missing),
            "has_archive_key": "archive_key" in inventory,
            "has_security_keycard": "security_keycard" in inventory,
            "has_safe_code": bool(puzzle.get("safe_code_known")),
            "safe_code_derivable": code_from_visible_puzzle(payload) is not None,
            "has_secret_documents": "secret_documents" in inventory,
            "has_passport": "passport" in inventory,
            "mission_complete": bool(mission.get("ready_to_exit")),
            "exit_known": bool(exit_facts.get("known")),
            "exit_reachable_if_observable": isinstance(exit_facts.get("eta_s"), int | float),
            "hearts": lives,
            "health_band": health_band(lives),
            "trap_hits": me.get("traps_triggered_on_you") or 0,
            "current_room": room or "unknown",
            "trap_inventory_total": sum(v for v in (me.get("trap_inventory") or {}).values() if isinstance(v, int)),
            "own_traps_armed": len(me.get("own_traps") or []),
            "near_object_type": (types.get(near) or "object") if near else "none",
            "last_action_type": recent[0] if recent else "none",
            "opponent_visible": bool(opp.get("visible")),
            "opponent_alert_present": bool(enc.get("alert")),
            "opponent_same_room": bool(opp.get("visible")) and enc.get("opponent_room") == room,
            "opponent_distance_bucket": distance_bucket(enc.get("opponent_distance")),
            "last_seen_opponent_room": last_seen.get("room") or "never",
            "opponent_hearts": opp_lives,
            "life_balance": ("unknown" if not isinstance(lives, int) or not isinstance(opp_lives, int)
                             else "lead" if lives > opp_lives else "trail" if lives < opp_lives else "equal"),
            "fight_possible_now": bool(enc.get("fight_possible_now")),
            "inspect_legal": "inspect" in legal,
            "search_legal": any(t != "exit_door" and types.get(t) != "door" and not str(t).startswith("door_") for t in interact),
            "door_interact_legal": any(types.get(t) == "door" or str(t).startswith("door_") for t in interact),
            "place_trap_legal": "place_trap" in legal,
            "disarm_legal": "disarm" in legal,
            "approach_opponent_legal": "approach_opponent" in legal,
            "avoid_opponent_legal": "avoid_opponent" in legal,
            "enter_code_legal": "enter_safe_code" in legal,
            "exit_interact_legal": "exit_door" in interact,
            "fight_related_option_present": "approach_opponent" in legal or "avoid_opponent" in legal,
            "legal_action_count": sum(len(v) for v in legal.values()),
            "move_target_count": len(legal.get("move", [])),
            "room_status": str(room_facts.get("status") or "unknown").lower(),
            "room_unsearched_count": len(unsearched),
            "room_uninspected_count": len(uninspected),
            "room_inspected_unsearched_count": sum(1 for t in unsearched if t not in uninspected),
            "room_untraversed_exits_count": len(room_facts.get("untraversed_exits") or []),
            "known_unsearched_elsewhere_count": sum(len(r.get("unsearched") or []) for n, r in rooms.items() if n != here),
            "visited_room_count": len(memory.get("visited_rooms") or []),
            "known_locked_object_count": len(memory.get("known_locked_objects") or []),
            "recent_action_1": recent[0] if len(recent) > 0 else "none",
            "recent_action_2": recent[1] if len(recent) > 1 else "none",
            "recent_action_3": recent[2] if len(recent) > 2 else "none",
            "recent_trap_attempt": "place_trap" in recent,
            "opponent_seen_before": bool(last_seen),
        }


def feature_schema() -> dict[str, Any]:
    return {"feature_schema_version": FEATURE_SCHEMA_VERSION,
            "input": "annotated obs-v1 payload (serialize_observation(live_state, live_workshop) + annotate_live_payload)",
            "read_paths": sorted(f"{a}.{b}" if b else a for a, b in READ_PATHS),
            "features": [{"name": n, "group": g, "type": t, "source": s} for n, g, t, s in FEATURES],
            "unavailable": UNAVAILABLE}
