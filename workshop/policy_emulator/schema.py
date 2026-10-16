"""Versions, policy identities, action categories and target classes (Stage 3 v1).

Everything here reads only the model-visible payload (the annotated obs-v1
dict the LLM controller logs as `observation`): legal actions with targets,
room objects, exploration, mission, time budget, encounter. Never engine state.
"""
from __future__ import annotations

import re
from typing import Any

FEATURE_SCHEMA_VERSION = "policy-features-v1"
CANONICAL_SCHEMA_VERSION = "policy-data-v1"
NORMALIZATION_VERSION = "policy-normalize-v1"
TABLE_POLICY_VERSION = "table-v1"
FREQUENCY_POLICY_VERSION = "frequency-v1"
TARGET_POLICY_VERSION = "target-class-v1"
SURROGATE_RUNTIME = "surrogate"
SURROGATE_RUNTIME_VERSION = "surrogate-v1"

FULL_MATCH = "full_match"
CONTROLLED_SNAPSHOT = "controlled_snapshot"
SOURCE_TYPES = (FULL_MATCH, CONTROLLED_SNAPSHOT)

# engine action labels the policy predicts (the gameplay vocabulary of the live payload)
ACTIONS = ("wait", "move", "interact", "inspect", "place_trap", "disarm", "approach_opponent", "avoid_opponent",
           "enter_safe_code")
# actions whose target is the one object in interaction range (no choice to make)
NEAR_OBJECT_ACTIONS = ("interact", "inspect", "place_trap", "disarm", "enter_safe_code")
OPPONENT_ACTIONS = ("approach_opponent", "avoid_opponent")

# analysis categories (spec section 11); gameplay always uses the engine action
CATEGORIES = ("APPROACH", "AVOID", "INSPECT", "SEARCH", "PLACE_TRAP", "DISARM", "EXIT", "MOVE_PROGRESS", "WAIT", "OTHER")

# Phase-A lab model keys -> Stage 3 model names (policy identity) and display names
MODELS = {
    "sonnet": {"name": "sonnet55", "display": "Claude Sonnet 5.5", "model_id": "us.anthropic.claude-sonnet-5-5"},
    "gpt": {"name": "gpt61sol", "display": "GPT-6.1 Sol", "model_id": "us.openai.gpt-6.1-sol"},
}
MODEL_BY_NAME = {v["name"]: {**v, "lab_key": k} for k, v in MODELS.items()}
PERSONALITIES = ("racer", "cautious", "saboteur", "opportunist")


def policy_key(model_name: str, personality: str) -> str:
    """The CLI / directory form: gpt61sol:opportunist."""
    return f"{model_name}:{personality}"


def policy_id(model_name: str, personality: str, dataset: str) -> str:
    """The canonical record form: gpt61sol__opportunist__phase-a-v2."""
    return f"{model_name}__{personality}__{dataset}"


def parse_policy_key(key: str) -> tuple[str, str]:
    model, sep, personality = key.partition(":")
    if not sep or model not in MODEL_BY_NAME or personality not in PERSONALITIES:
        raise ValueError(f"unknown policy '{key}': expected <model>:<personality> with model in "
                         f"{sorted(MODEL_BY_NAME)} and personality in {list(PERSONALITIES)}")
    return model, personality


def all_policy_keys() -> list[str]:
    return [policy_key(m["name"], p) for m in MODELS.values() for p in PERSONALITIES]


# -- payload helpers -------------------------------------------------------------------------

def legal_targets(payload: dict[str, Any]) -> dict[str, list[str | None]]:
    """action -> legal targets (None for an untargeted action), in payload order."""
    out: dict[str, list[str | None]] = {}
    for entry in payload.get("legal_actions") or []:
        if isinstance(entry, dict) and entry.get("action"):
            out.setdefault(entry["action"], []).append(entry.get("target_id"))
    return out


def object_types(payload: dict[str, Any]) -> dict[str, str]:
    return {o["id"]: o.get("type") for o in (payload.get("room") or {}).get("objects") or [] if isinstance(o, dict) and o.get("id")}


def _is_door(target: str | None, types: dict[str, str]) -> bool:
    return bool(target) and (types.get(target) == "door" or str(target).startswith("door_")) and target != "exit_door"


def category(action: str | None, target: str | None, payload: dict[str, Any]) -> str:
    """Analysis category of one (action, target) in this payload."""
    if not action:
        return "OTHER"
    if action in ("move", "interact") and target == "exit_door":
        return "EXIT"
    types = object_types(payload)
    if action == "move":
        return "MOVE_PROGRESS"
    if action == "interact":
        return "MOVE_PROGRESS" if _is_door(target, types) else "SEARCH"
    return {"approach_opponent": "APPROACH", "avoid_opponent": "AVOID", "inspect": "INSPECT",
            "place_trap": "PLACE_TRAP", "disarm": "DISARM", "wait": "WAIT",
            # opening the safe with its code is how the safe is searched
            "enter_safe_code": "SEARCH"}.get(action, "OTHER")


# -- target classes (target policy v1) ----------------------------------------------------------

TARGET_CLASSES = ("exit_door", "door_untraversed", "door_traversed", "door_blocked", "safe", "unsearched_here",
                  "unsearched_elsewhere", "other_object", "opponent", "near_object", "none")


def target_class(action: str, target: str | None, payload: dict[str, Any]) -> str:
    """What kind of target this is, from the model-visible exploration facts only."""
    if target is None:
        return "none"
    if action in OPPONENT_ACTIONS:
        return "opponent"
    if action in NEAR_OBJECT_ACTIONS:
        return "near_object"
    if target == "exit_door":
        return "exit_door"
    types = object_types(payload)
    exploration = payload.get("exploration") or {}
    rooms = exploration.get("rooms") or {}
    here = exploration.get("current_room") or (payload.get("self") or {}).get("room")
    if _is_door(target, types):
        if any(target in (r.get("blocked") or {}) for r in rooms.values()):
            return "door_blocked"
        if any(target in (r.get("untraversed_exits") or []) for r in rooms.values()):
            return "door_untraversed"
        return "door_traversed"
    if types.get(target) == "safe" or target == "ambassador_safe":
        return "safe"
    if target in ((rooms.get(here) or {}).get("unsearched") or []):
        return "unsearched_here"
    if any(target in (r.get("unsearched") or []) for name, r in rooms.items() if name != here):
        return "unsearched_elsewhere"
    return "other_object"


# -- the riddle code, from what the spy has read (gameplay only) -----------------------------------

_ID_IN_CLUE = re.compile(r"\(([A-Za-z0-9_]+)\)")


def code_from_visible_puzzle(payload: dict[str, Any]) -> str | None:
    """The safe code a spy can work out from the clue and the inscriptions it has
    read, or None. Uses only the payload's `puzzle` block (what the LLM sees):
    the clue names each object as "label (id)"; the four clue wordings are the
    ones src/puzzles.py writes. Never reads the engine's code."""
    puzzle = payload.get("puzzle") or {}
    clue = puzzle.get("clue")
    read = puzzle.get("inscriptions_you_have_read") or {}
    if not clue or not isinstance(read, dict):
        return None
    ids = _ID_IN_CLUE.findall(clue)
    if len(ids) < 2:
        return None

    def value(obj_id: str) -> str | None:
        text = read.get(obj_id)
        return text.split(": ", 1)[1].strip() if isinstance(text, str) and ": " in text else None
    first, second = value(ids[0]), value(ids[1])
    try:
        if "followed by" in clue:  # "number on A ... followed by the number on B"
            return first + second if first and second else None
        if clue.startswith("Add"):  # "Add the number on B ... to the year on A"
            return str(int(first) + int(second)) if first and second else None
        if "subtract" in clue:  # "Take the year on A ... subtract the number on B"
            return str(int(first) - int(second)) if first and second else None
        if "written backwards" in clue:  # "year on A, written backwards (B is a red herring)"
            return first[::-1] if first else None
    except (TypeError, ValueError):
        return None
    return None
