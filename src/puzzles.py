"""Live safe puzzles (live game only; recorded runs keep the plain safe code).

The level's lock chain is: archive key -> archives -> security keycard ->
security room -> safe code -> ambassador safe -> secret documents. In the
recorded rules one object in the security room reveals `safe_code` when
searched. `LiveGame` can replace that with:

* SPLIT - the code comes in two halves, `safe_code_part_1` (the security
  room object that used to reveal the whole code) and `safe_code_part_2`
  (an object in another room). A spy that knows both halves knows
  `safe_code` (`KNOWLEDGE_COMBINED`), so the safe's own lock rule is unchanged.

* RIDDLE - the code is written nowhere. The security room object reveals a
  clue instead (`safe_clue`), e.g. "the year on the painting in the
  ambassador office, written backwards". The facts the clue refers to are
  inscriptions on two other objects; a spy reads an inscription when it
  inspects or searches that object. Decoy inscriptions sit on other objects.
  The spy enters its answer with `enter_safe_code` (target the safe, field
  `code`): right -> it knows `safe_code` and the safe opens; wrong -> it is
  busy WRONG_CODE_SECONDS and the safe's alarm shows its position to the
  opponent for ALARM_REVEAL_DURATION.

Everything is drawn from an RNG of its own (`puzzle:<seed>`), never the
shared game RNG, and each riddle is checked by `solve_riddle` - the
generator's facts must give exactly the stored code.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from src import constants

if TYPE_CHECKING:  # pragma: no cover
    from src.state import GameState

NONE = "none"
SPLIT = "split"
RIDDLE = "riddle"
PUZZLES = (NONE, SPLIT, RIDDLE)

SAFE_ID = "ambassador_safe"
SAFE_CODE = "safe_code"
PART_1, PART_2 = "safe_code_part_1", "safe_code_part_2"
CLUE = "safe_clue"
ENTER_SAFE_CODE = "enter_safe_code"
ENTER_CODE_ALIASES = {"enter_safe_code", "enter_code", "crack_safe", "safe_code", "enter_combination"}
WRONG_CODE_SECONDS = 4.0
KNOWLEDGE_NAMES = {PART_1: "Safe code - first half", PART_2: "Safe code - second half", CLUE: "Safe riddle"}
TEMPLATES = ("concat", "sum", "reverse", "difference")


class PuzzleSetupError(Exception):
    """The level has no room for this puzzle (not enough objects / rooms)."""


# -- helpers ---------------------------------------------------------------------------

def _code_object(state: "GameState"):
    return next((o for o in state.level.objects if o.reveals_knowledge == SAFE_CODE), None)


def _reachable_rooms(state: "GameState") -> set[str]:
    from src.dependency_generation import _reachable_rooms as reach, _room_graph

    graph = _room_graph(state.level)
    spawn = state.level.room_id_at(*state.level.player_spawn)
    keys = {i for i in state.dependency_placement.item_locations} - set(state.items.required)
    return reach(graph, spawn, keys, set())


def _candidates(state: "GameState", exclude: set[str], trappable_only: bool = False) -> list:
    rooms = _reachable_rooms(state)
    out = []
    for obj in state.level.objects:
        if obj.type == "door" or not obj.interactable or obj.id in exclude or obj.id == SAFE_ID:
            continue
        if obj.reveals_knowledge is not None:
            continue
        if trappable_only and not obj.trappable:
            continue
        if state.level.object_room(obj) in rooms:
            out.append(obj)
    return sorted(out, key=lambda o: o.id)


def _room_label(state: "GameState", obj) -> str:
    room = state.level.object_room(obj)
    rooms = {r.id: r.label for r in state.level.rooms}
    return rooms.get(room, room or "?")


def _name(obj) -> str:
    """Label plus id: two objects can share a label in one room (two bookshelves)."""
    return f"{obj.label} ({obj.id})"


# -- setup -------------------------------------------------------------------------------

def setup(state: "GameState", mode: str) -> dict[str, Any] | None:
    """Turns the plain safe code into `mode` on this game's own Level copy."""
    if mode not in PUZZLES:
        raise ValueError(f"unknown puzzle {mode!r}")
    state.puzzle_mode = mode
    if mode == NONE:
        return None
    code_obj = _code_object(state)
    if code_obj is None:
        raise PuzzleSetupError("no object reveals the safe code")
    rng = random.Random(f"puzzle:{state.seed}")
    state.items.knowledge.update(KNOWLEDGE_NAMES)
    record = (_setup_split if mode == SPLIT else _setup_riddle)(state, code_obj, rng)
    state.puzzle = record
    # the record's own "type" key would overwrite the event type (log_event
    # builds {"type": event_type, **fields}); `puzzle=mode` already says it
    state._log_event("PUZZLE_SET_UP", puzzle=mode,
                     **{k: v for k, v in record.items() if k not in ("type", "code", "clue")})
    return record


def _setup_split(state, code_obj, rng) -> dict[str, Any]:
    first_room = state.level.object_room(code_obj)
    pool = [o for o in _candidates(state, {code_obj.id}) if state.level.object_room(o) != first_room]
    if not pool:
        raise PuzzleSetupError("no object outside the code room for the second half")
    second = rng.choice(pool)
    code_obj.reveals_knowledge = PART_1
    second.reveals_knowledge = PART_2
    return {"type": SPLIT, "parts": {PART_1: code_obj.id, PART_2: second.id}}


def _setup_riddle(state, code_obj, rng) -> dict[str, Any]:
    pool = _candidates(state, {code_obj.id}, trappable_only=True)
    by_room: dict[str, list] = {}
    for obj in pool:
        by_room.setdefault(state.level.object_room(obj), []).append(obj)
    rooms = sorted(by_room)
    if len(rooms) < 2:
        raise PuzzleSetupError("a riddle needs inscribed objects in two different rooms")
    room_a, room_b = rng.sample(rooms, 2)
    obj_a, obj_b = rng.choice(by_room[room_a]), rng.choice(by_room[room_b])
    template = rng.choice(TEMPLATES)
    year = rng.randrange(1903, 1990)
    while year % 10 == 0:  # a reversed year must not start with 0
        year = rng.randrange(1903, 1990)
    number = rng.randrange(12, 89)
    if template == "concat":
        a_value, b_value = str(rng.randrange(10, 99)), str(rng.randrange(10, 99))
        code = a_value + b_value
        text = (f"The safe code is the number on the {_name(obj_a)} in the {_room_label(state, obj_a)} "
                f"followed by the number on the {_name(obj_b)} in the {_room_label(state, obj_b)}.")
    elif template == "sum":
        a_value, b_value = str(year), str(number)
        code = str(year + number)
        text = (f"Add the number on the {_name(obj_b)} in the {_room_label(state, obj_b)} to the year on the "
                f"{_name(obj_a)} in the {_room_label(state, obj_a)}: that is the safe code.")
    elif template == "difference":
        a_value, b_value = str(year), str(number)
        code = str(year - number)
        text = (f"Take the year on the {_name(obj_a)} in the {_room_label(state, obj_a)} and subtract the number "
                f"on the {_name(obj_b)} in the {_room_label(state, obj_b)}: that is the safe code.")
    else:  # reverse
        a_value, b_value = str(year), str(number)
        code = str(year)[::-1]
        text = (f"The safe code is the year on the {_name(obj_a)} in the {_room_label(state, obj_a)}, written "
                f"backwards. (The number on the {_name(obj_b)} in the {_room_label(state, obj_b)} is a red herring.)")
    inscriptions = {obj_a.id: f"engraved: {a_value}", obj_b.id: f"engraved: {b_value}"}
    decoys = [o for o in pool if o.id not in inscriptions]
    rng.shuffle(decoys)
    used = {a_value, b_value}
    for obj in decoys[:2]:
        value = a_value
        while value in used:  # a decoy never repeats a real fact
            value = str(rng.choice([rng.randrange(1903, 1990), rng.randrange(12, 89)]))
        used.add(value)
        inscriptions[obj.id] = f"engraved: {value}"
    code_obj.reveals_knowledge = CLUE
    record = {"type": RIDDLE, "template": template, "clue_object": code_obj.id, "clue": text, "code": code,
              "facts": {"a": obj_a.id, "b": obj_b.id}, "inscriptions": inscriptions}
    if solve_riddle(record) != code:
        raise PuzzleSetupError("generated riddle does not solve to its code")
    state.inscriptions = dict(inscriptions)
    return record


def solve_riddle(record: dict[str, Any]) -> str:
    """Independent check: the clue's template applied to the inscriptions."""
    a = record["inscriptions"][record["facts"]["a"]].split(": ")[1]
    b = record["inscriptions"][record["facts"]["b"]].split(": ")[1]
    template = record["template"]
    if template == "concat":
        return a + b
    if template == "sum":
        return str(int(a) + int(b))
    if template == "difference":
        return str(int(a) - int(b))
    return a[::-1]


# -- engine hooks ----------------------------------------------------------------------------

def after_knowledge(state: "GameState", actor) -> None:
    """SPLIT: both halves known -> the whole code is known."""
    if state.puzzle_mode == SPLIT and PART_1 in actor.knowledge and PART_2 in actor.knowledge \
            and SAFE_CODE not in actor.knowledge:
        actor.knowledge.add(SAFE_CODE)
        state._log_event("KNOWLEDGE_COMBINED", actor_id=actor.id, knowledge_id=SAFE_CODE, parts=[PART_1, PART_2])


def read_inscription(state: "GameState", actor, obj) -> str | None:
    """RIDDLE: inspecting or searching an inscribed object reads it."""
    text = state.inscriptions.get(obj.id) if state.puzzle_mode == RIDDLE else None
    if text is None:
        return None
    seen = state.seen_inscriptions.setdefault(actor.id, {})
    if obj.id not in seen:
        seen[obj.id] = text
        state._log_event("INSCRIPTION_READ", actor_id=actor.id, object_id=obj.id, text=text)
    return text


def enter_code(state: "GameState", actor, obj, code: str):
    from src.actions import InteractionResult

    state.code_attempts[actor.id] = state.code_attempts.get(actor.id, 0) + 1
    entered = str(code).strip()
    if entered == state.puzzle["code"]:
        actor.knowledge.add(SAFE_CODE)
        state._log_event("SAFE_CODE_SOLVED", actor_id=actor.id, object_id=obj.id,
                         attempts=state.code_attempts[actor.id])
        result = state._search(actor, obj)
        return InteractionResult(True, "The safe clicks open.\n" + result.message, item_found=result.item_found)
    actor.action_cooldown = WRONG_CODE_SECONDS
    other = state._other_id(actor.id)
    state.actors[other].alarm_reveal_timer = constants.ALARM_REVEAL_DURATION
    state._log_event("SAFE_CODE_WRONG", actor_id=actor.id, object_id=obj.id, entered=entered,
                     attempts=state.code_attempts[actor.id])
    state._log_event("ALARM_REVEALED", actor_id=other, victim_id=actor.id, seconds=constants.ALARM_REVEAL_DURATION,
                     cause="wrong_safe_code")
    return InteractionResult(False, "Wrong code - the safe's alarm sounds.")


# -- what an actor knows -----------------------------------------------------------------------

def view(state: "GameState", actor_id: str) -> dict[str, Any] | None:
    """The puzzle facts this actor legitimately knows (live payload)."""
    mode = getattr(state, "puzzle_mode", NONE)
    if mode == NONE or state.puzzle is None:
        return None
    actor = state.actors[actor_id]
    if mode == SPLIT:
        parts = [p for p in (PART_1, PART_2) if p in actor.knowledge]
        return {"type": SPLIT, "safe_code_halves_known": parts, "halves_needed": 2,
                "safe_code_known": SAFE_CODE in actor.knowledge,
                "rule": "the safe code is split in two halves revealed by searching two objects in two different "
                        "rooms; knowing both halves means knowing the safe code"}
    return {
        "type": RIDDLE,
        "clue": state.puzzle["clue"] if CLUE in actor.knowledge else None,
        "inscriptions_you_have_read": dict(sorted(state.seen_inscriptions.get(actor_id, {}).items())),
        "safe_code_known": SAFE_CODE in actor.knowledge,
        "code_attempts": state.code_attempts.get(actor_id, 0),
        "how_to_answer": f"{ENTER_SAFE_CODE} with target_id {SAFE_ID} and code \"<digits>\" while next to the safe",
        "wrong_code_costs": f"{WRONG_CODE_SECONDS:g} s busy, and the alarm shows your position to the opponent "
                            f"for {constants.ALARM_REVEAL_DURATION:g} s",
    }
