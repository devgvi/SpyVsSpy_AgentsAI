"""M6C spec sections 24-26: 8 frozen decision states where memory should
plausibly matter (plus one negative control), each paired with a
deterministic SYNTHETIC `AgentMemory` (spec section 24's "build or reuse
real game states where memory should plausibly matter").

Frozen single-decision snapshots have no organic engine-tick history (see
runtime_snapshot_replay.py's own documented scope decision), so the
GameState-level per-actor room/lock memory (`Actor.visited_rooms`/
`known_locked_objects`, exposed independently via Observation - see
`memory_representation.py`'s module docstring on duplication) reflects
only the frozen snapshot's own spawn position, never any seeded backstory.
This means a synthetic `AgentMemory` seeded here carries genuinely NEW
information beyond what Observation already shows - exactly the clean
"does memory content change behavior" test this milestone needs, without
the same confound organic multi-decision play would have.

Each snapshot's synthetic memory is built with `AgentMemory`'s own public
`record_*` API (never hand-built dicts) - it is real production code,
never a parallel reimplementation - then frozen via `.to_compact_dict()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src import constants
from src.evaluation.runtime_snapshot_replay import (
    DecisionSnapshot,
    _fresh_state,
    _place,
    _place_near_object,
    _snapshot_from_state,
)
from src.llm.agent_memory import AgentMemory
from src.state import GameState

VISITED_ROOM_AVOIDANCE = "VISITED_ROOM_AVOIDANCE"
KNOWN_LOCKED_OBJECT = "KNOWN_LOCKED_OBJECT"
LAST_SEEN_OPPONENT = "LAST_SEEN_OPPONENT"
RECENT_FAILED_ACTION = "RECENT_FAILED_ACTION"
RECENT_SUCCESSFUL_ACTION = "RECENT_SUCCESSFUL_ACTION"
BACKTRACK_REQUIRED = "BACKTRACK_REQUIRED"
MULTIPLE_LEGAL_TARGETS = "MULTIPLE_LEGAL_TARGETS"
NO_MEMORY_RELEVANT = "NO_MEMORY_RELEVANT"

MEMORY_SNAPSHOT_CATEGORIES = (
    VISITED_ROOM_AVOIDANCE, KNOWN_LOCKED_OBJECT, LAST_SEEN_OPPONENT, RECENT_FAILED_ACTION,
    RECENT_SUCCESSFUL_ACTION, BACKTRACK_REQUIRED, MULTIPLE_LEGAL_TARGETS, NO_MEMORY_RELEVANT,
)


@dataclass(frozen=True)
class MemorySnapshot:
    snapshot_id: str
    category: str
    observation: Any
    legal_actions: list[dict[str, Any]] = field(repr=False)
    synthetic_memory: dict[str, Any] = field(repr=False)  # AgentMemory.to_compact_dict() output


def _fresh_memory() -> AgentMemory:
    return AgentMemory(constants.WHITE_SPY_ID)


def build_memory_snapshot_set() -> list[MemorySnapshot]:
    snapshots: list[MemorySnapshot] = []

    # 1. VISITED_ROOM_AVOIDANCE - actor at spawn (corridor), several doors
    # to unvisited rooms legal, memory claims 2 rooms already visited
    # (reception, archives) - a real room this actor has never actually
    # entered in this frozen snapshot's own GameState, so this fact exists
    # ONLY in memory, never leaked via Observation's own known_rooms.
    state = _fresh_state()
    memory = _fresh_memory()
    memory.record_visited_room("corridor")
    memory.record_visited_room("reception")
    memory.record_visited_room("archives")
    ds = _snapshot_from_state("mem01_visited_room_avoidance", VISITED_ROOM_AVOIDANCE, state)
    snapshots.append(MemorySnapshot(ds.snapshot_id, ds.category, ds.observation, ds.legal_actions, memory.to_compact_dict()))

    # 2. KNOWN_LOCKED_OBJECT - actor near ambassador_safe (a real code
    # lock), memory records it as already-known-locked and NOT currently
    # satisfied - a deterrent fact, distinct from `obj.can_unlock` (which
    # Observation still shows independently, unaffected by this).
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "ambassador_safe")
    memory = _fresh_memory()
    memory.record_known_lock("ambassador_safe", "code", "unknown", satisfied=False)
    memory.record_failure(5.0, "interact", "ambassador_safe", "locked", detail="requires a code this actor does not have")
    ds = _snapshot_from_state("mem02_known_locked_object", KNOWN_LOCKED_OBJECT, state)
    snapshots.append(MemorySnapshot(ds.snapshot_id, ds.category, ds.observation, ds.legal_actions, memory.to_compact_dict()))

    # 3. LAST_SEEN_OPPONENT - opponent not currently visible in this frozen
    # Observation (placed far away), but memory claims a recent sighting
    # in a specific room - real information Observation cannot supply on
    # its own here (Actor.last_seen_opponent is engine-tracked and empty
    # for an un-ticked frozen state).
    state = _fresh_state()
    white = state.actors[constants.WHITE_SPY_ID]
    _place(state, constants.BLACK_SPY_ID, white.x + 900, white.y + 900)  # far away, not visible
    memory = _fresh_memory()
    memory.record_opponent_sighting(game_time=170.0, room_id="security_room")
    memory.record_opponent_left_visibility()
    ds = _snapshot_from_state("mem03_last_seen_opponent", LAST_SEEN_OPPONENT, state)
    snapshots.append(MemorySnapshot(ds.snapshot_id, ds.category, ds.observation, ds.legal_actions, memory.to_compact_dict()))

    # 4. RECENT_FAILED_ACTION - memory records a recent failed interaction
    # with the object the actor is now standing next to again - a fact
    # with no other prompt-visible source at all (recent_failures is
    # AgentMemory-only, never duplicated in Observation).
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "filing_cabinet_01")
    memory = _fresh_memory()
    memory.record_failure(8.0, "search", "filing_cabinet_01", "already_searched", detail="found nothing of value")
    memory.record_interaction(8.0, "search", "filing_cabinet_01", "empty")
    ds = _snapshot_from_state("mem04_recent_failed_action", RECENT_FAILED_ACTION, state)
    snapshots.append(MemorySnapshot(ds.snapshot_id, ds.category, ds.observation, ds.legal_actions, memory.to_compact_dict()))

    # 5. RECENT_SUCCESSFUL_ACTION - memory records a recent SUCCESSFUL
    # interaction (found an item) with a nearby, still-legal target -
    # positive reinforcement framing, same "AgentMemory-only fact" shape
    # as case 4.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "locker_01")
    memory = _fresh_memory()
    memory.record_interaction(9.0, "search", "locker_01", "found_item", detail="found passport")
    ds = _snapshot_from_state("mem05_recent_successful_action", RECENT_SUCCESSFUL_ACTION, state)
    snapshots.append(MemorySnapshot(ds.snapshot_id, ds.category, ds.observation, ds.legal_actions, memory.to_compact_dict()))

    # 6. BACKTRACK_REQUIRED - actor now holds (per Observation's own
    # inventory) an item memory recalls was needed to unlock a specific
    # door back in a previously-visited room - the model must connect
    # "I now have this" with "that unlocks the thing memory told me
    # about" (a within-episode strategic-recall test, not a bare fact
    # recall). Actor is deliberately NOT near that door right now (it must
    # be recalled, not just re-observed).
    state = _fresh_state()
    _place(state, constants.WHITE_SPY_ID, *state.level.player_spawn)
    memory = _fresh_memory()
    memory.record_visited_room("ambassador_office")
    memory.record_known_lock("ambassador_safe", "code", "requires archive_code", satisfied=False)
    memory.record_failure(20.0, "interact", "ambassador_safe", "locked", detail="requires archive_code")
    ds = _snapshot_from_state("mem06_backtrack_required", BACKTRACK_REQUIRED, state)
    snapshots.append(MemorySnapshot(ds.snapshot_id, ds.category, ds.observation, ds.legal_actions, memory.to_compact_dict()))

    # 7. MULTIPLE_LEGAL_TARGETS - centered in main_office with several
    # searchable objects (reuses the exact same position M6A's own
    # MULTIPLE_LEGAL_TARGETS category already validated - desk_01 nearby,
    # several other objects in room), memory adds a recent-repetitions-
    # worthy decision history (the same target attempted twice already) to
    # test whether that loop-detection signal shifts target choice among
    # several otherwise-equal options.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "desk_01")
    memory = _fresh_memory()
    memory.record_decision(2.0, "d1", "interact", "desk_01", True, "llm")
    memory.record_decision(4.0, "d2", "interact", "desk_01", True, "llm")
    memory.record_interaction(2.0, "search", "desk_01", "empty")
    memory.record_interaction(4.0, "search", "desk_01", "empty")
    ds = _snapshot_from_state("mem07_multiple_legal_targets", MULTIPLE_LEGAL_TARGETS, state)
    snapshots.append(MemorySnapshot(ds.snapshot_id, ds.category, ds.observation, ds.legal_actions, memory.to_compact_dict()))

    # 8. NO_MEMORY_RELEVANT - negative control (spec section 25): open
    # floor, nothing interactable nearby, and a genuinely EMPTY synthetic
    # memory (fresh AgentMemory, no seeded facts at all) - if representation
    # still measurably shifts behavior here, that is itself a finding
    # (representation salience/bias with no real facts behind it).
    # Reuses the exact offset runtime_snapshot_replay.py's own
    # SIMPLE_NAVIGATION category already proved interactable-free
    # (player_spawn itself sits within INTERACTION_DISTANCE of
    # door_main_office_ambassador).
    state = _fresh_state()
    _place(state, constants.WHITE_SPY_ID, state.level.player_spawn[0], state.level.player_spawn[1] - 50)
    memory = _fresh_memory()
    ds = _snapshot_from_state("mem08_no_memory_relevant", NO_MEMORY_RELEVANT, state)
    snapshots.append(MemorySnapshot(ds.snapshot_id, ds.category, ds.observation, ds.legal_actions, memory.to_compact_dict()))

    return snapshots
