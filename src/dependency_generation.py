"""Milestone 4B: seeded placement of keys, keycards and codes.

`level_01.json` authors the lock *topology* - which doors are gated, and
by what kind of lock (`GameObject.lock_type`/`requires`) - as static
architecture, same as where the walls are. This module decides, fresh
every game from `GameState`'s own seeded `rng`, which specific object in
the level actually holds each key/card/passport, and which object reveals
the safe's code - Milestone_4.md's explicit "do not hard-code one route".

The placement is built tier by tier, where a tier is "the set of rooms
reachable given what a perfectly-informed actor could be holding so far"
(`_reachable_rooms`, walking the level's own door graph via each door's
`connects`/`lock_type`/`requires` - the same fields physics and navigation
already treat as ground truth, not a second approximate model of the
level). Because each tier's item is only ever drawn from rooms reachable
*without* it, the placement is solvable by construction - and
`validate_solvable` re-checks that generically afterwards (a fixpoint
closure over the level's lock graph, not a re-run of the same tiering
logic) and raises loudly if it isn't, the same "generate, then validate
independently" pattern `layout_validation.py` established for M4A.2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.interactable import GameObject
from src.items import ItemCatalog
from src.level import Level

# The safe's own contents are the fixed capstone of the chain (Milestone_4.md's
# example: "Ambassador safe + 1947 -> Secret Documents") - not randomized,
# since there's only one such object and only one such item.
SAFE_OBJECT_ID = "ambassador_safe"
SAFE_ITEM = "secret_documents"
SAFE_KNOWLEDGE = "safe_code"

# Randomized per seed: which eligible container in the right tier holds each.
_TIER0_ITEMS = ("archive_key", "passport")  # freely reachable from spawn
_TIER1_ITEM = "security_keycard"  # reachable once archive_key is held
_TIER2_KNOWLEDGE = SAFE_KNOWLEDGE  # reachable once security_keycard is held


class UnsolvableLevelError(Exception):
    """Raised when a generated (or hand-authored) placement can't actually
    be completed - a circular lock, or a lock naming an item/knowledge id
    nothing ever grants. Fails loudly at startup rather than shipping an
    uncompletable episode (Milestone_4.md's explicit acceptance bar)."""


@dataclass(frozen=True)
class DependencyPlacement:
    item_locations: dict[str, str]  # item_id -> object_id holding it
    knowledge_locations: dict[str, str]  # knowledge_id -> object_id revealing it


def _requirement_met(held_items: set[str], known: set[str], lock_type: str, requires: str) -> bool:
    """Same rule as GameState.requirement_met, restated over plain sets:
    this runs before any GameState/Actor exists (it's what decides the
    world those will start in), so it can't call that method directly."""
    if lock_type == "code":
        return requires in known
    return requires in held_items


def _room_graph(level: Level) -> dict[str, list[tuple[str, GameObject]]]:
    graph: dict[str, list[tuple[str, GameObject]]] = {room.id: [] for room in level.rooms}
    for obj in level.objects:
        if obj.type != "door" or obj.connects is None:
            continue
        a, b = obj.connects
        graph.setdefault(a, []).append((b, obj))
        graph.setdefault(b, []).append((a, obj))
    return graph


def _reachable_rooms(
    graph: dict[str, list[tuple[str, GameObject]]], spawn_room_id: str, held_items: set[str], known: set[str]
) -> set[str]:
    reachable = {spawn_room_id}
    changed = True
    while changed:
        changed = False
        for room_id in list(reachable):
            for other, door in graph.get(room_id, []):
                if other in reachable:
                    continue
                if door.lock_type is None or _requirement_met(held_items, known, door.lock_type, door.requires):
                    reachable.add(other)
                    changed = True
    return reachable


def _eligible_containers(level: Level, exclude_ids: set[str]) -> dict[str, list[GameObject]]:
    """Interactable, non-door objects with no pre-authored contents,
    grouped by the room they're in - the pool `generate_dependencies` draws
    from. `exclude_ids` keeps a fixed-contents object (the safe) out of the
    pool for everything else."""
    by_room: dict[str, list[GameObject]] = {}
    for obj in level.objects:
        if obj.type == "door" or not obj.interactable or obj.contains or obj.id in exclude_ids:
            continue
        room_id = level.room_id_at(*obj.center)
        if room_id is None:
            continue
        by_room.setdefault(room_id, []).append(obj)
    return by_room


def _pool(by_room: dict[str, list[GameObject]], room_ids: set[str]) -> list[GameObject]:
    pool: list[GameObject] = []
    for room_id in room_ids:
        pool.extend(by_room.get(room_id, []))
    pool.sort(key=lambda o: o.id)  # deterministic order before rng.shuffle - dict/set iteration order isn't
    return pool


def generate_dependencies(level: Level, items: ItemCatalog, rng: random.Random) -> DependencyPlacement:
    graph = _room_graph(level)
    spawn_room_id = level.room_id_at(*level.player_spawn)
    if spawn_room_id is None:
        raise UnsolvableLevelError("player_spawn is not inside any room")

    by_room = _eligible_containers(level, exclude_ids={SAFE_OBJECT_ID})

    tier0_rooms = _reachable_rooms(graph, spawn_room_id, set(), set())
    tier0_pool = _pool(by_room, tier0_rooms)
    if len(tier0_pool) < len(_TIER0_ITEMS):
        raise UnsolvableLevelError(f"need {len(_TIER0_ITEMS)} freely-reachable containers, found {len(tier0_pool)}")
    rng.shuffle(tier0_pool)
    archive_key_obj, passport_obj = tier0_pool[0], tier0_pool[1]

    tier1_rooms = _reachable_rooms(graph, spawn_room_id, {"archive_key"}, set()) - tier0_rooms
    tier1_pool = _pool(by_room, tier1_rooms)
    if not tier1_pool:
        raise UnsolvableLevelError("no container reachable behind the archive_key lock")
    rng.shuffle(tier1_pool)
    keycard_obj = tier1_pool[0]

    tier2_rooms = _reachable_rooms(graph, spawn_room_id, {"archive_key", "security_keycard"}, set())
    tier2_rooms -= tier0_rooms | tier1_rooms
    tier2_pool = _pool(by_room, tier2_rooms)
    if not tier2_pool:
        raise UnsolvableLevelError("no container reachable behind the security_keycard lock")
    rng.shuffle(tier2_pool)
    code_obj = tier2_pool[0]

    placement = DependencyPlacement(
        item_locations={
            "archive_key": archive_key_obj.id,
            "passport": passport_obj.id,
            "security_keycard": keycard_obj.id,
            SAFE_ITEM: SAFE_OBJECT_ID,
        },
        knowledge_locations={_TIER2_KNOWLEDGE: code_obj.id},
    )

    if not validate_solvable(level, items, placement):
        raise UnsolvableLevelError("generated placement is not solvable - check level_01.json's lock graph")
    return placement


def validate_solvable(level: Level, items: ItemCatalog, placement: DependencyPlacement) -> bool:
    """Generic fixpoint closure: starting from nothing, repeatedly expand
    (rooms reachable) x (objects unlockable in those rooms) x (items/
    knowledge those objects grant) until nothing new is gained, then check
    every required mission item was reached. Independent of the tiering
    logic `generate_dependencies` used to build the placement - this would
    catch a mistake there, or in hand-authored level data, just as well.
    """
    graph = _room_graph(level)
    spawn_room_id = level.room_id_at(*level.player_spawn)
    if spawn_room_id is None:
        return False

    held: set[str] = set()
    known: set[str] = set()
    opened: set[str] = set()

    changed = True
    while changed:
        changed = False
        reachable = _reachable_rooms(graph, spawn_room_id, held, known)
        for obj in level.objects:
            if obj.type == "door" or obj.id in opened:
                continue
            if level.room_id_at(*obj.center) not in reachable:
                continue
            if obj.lock_type is not None and not _requirement_met(held, known, obj.lock_type, obj.requires):
                continue
            opened.add(obj.id)
            for item_id, object_id in placement.item_locations.items():
                if object_id == obj.id and item_id not in held:
                    held.add(item_id)
                    changed = True
            for knowledge_id, object_id in placement.knowledge_locations.items():
                if object_id == obj.id and knowledge_id not in known:
                    known.add(knowledge_id)
                    changed = True

    return all(item_id in held for item_id in items.required)


def apply_placement(level: Level, placement: DependencyPlacement) -> None:
    """Mutates the freshly-loaded `level` in place, injecting each
    generated assignment exactly as if it had been authored directly into
    level_01.json. Safe to call once per Level instance's lifetime -
    `load_level` returns a brand-new Level every time (see
    GameState.new_game / game.py), so nothing here leaks between games.
    """
    by_id = {obj.id: obj for obj in level.objects}
    for item_id, object_id in placement.item_locations.items():
        by_id[object_id].contains.append(item_id)
    for knowledge_id, object_id in placement.knowledge_locations.items():
        by_id[object_id].reveals_knowledge = knowledge_id
