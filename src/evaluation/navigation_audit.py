"""Bugfix audit: does a production MOVE_TO(object) end where the intended
interaction is legal? (BUGFIX_NAVIGATION_AND_EXPLORATION_MEMORY.md, §6.)

Drives the real engine: GameState, build_observation, ActionExecutor,
Actor.move collisions and the engine's own interaction-range rule. For
every interactable object and a set of start positions it records the
instrumentation §6 asks for and classifies each case:

    OK                           reached; the target is the engine's near object
    PATH_FAILED                  start() found no path although the target is reachable
    APPROACH_POINT_INVALID       the chosen approach point cannot make the target the near object
    STOPPED_TOO_EARLY            no boundary within the tick budget, actor stuck short of the path
    COLLISION_PREVENTED_APPROACH stuck because a step was rejected by collision
    INTERACTION_RANGE_MISMATCH   TARGET_REACHED, but the near object is another object (or none)
    UNREACHABLE_EXPECTED         no walkable cell of the grid can reach an approach point (legal
                                 MOVE to an unreachable target - the known legality gap)

Read-only with respect to production code; writes only where asked.

    python -m src.evaluation.navigation_audit [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src import constants  # noqa: E402
from src.agents.action_lifecycle import ActionExecutor, PATH_UNREACHABLE, TARGET_REACHED  # noqa: E402
from src.items import load_items  # noqa: E402
from src.level import load_level  # noqa: E402
from src.observation import build_observation  # noqa: E402
from src.state import GameState  # noqa: E402

ACTOR = constants.WHITE_SPY_ID
OTHER = constants.BLACK_SPY_ID
# 1/60 is the normal 60 fps frame; 1/30 is what the existing tests use; 0.1
# is the live loop's per-frame cap (workshop FRAME_DT_CAP); 0.2 is that cap at
# speed x2. src.main's Game.run() has no cap at all.
DTS = (1 / 60, 1 / 30, 0.1, 0.2)
TICK_BUDGET_SECONDS = 20.0


def fresh_state(seed: int = 501, doors_open: bool = True) -> GameState:
    state = GameState.new_game(load_level(constants.LEVEL_PATH), load_items(constants.ITEMS_PATH), seed=seed)
    if doors_open:
        for obj in state.level.objects:
            if obj.type == "door" and obj.id != "exit_door":
                state.door_states[obj.id] = "open"
        state._rebuild_nav_grid()
    # park the other actor far away so it never becomes the near object or blocks
    other = state.actors[OTHER]
    other.x, other.y = state.level.player_spawn
    state.update(0.0)
    return state


def start_positions(state: GameState, room_id: str, n: int, rng: random.Random) -> list[tuple[float, float]]:
    """Walkable cell centres in `room_id` where the actor's collision box is free."""
    grid = state.nav_grid
    solids = state.level.solid_rects(state.door_states)
    w, h = constants.PLAYER_COLLISION_SIZE
    import pygame
    cells = []
    for row in range(grid.rows):
        for col in range(grid.cols):
            if not grid.is_walkable((row, col)):
                continue
            cx, cy = grid.center_of((row, col))
            if state.level.room_id_at(cx, cy) != room_id:
                continue
            box = pygame.Rect(int(cx - w / 2), int(cy - h / 2), int(w), int(h))
            if any(box.colliderect(s) for s in solids):
                continue
            cells.append((cx, cy))
    rng.shuffle(cells)
    return sorted(cells[:n])


def distance_to_rect(x: float, y: float, rect) -> float:
    cx = max(rect.left, min(x, rect.right))
    cy = max(rect.top, min(y, rect.bottom))
    return math.hypot(x - cx, y - cy)


def run_move(state: GameState, target_id: str, start: tuple[float, float], dt: float) -> dict[str, Any]:
    actor = state.actors[ACTOR]
    actor.x, actor.y = start
    actor.moving = False
    state.update(0.0)
    target = state.level.get_object(target_id)
    obs = build_observation(state, ACTOR)
    executor = ActionExecutor()
    action, boundary, _ = executor.start(obs, "d", "move", target_id, None, obs.mission_time_remaining)
    active = executor.active or executor.last_finished
    path = list(getattr(active, "path", None) or [])
    approach = path[-1] if path else None
    ticks, rejected_steps, last_pos = 0, 0, (actor.x, actor.y)
    max_ticks = int(TICK_BUDGET_SECONDS / dt)
    while boundary is None and ticks < max_ticks:
        before = (actor.x, actor.y)
        state.apply_action(ACTOR, action, dt)
        state.update(dt)
        if action.direction is not None and (actor.x, actor.y) == before:
            rejected_steps += 1
        ticks += 1
        obs = build_observation(state, ACTOR)
        action, boundary = executor.continue_(obs)
        last_pos = (actor.x, actor.y)
    obs = build_observation(state, ACTOR)
    near = obs.nearby_object_id
    dist = distance_to_rect(actor.x, actor.y, target.collision_rect)
    legal = {"inspect": near == target_id, "interact": near == target_id}
    if boundary == PATH_UNREACHABLE:
        klass = "PATH_FAILED"
    elif boundary is None:
        klass = "COLLISION_PREVENTED_APPROACH" if rejected_steps > ticks // 2 else "STOPPED_TOO_EARLY"
    elif boundary == TARGET_REACHED and near == target_id:
        klass = "OK"
    elif boundary == TARGET_REACHED:
        klass = "APPROACH_POINT_INVALID" if approach is not None and distance_to_rect(*approach, target.collision_rect) > constants.INTERACTION_DISTANCE else "INTERACTION_RANGE_MISMATCH"
    else:
        klass = f"OTHER:{boundary}"
    return {
        "target_id": target_id, "target_type": target.type, "start": [round(v, 1) for v in start], "dt": round(dt, 4),
        "target_bounds": list(target.collision_rect), "approach_point": approach, "path_len": len(path),
        "final": [round(v, 1) for v in last_pos], "boundary": boundary, "ticks": ticks, "rejected_steps": rejected_steps,
        "distance_to_target": round(dist, 1), "interaction_range": constants.INTERACTION_DISTANCE,
        "near_object_id": near, "inspect_legal": legal["inspect"], "interact_legal": legal["interact"], "class": klass,
    }


def reachable_somehow(state: GameState, target_id: str, start: tuple[float, float]) -> bool:
    """Is there ANY walkable cell within interaction range of the target reachable from start?"""
    grid = state.nav_grid
    rect = state.level.get_object(target_id).collision_rect
    reach = constants.INTERACTION_DISTANCE
    for row in range(grid.rows):
        for col in range(grid.cols):
            if not grid.is_walkable((row, col)):
                continue
            cx, cy = grid.center_of((row, col))
            if distance_to_rect(cx, cy, rect) <= reach and grid.find_path(start, (cx, cy)):
                return True
    return False


def audit(per_room: int = 4, seed: int = 11) -> dict[str, Any]:
    rng = random.Random(seed)
    state = fresh_state()
    targets = [o for o in state.level.interactables() if o.type != "door"]
    rooms = {o.id: state.level.room_id_at(*o.center) for o in targets}
    starts = {room.id: start_positions(state, room.id, per_room, rng) for room in state.level.rooms}
    rows = []
    for obj in targets:
        room = rooms[obj.id]
        # same room (typical LLM case) + one start from every other room
        cases = list(starts.get(room, []))
        cases += [starts[r][0] for r in starts if r != room and starts[r]]
        for start in cases:
            for dt in DTS:
                row = run_move(fresh_state(), obj.id, start, dt)
                row["target_room"] = room
                if row["class"] == "PATH_FAILED" and not reachable_somehow(fresh_state(), obj.id, start):
                    row["class"] = "UNREACHABLE_EXPECTED"
                rows.append(row)
    # legality gap with the default door states: legal MOVE targets that are unreachable
    default = fresh_state(doors_open=False)
    from src.llm_controller import enumerate_legal_actions
    obs = build_observation(default, ACTOR)
    spawn = (default.actors[ACTOR].x, default.actors[ACTOR].y)

    def unreachable(legal):
        return [tid for tid in sorted(a["target_id"] for a in legal if a["action"] == "move")
                if default.level.get_object(tid).type != "door" and not reachable_somehow(default, tid, spawn)]

    legacy_legal = enumerate_legal_actions(obs)  # the recorded M5/M6 surface
    live_legal = enumerate_legal_actions(obs, reachable_only=True)  # what the live controllers send
    legal_moves = sorted(a["target_id"] for a in live_legal if a["action"] == "move")
    unreachable_legal = unreachable(live_legal)
    legacy_unreachable = unreachable(legacy_legal)
    by_dt = {}
    for dt in DTS:
        subset = [r for r in rows if r["dt"] == round(dt, 4)]
        by_dt[f"{dt:.4f}"] = dict(Counter(r["class"] for r in subset))
    return {"cases": len(rows), "by_dt": by_dt, "by_class": dict(Counter(r["class"] for r in rows)),
            "legal_move_targets_from_white_spawn_default_doors": legal_moves,
            "legal_but_unreachable_from_white_spawn": unreachable_legal,
            "legacy_surface_legal_but_unreachable_from_white_spawn": legacy_unreachable,
            "failures": [r for r in rows if r["class"] not in ("OK", "UNREACHABLE_EXPECTED")]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.navigation_audit")
    parser.add_argument("--out", help="directory for audit.json (default: print a summary only)")
    parser.add_argument("--per-room", type=int, default=4)
    args = parser.parse_args(argv)
    report = audit(per_room=args.per_room)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=1))
    failures = report["failures"]
    print(f"failures: {len(failures)}")
    for r in failures[:15]:
        print(" ", r["class"], r["target_id"], "dt", r["dt"], "start", r["start"], "final", r["final"], "near", r["near_object_id"],
              "dist", r["distance_to_target"], "boundary", r["boundary"], "approach", r["approach_point"])
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
