"""Visual gap at real interactions (BUGFIX live flow, Part B §23-24).

The engine decides "in range" from the actor's feet point (x, y) - the
centre of its 28x28 collision box - to the object's *collision* rect,
limit `constants.INTERACTION_DISTANCE`. The viewer sees two sprites. This
audit measures, for every INSPECT / INTERACT / PLACE_TRAP / DISARM a live
run actually executed, both sides with the production renderer's own
placement code and the real sprite images:

* engine distance (feet -> collision rect);
* actor collision rect and actor render rect (visible pixels of the idle
  sprite for the recorded facing, bottom-anchored at the feet);
* object collision rect and object render rect (visible pixels);
* the screen-space gap between the two *visible* sprites (0 = touching or
  overlapping), in world px and in screen px (world px x CAMERA_ZOOM);
* where the actor stood relative to the object (above / below / left /
  right), since the sprite is drawn entirely above the feet.

Each interaction is classified:

* TRUE_ENGINE_RANGE_BUG - the engine let it happen beyond its own range;
* OBJECT_VISUAL_BOUNDS_MISMATCH - the object's collision rect reaches well
  outside its visible sprite, so "near the rect" is not "near the art";
* RENDER_ANCHOR_MISMATCH - the actor's sprite is drawn away from its
  collision/feet point (in this project: entirely above the feet), which
  is what opens the gap;
* OVERSIZED_INTERACTION_RANGE - sprites are where the engine says, the gap
  is simply the allowed range;
* VISUALLY_CLOSE - gap within VISUAL_TOUCH_PX.

    python -m src.evaluation.interaction_visual_audit RUN_DIR [...] [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from src import assets, constants  # noqa: E402
from src.interaction_range import distance_to_rect  # noqa: E402
from src.level import load_level  # noqa: E402

INTERACT_ACTIONS = ("inspect", "interact", "place_trap", "disarm")
VISUAL_TOUCH_PX = 12  # world px between visible sprites that still reads as "at the object"
BOUNDS_MISMATCH_PX = 12  # collision rect sticking out of the visible art by more than this


def _renderer():
    from src.renderer import Renderer
    pygame.init()
    screen = pygame.display.set_mode((constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT))
    return Renderer(screen)


def _visible(image: pygame.Surface, placed: pygame.Rect) -> pygame.Rect:
    bounds = image.get_bounding_rect()
    return pygame.Rect(placed.x + bounds.x, placed.y + bounds.y, bounds.width, bounds.height)


def rect_gap(a: pygame.Rect, b: pygame.Rect) -> float:
    dx = max(b.left - a.right, a.left - b.right, 0)
    dy = max(b.top - a.bottom, a.top - b.bottom, 0)
    return math.hypot(dx, dy)


def _side(x: float, y: float, rect: pygame.Rect) -> str:
    cx = max(rect.left, min(x, rect.right))
    cy = max(rect.top, min(y, rect.bottom))
    dx, dy = x - cx, y - cy
    if abs(dy) >= abs(dx):
        return "below" if dy > 0 else ("above" if dy < 0 else "inside")
    return "right" if dx > 0 else "left"


def actor_render_rect(actor_id: str, x: float, y: float, facing: str) -> pygame.Rect:
    """Visible pixels of the idle frame exactly as the renderer draws it
    (src/character_art.py: one canvas, one scale, feet anchor)."""
    from src import character_art

    return character_art.visible_rect(actor_id, x, y, facing)


def actor_collision_rect(x: float, y: float) -> pygame.Rect:
    w, h = constants.PLAYER_COLLISION_SIZE
    return pygame.Rect(int(x - w / 2), int(y - h / 2), int(w), int(h))


def object_render_rect(renderer, obj) -> pygame.Rect:
    from src.renderer import DOOR_ART_DIR, _door_sprite_path  # the renderer's own sprite choice (any object type)
    path = _door_sprite_path(obj, "closed" if obj.type == "door" else None)
    if path and path.startswith(DOOR_ART_DIR):
        return obj.collision_rect.copy()  # doors_v2 art fills the doorway exactly (Renderer._draw_object)
    image = assets.get_fitted(path, obj.render_max_size)
    placed = renderer._placement_rect(obj, image)
    return _visible(image, placed) if image is not None else placed


def measure(renderer, obj, actor_id: str, x: float, y: float, facing: str) -> dict[str, Any]:
    collision = obj.collision_rect
    engine = distance_to_rect(x, y, collision)
    actor_art = actor_render_rect(actor_id, x, y, facing)
    obj_art = object_render_rect(renderer, obj)
    gap = rect_gap(actor_art, obj_art)
    # how far the collision rect reaches beyond the visible art
    overhang = max(obj_art.left - collision.left, collision.right - obj_art.right,
                   obj_art.top - collision.top, collision.bottom - obj_art.bottom, 0)
    # the gap the viewer would see if the actor art were centred on the feet
    centred = actor_art.copy()
    centred.center = (int(x), int(y))
    gap_if_centred = rect_gap(centred, obj_art)
    side = _side(x, y, collision)
    if engine > constants.INTERACTION_DISTANCE + 0.5:
        cls = "TRUE_ENGINE_RANGE_BUG"
    elif gap <= VISUAL_TOUCH_PX:
        cls = "VISUALLY_CLOSE"
    elif overhang > BOUNDS_MISMATCH_PX and rect_gap(actor_collision_rect(x, y), collision) + BOUNDS_MISMATCH_PX < gap:
        cls = "OBJECT_VISUAL_BOUNDS_MISMATCH"
    elif gap - gap_if_centred > VISUAL_TOUCH_PX:
        cls = "RENDER_ANCHOR_MISMATCH"
    else:
        cls = "OVERSIZED_INTERACTION_RANGE"
    return {
        "object_id": obj.id, "actor_id": actor_id, "actor_feet": [round(x, 1), round(y, 1)], "facing": facing,
        "side_of_object": side, "engine_distance_px": round(engine, 1),
        "actor_collision_rect": list(actor_collision_rect(x, y)), "actor_render_rect": list(actor_art),
        "object_collision_rect": list(collision), "object_render_rect": list(obj_art),
        "collision_overhang_px": overhang,
        "visual_gap_world_px": round(gap, 1), "visual_gap_screen_px": round(gap * constants.CAMERA_ZOOM, 1),
        "visual_gap_if_sprite_centred_px": round(gap_if_centred, 1),
        "classification": cls,
    }


def analyse(run_dirs: list[Path]) -> dict[str, Any]:
    renderer = _renderer()
    level = load_level(constants.LEVEL_PATH)
    objects = {o.id: o for o in level.objects}
    rows = []
    for run_dir in run_dirs:
        for f in sorted(run_dir.glob("telemetry_match*.jsonl")):
            for line in f.open():
                e = json.loads(line)
                if e.get("event_type") != "llm_decision":
                    continue
                act = e.get("executed_action") or {}
                if act.get("action") not in INTERACT_ACTIONS or act.get("target_id") not in objects:
                    continue
                me = e["observation"]["self"]
                x, y = me["position"]
                row = measure(renderer, objects[act["target_id"]], e["actor_id"], x, y, me.get("facing") or "down")
                row["action"] = act["action"]
                rows.append(row)
    return summarize(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values):
        values = sorted(values)
        if not values:
            return None
        return {"n": len(values), "median": round(statistics.median(values), 1), "max": round(max(values), 1),
                "p90": round(values[int(0.9 * (len(values) - 1))], 1)}
    by_side = collections.defaultdict(list)
    for r in rows:
        by_side[r["side_of_object"]].append(r["visual_gap_world_px"])
    return {
        "interactions": len(rows),
        "interaction_distance_px": constants.INTERACTION_DISTANCE,
        "camera_zoom": constants.CAMERA_ZOOM,
        "engine_distance_px": stats([r["engine_distance_px"] for r in rows]),
        "visual_gap_world_px": stats([r["visual_gap_world_px"] for r in rows]),
        "visual_gap_screen_px": stats([r["visual_gap_screen_px"] for r in rows]),
        "visual_gap_if_sprite_centred_px": stats([r["visual_gap_if_sprite_centred_px"] for r in rows]),
        "visual_gap_by_side_world_px": {k: stats(v) for k, v in sorted(by_side.items())},
        "classification": dict(collections.Counter(r["classification"] for r in rows).most_common()),
        "rows": rows,
    }


def approach_audit(per_room: int = 4, seed: int = 11, dt: float = 1 / 60) -> dict[str, Any]:
    """MOVE_TO every furniture object from the navigation audit's start
    positions with the production engine + ActionExecutor, once with the
    evaluation completion (stop on the first frame in range) and once with
    the live close approach; measure the visual gap where each MOVE ends."""
    import random

    from src.agents.action_lifecycle import ActionExecutor, TARGET_REACHED
    from src.evaluation import navigation_audit as nav
    from src.observation import build_observation

    renderer = _renderer()
    rng = random.Random(seed)
    base = nav.fresh_state()
    targets = [o for o in base.level.interactables() if o.type != "door"]
    starts = {room.id: nav.start_positions(base, room.id, per_room, rng) for room in base.level.rooms}
    result = {}
    for mode, close in (("evaluation_first_frame_in_range", False), ("live_close_approach", True)):
        rows = []
        for obj in targets:
            room = base.level.room_id_at(*obj.center)
            for start in starts.get(room, []):
                state = nav.fresh_state()
                actor = state.actors[nav.ACTOR]
                actor.x, actor.y = start
                state.update(0.0)
                obs = build_observation(state, nav.ACTOR)
                executor = ActionExecutor()
                executor.close_approach = close
                action, boundary, _ = executor.start(obs, "d", "move", obj.id, None, obs.mission_time_remaining)
                ticks = 0
                while boundary is None and ticks < int(nav.TICK_BUDGET_SECONDS / dt):
                    state.apply_action(nav.ACTOR, action, dt)
                    state.update(dt)
                    ticks += 1
                    obs = build_observation(state, nav.ACTOR)
                    action, boundary = executor.continue_(obs)
                obs = build_observation(state, nav.ACTOR)
                if boundary != TARGET_REACHED or obs.nearby_object_id != obj.id:
                    rows.append({"object_id": obj.id, "boundary": boundary, "classification": "NOT_REACHED"})
                    continue
                row = measure(renderer, obj, nav.ACTOR, actor.x, actor.y, actor.facing)
                row["ticks"] = ticks
                rows.append(row)
        reached = [r for r in rows if r["classification"] != "NOT_REACHED"]
        summary = summarize(reached)
        summary["not_reached"] = len(rows) - len(reached)
        summary.pop("rows")
        result[mode] = summary
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--approach", action="store_true", help="synthetic MOVE_TO audit, evaluation vs live close approach")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.approach:
        result = approach_audit()
        text = json.dumps(result, indent=1, sort_keys=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    result = analyse([Path(r) for r in args.runs])
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    brief = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(brief, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
