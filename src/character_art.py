"""The spies' animation frames (bugfix spawn/combat/visual polish, Part D).

Two built frame sets, both produced offline (no runtime image cleanup):

* assets/characters_v4/ (default) - the V4 art pack, cleaned and normalised
  by tools/build_v4_sprites.py: hard alpha, noise removed, every frame on one
  160x120 canvas at world resolution with the feet on one row. Frame counts
  differ per state and facing; they are read from frames.json.
* assets/characters_v2/ - the earlier 64x64 set (tools/build_live_sprites.py),
  drawn at CHARACTER_SCALE 1.5. Kept as the fallback when V4 is absent.

Either way every frame is drawn the same way: the canvas at one constant
scale (nearest neighbour), with canvas pixel ANCHOR on the actor's feet point
(x, y). No per-frame crop or fit, so a frame never makes a spy grow, shrink
or jump sideways.
"""
from __future__ import annotations

import json
from pathlib import Path

import pygame

from src import constants

V4_DIR = "assets/characters_v4"
V2_DIR = "assets/characters_v2"
STATES = ("idle", "walk", "inspect", "search", "place_trap", "disarm", "fight", "hit", "ko", "disappear",
          "respawn", "celebrate")
V2_FRAMES = {"idle": 1, "walk": 4, "inspect": 2, "search": 2, "place_trap": 2, "disarm": 2, "fight": 2,
             "hit": 2, "ko": 2, "disappear": 4, "respawn": 4, "celebrate": 4}
STATE_FRAMES = V2_FRAMES  # the v2 counts (tests of that set); use frame_count() for the active set
_PACKS = {constants.BLACK_SPY_ID: ("black_spy", "black"), constants.WHITE_SPY_ID: ("white_spy", "white")}


def _load_v4() -> dict | None:
    path = Path(V4_DIR) / "frames.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


_V4 = _load_v4()
if _V4 is not None:
    FRAMES_DIR = V4_DIR
    CANVAS = tuple(_V4["canvas"])
    CHARACTER_SCALE = 1.0  # built at world resolution
    ANCHOR = tuple(_V4["anchor"])
else:
    FRAMES_DIR = V2_DIR
    CANVAS = (64, 64)
    CHARACTER_SCALE = 1.5  # world px per sprite px: 78 px visible height
    ANCHOR = (32, 58)  # centre column, just under the baseline row 57
_scaled_cache: dict[str, pygame.Surface | None] = {}


def frame_count(actor_id: str, state: str, facing: str) -> int:
    if _V4 is not None:
        return _V4["frames"][_PACKS[actor_id][0]][state][facing]
    return V2_FRAMES[state]


def frame_path(actor_id: str, state: str, facing: str, index: int = 0) -> str:
    folder, prefix = _PACKS[actor_id]
    index %= frame_count(actor_id, state, facing)
    return f"{FRAMES_DIR}/{folder}/{prefix}_{state}_{facing}_{index}.png"


def scaled_frame(path: str) -> pygame.Surface | None:
    if path not in _scaled_cache:
        image = None
        if Path(path).is_file():
            try:
                raw = pygame.image.load(path).convert_alpha()
                size = (round(CANVAS[0] * CHARACTER_SCALE), round(CANVAS[1] * CHARACTER_SCALE))
                image = raw if size == raw.get_size() else pygame.transform.scale(raw, size)  # nearest neighbour
            except pygame.error:
                image = None
        _scaled_cache[path] = image
    return _scaled_cache[path]


def placed_rect(x: float, y: float) -> pygame.Rect:
    """World rect of the scaled canvas for an actor whose feet are at (x, y)."""
    w, h = round(CANVAS[0] * CHARACTER_SCALE), round(CANVAS[1] * CHARACTER_SCALE)
    return pygame.Rect(round(x - ANCHOR[0] * CHARACTER_SCALE), round(y - ANCHOR[1] * CHARACTER_SCALE), w, h)


def visible_rect(actor_id: str, x: float, y: float, facing: str, state: str = "idle", index: int = 0) -> pygame.Rect:
    """World rect of the frame's visible pixels (for audits and tests)."""
    placed = placed_rect(x, y)
    image = scaled_frame(frame_path(actor_id, state, facing, index))
    if image is None:
        return placed
    bounds = image.get_bounding_rect()
    return pygame.Rect(placed.x + bounds.x, placed.y + bounds.y, bounds.width, bounds.height)
