"""Loads and caches sprite images, with a placeholder fallback.

If a PNG referenced by level or item data is missing, callers get None back
and fall back to drawing a colored rectangle instead - see renderer.py. This
keeps the game fully playable before final art exists (spec section 1).

Every source PNG in this project is a 64x64 canvas with inconsistent
transparent padding around the actual artwork (an artifact of the
individually-generated icon pack). `_cropped()` trims that padding once;
`get_fitted()` and `get_tile()` both build on it so nothing ever
independently stretches a sprite's width and height - Milestone 2.6's
"never stretch a sprite to fill its collision rectangle" rule.
"""

from pathlib import Path

import pygame

_image_cache: dict[str, pygame.Surface | None] = {}
_cropped_cache: dict[str, pygame.Surface | None] = {}
_fitted_cache: dict[tuple[str, int, int], pygame.Surface] = {}
_tile_cache: dict[tuple[str, int], pygame.Surface | None] = {}


def load_image(path: str | None) -> pygame.Surface | None:
    if not path:
        return None

    if path not in _image_cache:
        image = None
        if Path(path).is_file():
            try:
                image = pygame.image.load(path).convert_alpha()
            except pygame.error:
                image = None
        _image_cache[path] = image

    return _image_cache[path]


def _cropped(path: str | None) -> pygame.Surface | None:
    """The sprite trimmed to its opaque bounding box, unscaled."""
    if not path:
        return None

    if path not in _cropped_cache:
        image = load_image(path)
        if image is not None:
            bounds = pygame.mask.from_surface(image).get_bounding_rects()
            if bounds:
                crop = bounds[0]
                for rect in bounds[1:]:
                    crop = crop.union(rect)
                image = image.subsurface(crop).copy()
        _cropped_cache[path] = image

    return _cropped_cache[path]


def get_fitted(path: str | None, max_size: tuple[float | None, float | None]) -> pygame.Surface | None:
    """The sprite's cropped content, scaled uniformly - preserving its
    aspect ratio, never stretching width and height independently.

    Either component of `max_size` may be None:
    - both given: contain-fit within the box (may end up smaller than
      max_size along one axis - callers anchor/center the result, e.g.
      via `surface.get_rect(midbottom=box.midbottom)`)
    - only one given: the other is derived from the sprite's real aspect
      ratio, so the result matches that one dimension exactly - only sound
      when the source art's own aspect ratio already suits the target (a
      landscape sprite fit by width, a portrait one fit by height).
      Forcing the wrong axis onto a mismatched sprite inflates the derived
      dimension instead of shrinking it - e.g. fitting a portrait door
      sprite by width alone once made M4A's horizontal-wall doors balloon
      to ~4x the wall's own thickness (Milestone_4A_1_FIXES.md). Prefer
      giving both dimensions for anything whose aspect ratio you haven't
      actually checked.
    - neither given: the cropped sprite at its native size
    """
    content = _cropped(path)
    if content is None:
        return None

    cw, ch = content.get_size()
    max_w, max_h = max_size

    if max_w is None and max_h is None:
        scale = 1.0
    elif max_w is None:
        scale = max(1, int(max_h)) / ch
    elif max_h is None:
        scale = max(1, int(max_w)) / cw
    else:
        scale = min(max(1, int(max_w)) / cw, max(1, int(max_h)) / ch)

    size = (max(1, round(cw * scale)), max(1, round(ch * scale)))

    key = (path, *size)
    if key not in _fitted_cache:
        _fitted_cache[key] = content if size == (cw, ch) else pygame.transform.scale(content, size)
    return _fitted_cache[key]


def get_tile(path: str | None, target_size: int | None = None) -> pygame.Surface | None:
    """A repeating texture, cropped to its opaque content and optionally
    scaled so its longest edge is `target_size` - used to turn an
    obviously large repeated pattern into a subtler one.
    """
    if not path:
        return None

    key = (path, target_size or 0)
    if key not in _tile_cache:
        tile = _cropped(path)
        if tile is not None and target_size is not None:
            cw, ch = tile.get_size()
            scale = target_size / max(cw, ch)
            size = (max(1, round(cw * scale)), max(1, round(ch * scale)))
            tile = pygame.transform.scale(tile, size)
        _tile_cache[key] = tile

    return _tile_cache[key]


_scaled_to_cache: dict[tuple[str, int, int], pygame.Surface | None] = {}


def get_scaled_to(path: str | None, size: tuple[int, int]) -> pygame.Surface | None:
    """The whole image (no crop) scaled to `size`, nearest neighbour. For
    art drawn to fill its canvas exactly (the doors_v2 set)."""
    if not path:
        return None
    key = (path, int(size[0]), int(size[1]))
    if key not in _scaled_to_cache:
        image = load_image(path)
        _scaled_to_cache[key] = None if image is None else pygame.transform.scale(image, key[1:])
    return _scaled_to_cache[key]
