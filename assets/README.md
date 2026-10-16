# Assets

The game art, almost all PNG. There is no audio: nothing uses `pygame.mixer`.

**Keep the folder names.** The paths are named by the frozen engine (`src/constants.py`, `src/renderer.py`,
`src/character_art.py`), the level files (`data/level_01.json`, `workshop/levels/level_0{2,3}.json`) and the runtime
(`spy_vs_ai/`). A missing file never crashes the game: `src/assets.py` returns nothing and the renderer draws a
placeholder. New art is added beside the old, and the old stays for compatibility.

## In use

| Folder | Files | Used by | Notes |
|---|---|---|---|
| `characters_v4/` | 368 + `frames.json` | `src/character_art.py` (default set) | the two spies: 12 states × 4 facings, 2-8 frames each, on a 160×120 canvas with the feet anchor at (80, 115); built offline from the source sprite sheets |
| `doors_v2/` | 10 | `src/renderer.py` (`DOOR_ART_DIR`) | every door and the exit, sized to its doorway |
| `floors/floor.png`, `floors/wall.png` | 2 | `src/constants.py` (`FLOOR_TILE_SPRITE`, `WALL_TILE_SPRITE`) | floor and wall tiles |
| `environment/` | 13 | the level files | the searchable, interactive furniture: armchair, bookshelf, chair, desk, filing cabinet, globe, lamp, painting, plants, side table, sofa, statue bust |
| `items/` | 3 | `data/items.json`, `data/level_01.json` | inventory icons: documents, embassy key, passport |
| `traps/` | 5 | `src/constants.py` (`TRAP_ICON_SPRITES`), `spy_vs_ai/ui/enhanced.py` | trap icons: bomb, electric, snare, alarm, and gas (used only by the door-trap markers) |
| `effects/` | 8 per trap | `src/constants.py` (`TRAP_EFFECT_FRAMES`), `spy_vs_ai/ui/enhanced.py` | 8-frame trap animations: bomb, electric, snare, alarm, gas |
| `props/` | 43 | `spy_vs_ai/runtime/dressing.py` | decorative room dressing, drawn by the renderer only: never a level object, never in an observation |

## Kept for compatibility

These are named by frozen code or data, but not drawn by default.

| Folder | Files | Why it stays |
|---|---|---|
| `characters_v2/` | 248 | the earlier 64×64 spy set; `src/character_art.py` falls back to it if `characters_v4/frames.json` is missing |
| `characters/` | 48 | the oldest spy set (6 poses × 4 facings), named by `src/constants.py` (`PLAYER_SPRITE_DIR`) |
| `doors/` | 12 | the door sprite paths written in the level files (the renderer draws `doors_v2/` instead) |

## Provenance

The spies, furniture and props were cut from four approved sprite atlases and cleaned offline. Some spy frames, the
floor and wall tiles, eight props, the gas effect and the gas trap icon were generated with an image model, in the
style of the existing sprites.
