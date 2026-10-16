"""Stage 4 level selection, kept outside the directories the frozen Stage-3 game-code fingerprint covers
(`src/`, `workshop/live/`, `data/` - workshop/scripts/tournament.code_fingerprint). Guard recovery G2,
Archive/STAGE4A_GUARD_RECOVERY_AND_STAGE4B_EXECUTION.md section 1.

The engine is untouched: `src/game.py` loads `constants.LEVEL_PATH` when a Game is built, exactly as in every
Stage-3 run. A Stage-4 match process (one process per match) points that module attribute at the level's file
before the game is built, and records every file `Game` loads. level_01 is the default and loads the unchanged
`data/level_01.json`; level_02 / level_03 live in `workshop/levels/` (byte-identical to the frozen Stage-4A
level files, same sha256).
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from workshop.runners.common import PROJECT_ROOT

DEFAULT_LEVEL = "level_01"
LEVEL_01_PATH = "data/level_01.json"  # src/constants.LEVEL_PATH, the frozen engine default
LEVELS: dict[str, str] = {
    "level_01": LEVEL_01_PATH,
    "level_02": "workshop/levels/level_02.json",
    "level_03": "workshop/levels/level_03.json",
}
# where the first Stage-4A run loaded the two new levels from (before guard recovery G2); same bytes
ORIGINAL_PATHS = {"level_02": "data/level_02.json", "level_03": "data/level_03.json"}
LOADS: list[dict[str, Any]] = []   # every level file Game loaded in this process (path, sha256, level id)
_STATE: dict[str, Any] = {"recording": False}


def level_path(level_id: str | None) -> str:
    """The project-relative file of a registered level (None -> level_01)."""
    level_id = level_id or DEFAULT_LEVEL
    if level_id not in LEVELS:
        raise ValueError(f"unknown level {level_id!r}. Available: {', '.join(LEVELS)}")
    return LEVELS[level_id]


def level_file(level_id: str | None) -> Path:
    return PROJECT_ROOT / level_path(level_id)


def file_sha256(level_id: str | None) -> str:
    return "sha256:" + hashlib.sha256(level_file(level_id).read_bytes()).hexdigest()


def _record_loads() -> None:
    """Wrap the name src.game uses to load the level, so each match records what it actually loaded."""
    if _STATE["recording"]:
        return
    import json

    import src.game as game_module
    original = game_module.load_level

    def recording_load_level(path: str):
        level = original(path)
        p = Path(path)
        p = p if p.is_absolute() else PROJECT_ROOT / p
        LOADS.append({"path": str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p),
                      "sha256": "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest(), "level_id": level.id,
                      "file_level_id": json.loads(p.read_text()).get("id")})
        return level
    game_module.load_level = recording_load_level
    _STATE["recording"] = True


def select(level_id: str | None) -> str:
    """Process-wide: every Game built afterwards in this process loads `level_id`. Returns the absolute path."""
    from src import constants
    if constants.LEVEL_PATH not in (LEVEL_01_PATH, *[str(PROJECT_ROOT / p) for p in LEVELS.values()]):
        raise RuntimeError(f"constants.LEVEL_PATH is {constants.LEVEL_PATH!r}: not a registered level")
    _record_loads()
    level_id = level_id or DEFAULT_LEVEL
    constants.LEVEL_PATH = LEVEL_01_PATH if level_id == DEFAULT_LEVEL else str(level_file(level_id))
    return str(level_file(level_id))


@contextmanager
def using(level_id: str | None) -> Iterator[str]:
    """In-process (validators, static checks): select a level, restore level_01 afterwards."""
    from src import constants
    before = constants.LEVEL_PATH
    try:
        yield select(level_id)
    finally:
        constants.LEVEL_PATH = before


def loaded_record(expected_level: str | None) -> dict[str, Any]:
    """What this process loaded, against what it should have loaded."""
    expected = level_path(expected_level)
    sha = file_sha256(expected_level)
    return {"expected_level": expected_level or DEFAULT_LEVEL, "expected_path": expected, "expected_sha256": sha,
            "loads": list(LOADS), "ok": bool(LOADS) and all(x["path"] == expected and x["sha256"] == sha for x in LOADS)}
