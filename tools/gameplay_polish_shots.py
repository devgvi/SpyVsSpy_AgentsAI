"""Screenshots for the gameplay-polish visual review (FINAL-2 spec 41): offscreen, call-free, no provider.

    SDL_VIDEODRIVER=dummy python -m tools.gameplay_polish_shots OUT_DIR [--levels level_01 level_02 level_03]

Per level: the whole-map overview with the room dressing (enhanced gameplay on), the same map without it, and the
split view after a few seconds of call-free play. Then, on level_01, one shot per door-trap variant at the moment an
opponent crosses the trapped doorway (explosive, electric, gas, snare, alarm), and the owner's / a detector's marker.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]

CROSS_DOOR = "door_corridor_main_office"  # level_01: corridor (north) / main office (south), open from the start


def _config(level: str, enhanced: bool):
    from spy_vs_ai.runtime import config as rc
    return rc.from_dict({"black": {"policy": "b2_v1", "personality": "saboteur"},
                         "white": {"policy": "b2_v1", "personality": "racer", "identity": "gpt61sol"},
                         "level": level, "seed": 42, "mode": "evaluation", "view": "play", "time_limit": 300,
                         "enhanced_gameplay": enhanced})


def _game(cfg, *, replay: bool = False):
    from spy_vs_ai.runtime.session import RuntimeGame, load_bundles, to_live_config
    from workshop.live.overlay import TelemetryView
    game = RuntimeGame(to_live_config(cfg), None, runtime_config=cfg, bundles=None if replay else load_bundles(cfg),
                       record=False, replay=replay)
    r = game.renderer
    r.game, r.view = game, TelemetryView(game.live_config)
    return game


def _shot(game, path: Path, *, overview: bool, split: bool = True) -> None:
    import pygame
    r = game.renderer
    r.overview, r.split = overview, split
    r.render(game.state, 0.0, None, None, game.bot_controller)
    pygame.image.save(game.screen, str(path))


def level_shots(level: str, out: Path) -> list[Path]:
    from workshop.policy_emulator import stage4_level_select as ls
    written = []
    with ls.using(level):
        for enhanced, tag in ((False, "plain"), (True, "dressed")):
            game = _game(_config(level, enhanced))
            path = out / f"{level}_{tag}_overview.png"
            _shot(game, path, overview=True)
            written.append(path)
        game = _game(_config(level, True))
        for _ in range(6 * 60):
            game._step(1 / 60)
        path = out / f"{level}_dressed_split.png"
        _shot(game, path, overview=False)
        written.append(path)
    return written


def trap_shots(out: Path) -> list[Path]:
    """Each variant on the same doorway: Black owns it, White walks south through it; the shot is taken a few frames
    after the crossing (the effect playing). The engine is driven directly (no agent): MOVE actions + update."""
    from src import constants
    from src.actions import ActionType, GameAction
    from workshop.policy_emulator import stage4_level_select as ls
    written = []
    black, white = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID
    with ls.using("level_01"):
        for variant in ("explosive", "electric", "gas", "snare", "alarm"):
            game = _game(_config("level_01", True), replay=True)
            state, resolver = game.state, game.traps
            door = state.level.get_object(CROSS_DOOR)
            cx, top = door.collision_rect.centerx, door.collision_rect.top
            state.actors[black].x, state.actors[black].y = cx - 70, top - 60
            state.actors[white].x, state.actors[white].y = cx + 10, top - 40
            resolver._prev = {a: (s.x, s.y) for a, s in state.actors.items()}
            resolver.instantiate(black, CROSS_DOOR, variant, source_trap_type="bomb")
            r = game.renderer
            r.split, r.overview = False, False
            frames_after = None
            for frame in range(240):
                state.apply_action(black, GameAction(ActionType.WAIT), 1 / 60)
                state.apply_action(white, GameAction(ActionType.MOVE, direction=(0.0, 1.0)), 1 / 60)
                state.update(1 / 60)
                r.render(state, 1 / 60, None, None, None)
                if frames_after is None and resolver.traps and not next(iter(resolver.traps.values())).armed:
                    frames_after = frame
                if frames_after is not None and frame - frames_after >= 14:
                    break
            path = out / f"door_trap_{variant}.png"
            _shot(game, path, overview=False, split=False)
            written.append(path)
        # markers: the owner sees its trap (left pane); a detector sees it with "!" (right pane)
        game = _game(_config("level_01", True), replay=True)
        state, resolver = game.state, game.traps
        door = state.level.get_object(CROSS_DOOR)
        state.actors[black].x, state.actors[black].y = door.collision_rect.centerx - 60, door.collision_rect.top - 70
        state.actors[white].x, state.actors[white].y = door.collision_rect.centerx + 60, door.collision_rect.bottom + 70
        trap = resolver.instantiate(black, CROSS_DOOR, "gas", source_trap_type="electric")
        other = resolver.instantiate(white, "door_main_office_ambassador", "snare", source_trap_type="snare")
        other.detected_by[black] = 0.0
        trap.detected_by[white] = 0.0
        state.update(0.0)
        path = out / "door_trap_markers_split.png"
        _shot(game, path, overview=False, split=True)
        written.append(path)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out")
    p.add_argument("--levels", nargs="*", default=["level_01", "level_02", "level_03"])
    p.add_argument("--no-traps", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    written = []
    for level in args.levels:
        written += level_shots(level, out)
    if not args.no_traps:
        written += trap_shots(out)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
