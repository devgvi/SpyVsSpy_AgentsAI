"""Watch a saved match again, offline (FINAL-2 spec 27, 28).

    python -m spy_vs_ai.replay runs/<run>/replay_match01.json            # the window, play / research views
    python -m spy_vs_ai.replay runs/<run>/replay_match01.json --headless --verify
    python -m spy_vs_ai.replay --historical workshop_runs/stage4c-confirmatory/matches/r0/level_02-9009-cautious-A
    python -m spy_vs_ai.replay --historical-examples                     # the four spec-28 examples

A replay makes 0 provider calls: it runs under the offline guard (no network connection or name lookup, no provider
client or SDK, credential-like environment variables removed by name before anything is imported), rebuilds the
recorded game with no agent and re-issues the recorded engine calls. At the end it checks the engine's event log, the
final spy states and the ranking against the recording: REPLAY REPRODUCED or REPLAY DIVERGED (exit code 1).

Keys: Space pause, + / - speed, V show / hide panels, M play / research view, I inspector, H history, Y AI telemetry, Tab spy,
Up / Down pick a decision, Home / End, T / V agent panels, P split, O overview, Z zoom, S screenshot, Esc quit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPS = 60


def _activate_guard(allowed: list[Path]):
    """Before any game / provider module is imported (the guard refuses to start otherwise)."""
    from workshop.policy_emulator.offline_guard import OfflineGuard
    return OfflineGuard().activate([Path(p).resolve() for p in allowed] + [Path(os.devnull)], scrub=True)


def _guard_line(guard) -> tuple[str, bool]:
    if guard is None:
        return "offline guard: not active (--no-guard)", True
    from workshop.policy_emulator.offline_guard import OfflineGuard
    rec = guard.record()
    clean = OfflineGuard.clean(rec)
    return (f"offline guard: provider calls 0, network attempts {len(rec['network_attempts'])}, provider clients "
            f"constructed {len(rec['provider_clients_constructed'])}, provider SDKs loaded {len(rec['provider_sdks_loaded'])}, "
            f"credential reads {len(rec['credential_env_accessed']) + len(rec['credential_files_opened'])}"
            + ("" if clean else " - NOT CLEAN")), clean


def interactive(pb, *, event_source=None, max_seconds: float | None = None, screenshot_dir: Path | None = None) -> dict:
    """The replay window: the recorded frames at the recorded pace (a long frozen stretch, while a model thought under
    the response barrier, is shortened), pause, speed, the play / research views."""
    import pygame
    from spy_vs_ai.runtime.playback import MAX_FROZEN_FRAMES_SHOWN
    from src import constants
    events = event_source or pygame.event.get
    game, renderer = pb.game, pb.game.renderer
    clock = pygame.time.Clock()
    speeds = (0.5, 1.0, 2.0, 4.0)
    speed = pb.config.speed if pb.config.speed in speeds else 1.0
    paused, acc, frozen_run, shots = False, 0.0, 0, 0
    started = time.monotonic()
    runtime_keys = (pygame.K_v, pygame.K_m, pygame.K_i, pygame.K_h, pygame.K_y, pygame.K_TAB, pygame.K_UP, pygame.K_DOWN,
                    pygame.K_HOME, pygame.K_END)
    try:
        while True:
            clock.tick(FPS)
            for event in events():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return pb.finish() if pb.done else {"verdict": "STOPPED", "frame": pb.i, "frames": len(pb.frames),
                                                        "provider_calls": 0}
                if event.type != pygame.KEYDOWN:
                    continue
                key = event.key
                if key == pygame.K_SPACE:
                    paused = not paused
                elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = speeds[min(speeds.index(speed) + 1, len(speeds) - 1)]
                elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = speeds[max(speeds.index(speed) - 1, 0)]
                elif key in runtime_keys:
                    renderer.handle_key(key)
                elif key == pygame.K_t:
                    renderer.show_telemetry = not renderer.show_telemetry
                elif key == pygame.K_p:
                    renderer.split, renderer.overview = not renderer.split, False
                elif key == pygame.K_o:
                    renderer.overview = not renderer.overview
                elif key == pygame.K_z:
                    renderer.cycle_zoom()
                elif key == pygame.K_s and screenshot_dir is not None:
                    shots += 1
                    pygame.image.save(game.screen, str(screenshot_dir / f"replay_screenshot_{shots:02d}.png"))
            dt = 0.0
            if not paused and not pb.done:
                acc += speed
                while acc >= 1.0 and not pb.done:
                    frozen = pb.next_frame_frozen()
                    pb.step()
                    if frozen:
                        frozen_run += 1
                        if frozen_run > MAX_FROZEN_FRAMES_SHOWN:
                            continue  # skip ahead through the rest of a frozen stretch
                    else:
                        frozen_run = 0
                    acc -= 1.0
                    dt += 1.0 / constants.FPS
                if pb.done:
                    pb.finish()
            renderer.paused, renderer.speed, renderer.match_over = paused, speed, pb.done
            renderer.render(game.state, dt, None, None, game.bot_controller)
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                return pb.finish() if pb.done else {"verdict": "STOPPED", "frame": pb.i, "frames": len(pb.frames),
                                                    "provider_calls": 0}
    finally:
        pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m spy_vs_ai.replay", description="Watch a saved match again, offline "
                                "(0 provider calls).")
    p.add_argument("path", nargs="?", help="a replay bundle (runs/<run>/replay_matchNN.json)")
    p.add_argument("--historical", metavar="MATCH_DIR", help="a frozen research match directory (Stages 4C / 5B / 5C): "
                   "re-simulated without a provider and shown only if it reproduces the historical engine events")
    p.add_argument("--historical-examples", action="store_true", help="list the spec-28 example matches and exit")
    p.add_argument("--save", metavar="PATH", help="--historical: also save the reproduced replay bundle here")
    p.add_argument("--headless", action="store_true", help="no window: re-issue every frame and verify")
    p.add_argument("--verify", action="store_true", help="exit code 1 unless the replay reproduces the recording")
    p.add_argument("--view", choices=("play", "research"), help="start in this view (default: the recorded one)")
    p.add_argument("--speed", type=float, choices=(0.5, 1.0, 2.0), help="playback speed")
    p.add_argument("--max-seconds", type=float, help="close the window after this many seconds")
    p.add_argument("--screenshot", metavar="PNG", help="--headless: save the final frame here")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--no-guard", action="store_true", help=argparse.SUPPRESS)  # tests only (the guard is process-wide)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.historical_examples:
        from spy_vs_ai.runtime.historical import EXAMPLES
        root = Path(__file__).resolve().parents[1]
        for label, path in EXAMPLES.items():
            print(f"{label:<46} python -m spy_vs_ai.replay --historical {path}")
        if not all((root / p).is_dir() for p in EXAMPLES.values()):
            print("These matches are in the research archive (workshop_runs/), not in this repository; "
                  "examples/replays/ holds them already re-simulated.")
        return 0
    if not args.path and not args.historical:
        build_parser().error("give a replay bundle path or --historical MATCH_DIR")
    if args.historical and os.environ.get("PYTHONHASHSEED") != "0":
        # the research matches ran with PYTHONHASHSEED=0: re-simulate under the same interpreter setting
        env = dict(os.environ, PYTHONHASHSEED="0")
        os.execve(sys.executable, [sys.executable, "-m", "spy_vs_ai.replay", *argv], env)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from spy_vs_ai.play import prepare_display
    try:
        prepare_display(args.headless)  # a visual replay needs a real window (see spy_vs_ai.play)
    except SystemExit as exc:
        print(f"[REPLAY] {exc}", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="spy-vs-ai-replay-"))
    allowed = [work, Path(tempfile.gettempdir())]
    for extra in (args.save, args.screenshot):
        if extra:
            allowed.append(Path(extra).expanduser().resolve().parent)
    if args.path:
        allowed.append(Path(args.path).expanduser().resolve().parent)  # S screenshots next to the bundle
    guard = None if args.no_guard else _activate_guard(allowed)
    from spy_vs_ai.runtime import playback as pbm  # only now: under the guard

    if args.historical:
        from spy_vs_ai.runtime.historical import HistoricalError, resimulate
        print(f"[REPLAY] re-simulating {args.historical} without a provider (frozen match process, fixed step) ...",
              flush=True)
        try:
            result = resimulate(args.historical, work / "resimulation")
        except HistoricalError as exc:
            print(f"[REPLAY] cannot replay this match: {exc}", file=sys.stderr)
            return 2
        c = result["comparison"]
        if result["verdict"] != "REPRODUCED":
            print(f"[REPLAY] HISTORICAL MATCH NOT REPRODUCED - not shown: engine events {c['regenerated_engine_events']} vs "
                  f"{c['historical_engine_events']} recorded, first difference at {c['first_difference']}, ranking "
                  f"{'identical' if c['ranking_identical'] else 'different'}", flush=True)
            print(_guard_line(guard)[0])
            if args.json:
                print(json.dumps({k: v for k, v in result.items() if k != "bundle"}, indent=1, default=str))
            return 1
        print(f"[REPLAY] historical engine events reproduced exactly ({c['historical_engine_events']} events, ranking "
              f"identical): {result['arm_label']} {result['match_id']}", flush=True)
        data = result["bundle"]
        if args.save:
            out = Path(args.save).expanduser()
            out.write_text(json.dumps(data, sort_keys=True, default=str) + "\n", encoding="utf-8")
            print(f"[REPLAY] saved {out}")
        data["_path"] = str(args.save or args.historical)
    else:
        try:
            data = pbm.load(args.path)
        except pbm.ReplayError as exc:
            print(f"[REPLAY] {exc}", file=sys.stderr)
            return 2
    try:
        pb = pbm.Playback(data, view=args.view, speed=args.speed)
    except Exception as exc:  # noqa: BLE001 - a bad bundle is reported, not a traceback
        print(f"[REPLAY] cannot build the recorded game: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for name, ok in pb.checks.items():
        if not ok:
            print(f"[REPLAY] warning: {name.replace('_', ' ')} is false - the recording was made with a different "
                  "level file / game code; the verification below decides", flush=True)
    if args.headless:
        result = pb.run_to_end()
        if args.screenshot:
            import pygame
            pb.game.renderer.render(pb.game.state, 0.0, None, None, pb.game.bot_controller)
            pygame.image.save(pb.game.screen, str(Path(args.screenshot).expanduser()))
            pygame.quit()
    else:
        result = interactive(pb, max_seconds=args.max_seconds,
                             screenshot_dir=Path(args.path).expanduser().resolve().parent if args.path else None)
    line, clean = _guard_line(guard)
    result["offline_guard_clean"] = clean
    verdict = result.get("verdict")
    print(f"[REPLAY] {'REPLAY ' + verdict if verdict in (pbm.REPRODUCED, pbm.DIVERGED) else verdict}: "
          f"{result.get('frames')} frames, {result.get('engine_events', '-')} engine events, "
          f"{result.get('decisions_shown', '-')} decisions shown, 0 provider calls", flush=True)
    print(f"[REPLAY] {line}", flush=True)
    if args.json:
        print(json.dumps(result, indent=1, default=str))
    if args.verify or args.headless:
        return 0 if verdict == pbm.REPRODUCED and clean else 1
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
