"""Acceptance demo of the gameplay polish (FINAL-2 spec 40): call-free, no provider, no network.

    SDL_VIDEODRIVER=dummy python -m tools.gameplay_polish_demo [--out runs/gameplay-polish-demo] [--seed 3]

One headless match on level_01 (B2-v1 saboteur vs B2-v2 cautious, both call-free surrogates; evaluation timing so the
match is deterministic and runs faster than real time, with the enhanced gameplay rules switched on):

    richer room -> a spy places a trap (PLACE_TRAP) -> the resolver puts it on a doorway -> a variant is chosen ->
    the opponent crosses the doorway -> the effect triggers -> the telemetry records it -> the replay saves it ->
    the offline replay (0 provider calls) reproduces it

Writes the run (manifest, telemetry, replay) plus demo_report.json and two screenshots (the dressed map, the moment
of the trigger, from the offline replay) into the output directory. Exit code 0 only if every step held.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]


def demo_config(seed: int, out: Path):
    from spy_vs_ai.runtime import config as rc
    return rc.from_dict({"black": {"policy": "b2_v1", "personality": "saboteur"},
                         "white": {"policy": "b2_v2", "personality": "cautious"},
                         "level": "level_01", "seed": seed, "mode": "evaluation", "view": "play", "time_limit": 120,
                         "enhanced_gameplay": True, "save_replay": True, "output_dir": str(out)})


def run(out: Path, seed: int = 3) -> dict:
    import pygame
    from spy_vs_ai.runtime import dressing, playback as pb
    from spy_vs_ai.runtime.session import RuntimeMatchRunner
    from workshop.policy_emulator import stage4_level_select as ls
    out.mkdir(parents=True, exist_ok=True)
    cfg = demo_config(seed, out)
    steps: dict[str, dict] = {}

    def step(name: str, ok: bool, detail) -> None:
        steps[name] = {"ok": bool(ok), "detail": detail}

    with ls.using(cfg.level):
        runner = RuntimeMatchRunner(cfg, out, headless=True, verbose=False)
        runner.run()
        game = runner.game
        props = dressing.props_for(game.state.level)
        check = dressing.validate(game.state.level, props)
        step("richer_room", len(props) > 0 and check["ok"],
             {"props": len(props), "per_room": dressing.summary(game.state.level, props), "validation": check["ok"]})
        records = game.traps.records
        placed = [r for r in records if r["kind"] == "placed"]
        step("spy_places_trap", bool(placed) or any(r["kind"] == "resolved" for r in records),
             {"placements": len(placed), "object_traps": sum(1 for r in records if r["kind"] == "resolved")})
        step("trap_resolves_to_doorway", bool(placed), [(r["owner"], r["target_id"]) for r in placed])
        step("variant_selected", bool(placed) and all(r.get("variant") for r in placed), [r["variant"] for r in placed])
        triggered = [r for r in records if r["kind"] == "triggered"]
        first = triggered[0] if triggered else None
        crossed = first is not None and first.get("cause") == "crossed_doorway" and first["victim"] != first["owner"]
        step("opponent_crosses_doorway", crossed, first)
        step("effect_triggers", first is not None and bool(first.get("effect")),
             {k: first.get(k) for k in ("variant", "effect", "cue", "victim", "game_time")} if first else None)
        telemetry_path = out / "telemetry_match01.jsonl"
        tel = [json.loads(line) for line in telemetry_path.read_text().splitlines() if line.strip()]
        trap_events = [e for e in tel if e.get("event_type") == "enhanced_trap"]
        step("telemetry_records_event", any(e.get("kind") == "triggered" for e in trap_events),
             {"file": str(telemetry_path), "enhanced_trap_events": len(trap_events)})
        replay_path = Path(runner.replays[0])
        bundle = json.loads(replay_path.read_text())
        saved = [e for e in bundle["telemetry"] if e.get("event_type") == "enhanced_trap"]
        step("replay_saves_event", bool(saved) and bundle.get("enhanced_gameplay") is True
             and bool((bundle.get("enhanced_traps") or {}).get("triggered")),
             {"replay": str(replay_path), "enhanced_trap_events": len(saved)})
    # the offline replay: no agent, no provider; screenshot the frame the first door trap goes off
    data = pb.load(replay_path)
    playback = pb.Playback(data)
    shot = None
    with ls.using(cfg.level):
        r = playback.game.renderer
        r.split, r.overview = True, False  # one pane per spy: the victim's pane shows the effect
        after = None
        while not playback.done:
            playback.step()
            traps = playback.game.traps
            if after is None and traps is not None and any(t.status == "TRIGGERED" for t in traps.traps.values()):
                after = playback.i
            if after is not None and playback.i == after + 12 and shot is None:
                r.render(playback.game.state, 1 / 60, None, None, None)
                shot = out / "demo_trigger.png"
                pygame.image.save(playback.game.screen, str(shot))
        result = playback.finish()
        r.overview = True
        r.render(playback.game.state, 0.0, None, None, None)
        overview = out / "demo_dressed_overview.png"
        pygame.image.save(playback.game.screen, str(overview))
    step("offline_replay_reproduces_event", result["verdict"] == pb.REPRODUCED
         and result["checks"].get("enhanced_traps_identical") is True and result["provider_calls"] == 0,
         {"verdict": result["verdict"], "checks": result["checks"], "provider_calls": result["provider_calls"]})
    report = {"demo": "FINAL-2 gameplay polish acceptance (call-free)", "config": cfg.public(),
              "ok": all(s["ok"] for s in steps.values()), "steps": steps,
              "screenshots": [str(p) for p in (shot, overview) if p is not None]}
    (out / "demo_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="runs/gameplay-polish-demo")
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args(argv)
    os.chdir(ROOT)
    report = run(Path(args.out).resolve(), args.seed)
    for name, s in report["steps"].items():
        print(f"{'PASS' if s['ok'] else 'FAIL'}  {name}")
    print(f"demo: {'PASS' if report['ok'] else 'FAIL'} - report {Path(args.out) / 'demo_report.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
