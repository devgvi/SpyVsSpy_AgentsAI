"""Room dressing report (FINAL-2 gameplay polish): props per room and the path / door validation, per level.

    SDL_VIDEODRIVER=dummy python -m tools.dressing_report [level_01 level_02 level_03] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def report(level_id: str) -> dict:
    from spy_vs_ai.runtime import dressing
    from src.level import load_level
    from workshop.policy_emulator import stage4_level_select as ls
    level = load_level(ls.level_path(level_id))
    started = time.monotonic()
    routes = dressing.route_rects(level)
    props = dressing.plan(level, routes=routes)
    check = dressing.validate(level, props, routes=routes)
    return {"level": level_id, "seconds": round(time.monotonic() - started, 2), "props": [p.public() for p in props],
            "rooms": dressing.summary(level, props), "validation": check}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("levels", nargs="*", default=["level_01", "level_02", "level_03"])
    p.add_argument("--json")
    args = p.parse_args(argv)
    out = [report(level) for level in args.levels]
    for r in out:
        v = r["validation"]
        print(f"{r['level']}: {len(r['props'])} props in {r['seconds']} s - validation {'PASS' if v['ok'] else 'FAIL'}"
              f", exit reachable {v['exit_reachable']}")
        for room, s in r["rooms"].items():
            print(f"    {room:<20} level objects {s['level_objects']}  +props {s['props']}  (target {s['target']})  "
                  f"{', '.join(s['families'])}")
        for name, reach in v["reachability"].items():
            print(f"    reachability ({name}): {reach['reachable_with_props']}/{reach['reachable']} kept, lost {reach['lost']}")
        for problem in v["problems"]:
            print(f"    PROBLEM {problem}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0 if all(r["validation"]["ok"] for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
