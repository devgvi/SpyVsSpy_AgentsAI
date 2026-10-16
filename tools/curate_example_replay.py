"""Make a replay bundle shareable: no local path, no provider profile name; then prove it still reproduces.

    python -m tools.curate_example_replay SRC [DEST]       # DEST defaults to SRC (rewritten in place)

Presentation-level only. Absolute paths inside the project (including its former folder name ``spy_vs_ai``) become
project-relative; any other absolute local path is an error. ``aws_profile`` values (provider options) are removed.
The engine frames, events and verification data are untouched, and the curated bundle is replayed offline: it is
written only if the verdict is REPLAY REPRODUCED. A ``curation`` record lists what changed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOTS = (str(ROOT), str(ROOT.parent / "spy_vs_ai"))  # the current folder and its name before the rename
LOCAL = re.compile(r"(/Users/|/home/|/private/|/var/folders/|/tmp/|[A-Za-z]:\\\\Users)")
PROFILE_KEYS = ("aws_profile",)


def curate(data: dict) -> dict:
    stats = {"local_paths_made_relative": 0, "provider_option_values_removed": []}

    def fix(value, path):
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if k in PROFILE_KEYS and isinstance(v, str) and v:
                    stats["provider_option_values_removed"].append(f"{path}/{k}")
                    out[k] = None
                else:
                    out[k] = fix(v, f"{path}/{k}")
            return out
        if isinstance(value, list):
            return [fix(v, path) for v in value]
        if isinstance(value, str) and LOCAL.search(value):
            new = value
            for root in PROJECT_ROOTS:
                if root + "/" in new:
                    stats["local_paths_made_relative"] += new.count(root + "/")
                    new = new.replace(root + "/", "")
                elif new == root:
                    stats["local_paths_made_relative"] += 1
                    new = "."
            if LOCAL.search(new):
                raise ValueError(f"an absolute local path outside the project at {path}: {new[:120]}")
            return new
        return value

    body = {k: v for k, v in data.items() if k not in ("_path", "curation")}
    curated = fix(body, "")
    stats["provider_option_values_removed"] = sorted(set(stats["provider_option_values_removed"]))
    curated["curation"] = {"tool": "tools.curate_example_replay", **stats,
                           "note": "presentation-level only: frames, engine events and verification data unchanged"}
    return curated


def reproduces(path: Path) -> dict:
    from spy_vs_ai.runtime import playback as pb
    return pb.Playback(pb.load(path)).run_to_end()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src")
    p.add_argument("dest", nargs="?")
    args = p.parse_args(argv)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.chdir(ROOT)
    src = Path(args.src).resolve()
    dest = Path(args.dest).resolve() if args.dest else src
    data = json.loads(src.read_text(encoding="utf-8"))
    curated = curate(data)
    text = json.dumps(curated, sort_keys=True, default=str) + "\n"
    with tempfile.TemporaryDirectory(prefix="curate-") as tmp:
        trial = Path(tmp) / dest.name
        trial.write_text(text, encoding="utf-8")
        result = reproduces(trial)
    from spy_vs_ai.runtime import playback as pb
    c = curated["curation"]
    print(f"{dest.name}: {result['verdict']}, {result.get('provider_calls', 0)} provider calls, "
          f"{c['local_paths_made_relative']} local path(s) made relative, "
          f"{len(c['provider_option_values_removed'])} provider option value(s) removed")
    if result["verdict"] != pb.REPRODUCED:
        print("not written: the curated bundle does not reproduce")
        return 1
    dest.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
