"""Headless tournaments for the live game (real model calls).

Two modes share one match runner.

**Self-play tournament.** Every pairing of the chosen personalities (each
against each, and each against itself) plays once per seed, with the same
model on both sides. The same seeds are used for every pairing and every
model, so each pairing meets the same maps, item placements and riddles. In a
mixed pairing the sides swap from one seed to the next (A is Black on seeds 1
and 3, B on seed 2).

    python -m workshop.scripts.tournament --label sonnet55 \\
        --model us.anthropic.claude-sonnet-5-5 --bedrock-temperature provider-default \\
        --seeds 7101 7202 7303 --parallel 4 [--mission-preset long]

**Cross-model lab** (`--lab`, workshop/labs/cross_model_full_match/). Two
models from a config file, the same personality on both sides, every seed of
a fixed panel, and every condition played in both orientations (model A
Black then model A White). The pair "same seed, same personality, swapped
sides" is the counterbalanced pair.

    python -m workshop.scripts.tournament --lab workshop/labs/cross_model_full_match/config.json --dry-run
    python -m workshop.scripts.tournament --lab workshop/labs/cross_model_full_match/config.json --preflight
    python -m workshop.scripts.tournament --lab workshop/labs/cross_model_full_match/config.json

Each match is `python -m workshop.play ... --headless` in its own process.
- Outputs go to `workshop_runs/<session>/live/<run-id>/`, where the session is `tournament-<label>` or `cross-model-<label>`.
- Each match's stdout goes to `logs/<match>.log`.
- The schedule, with each match's status, goes to `manifest.json`.

Resuming and reruns:
- Running the same command again resumes. A finished match is never run again unless `--rerun <match>` asks for it.
- A failed match records its reason and retry count, and is retried on the next run up to `--max-retries` times.
- `--exclude <match>=<reason>` takes a match out of the analysis without deleting it.
- `--dry-run` prints the plan and makes no model call.

Each match is one sample: see workshop.scripts.tournament_report and
workshop.scripts.cross_model_report for what the results can and cannot show.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from workshop.runners.common import PROJECT_ROOT, display_path, ensure_within_output_root, session_dir

PERSONALITIES = ("saboteur", "racer", "cautious", "opportunist")
from workshop.live.config import MISSION_PRESETS as MISSION_SECONDS  # noqa: E402  quick 150 / standard 300 / long 480
OUTPUTS_RE = re.compile(r"outputs: (\S+)")
PROVIDERS = ("bedrock", "openai")


@dataclass
class Match:
    name: str
    seed_index: int
    seed: int
    black: str  # personality
    white: str  # personality
    status: str = "pending"  # pending / running / done / failed / excluded
    run_dir: str | None = None
    exit_code: int | None = None
    wall_seconds: float | None = None
    # cross-model lab (None in a self-play tournament)
    pair_id: str | None = None
    orientation: str | None = None  # "A": model 1 is Black; "B": model 1 is White
    black_model_key: str | None = None
    white_model_key: str | None = None
    black_provider: str | None = None
    black_model: str | None = None
    white_provider: str | None = None
    white_model: str | None = None
    retry_count: int = 0
    failure_reason: str | None = None
    rerun_count: int = 0
    previous_run_dirs: list = field(default_factory=list)
    excluded_reason: str | None = None


def pairings(personalities: tuple[str, ...]) -> list[tuple[str, str]]:
    return list(itertools.combinations_with_replacement(personalities, 2))


def schedule(personalities: tuple[str, ...], seeds: list[int]) -> list[Match]:
    matches = []
    for index, seed in enumerate(seeds):
        for a, b in pairings(personalities):
            black, white = (a, b) if index % 2 == 0 else (b, a)
            matches.append(Match(f"s{index + 1}-{black}-vs-{white}", index + 1, seed, black, white))
    return matches


def command(args: argparse.Namespace, match: Match) -> list[str]:
    return [sys.executable, "-m", "workshop.play",
            "--provider", "bedrock", "--model", args.model, "--region", args.region, "--runtime", "native",
            "--bedrock-temperature", args.bedrock_temperature,
            "--black-personality", match.black, "--white-personality", match.white,
            "--mission-preset", args.mission_preset, "--seed", str(match.seed),
            "--timing", "live_async", "--encounters", "on", "--lives", "on", "--spawn", "different_rooms",
            "--items", "contested", "--puzzle", args.puzzle,
            "--session", f"tournament-{args.label}", "--headless",
            "--max-decisions", "900", "--max-minutes", str(args.max_minutes)]


# -- cross-model lab -------------------------------------------------------------------

class LabConfigError(ValueError):
    """The lab config or seed file is not usable; nothing is started."""


def load_lab(config_path: str | Path) -> dict[str, Any]:
    """The lab config with its seed panel, checked."""
    path = Path(config_path)
    if not path.is_file():
        raise LabConfigError(f"lab config not found: {path}")
    lab = json.loads(path.read_text())
    seeds_path = path.parent / lab.get("seeds_file", "seeds.json")
    if not seeds_path.is_file():
        raise LabConfigError(f"seed panel not found: {seeds_path}")
    panel = json.loads(seeds_path.read_text())
    seeds = [int(s["seed"]) for s in panel.get("seeds", [])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise LabConfigError("the seed panel must list distinct seeds")
    models = lab.get("models") or []
    if len(models) != 2:
        raise LabConfigError("a cross-model lab needs exactly two models")
    keys = [m.get("key") for m in models]
    if len(set(keys)) != 2 or not all(k and re.fullmatch(r"[a-z0-9]+", k) for k in keys):
        raise LabConfigError("each model needs a distinct lowercase key (letters and digits)")
    for m in models:
        for k in ("display_name", "provider", "provider_model_id", "temperature"):
            if not m.get(k):
                raise LabConfigError(f"model {m.get('key')!r} has no {k}")
        if m["provider"] not in PROVIDERS:
            raise LabConfigError(f"model {m['key']!r}: unknown provider {m['provider']!r}")
        if m["temperature"] not in ("0", "provider-default"):
            raise LabConfigError(f"model {m['key']!r}: temperature must be \"0\" or \"provider-default\"")
    # workshop.play takes one temperature setting per provider
    for provider in PROVIDERS:
        settings = {m["temperature"] for m in models if m["provider"] == provider}
        if len(settings) > 1:
            raise LabConfigError(f"both {provider} models must use the same temperature setting "
                                 f"(workshop.play has one --{provider}-temperature)")
    if lab.get("pairing") != "same_personality":
        raise LabConfigError("only the Phase A pairing 'same_personality' is implemented; Phase B pairings "
                             "need a documented hypothesis each and are added by hand")
    if lab.get("mission_preset", "standard") not in MISSION_SECONDS:
        raise LabConfigError(f"unknown mission preset {lab.get('mission_preset')!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,50}", str(lab.get("label", ""))):
        raise LabConfigError("the lab label must be 1-51 letters, digits, '_' or '-'")
    return {**lab, "_config_path": str(path), "_seeds_path": str(seeds_path), "_panel": panel, "_seeds": seeds}


def schedule_lab(lab: dict[str, Any]) -> list[Match]:
    """seeds x personalities x 2 orientations; the two matches of a counterbalanced pair are adjacent."""
    first, second = lab["models"]
    matches = []
    for index, seed in enumerate(lab["_seeds"]):
        for personality in lab["personalities"]:
            pair_id = f"{seed}-{personality}"
            for orientation, (black, white) in (("A", (first, second)), ("B", (second, first))):
                matches.append(Match(
                    name=f"{seed}-{personality}-{black['key']}-vs-{white['key']}", seed_index=index + 1, seed=seed,
                    black=personality, white=personality, pair_id=pair_id, orientation=orientation,
                    black_model_key=black["key"], white_model_key=white["key"],
                    black_provider=black["provider"], black_model=black["provider_model_id"],
                    white_provider=white["provider"], white_model=white["provider_model_id"]))
    return matches


def _temperature_flags(lab: dict[str, Any]) -> list[str]:
    flags = []
    for provider in PROVIDERS:
        settings = {m["temperature"] for m in lab["models"] if m["provider"] == provider}
        if settings:
            flags += [f"--{provider}-temperature", settings.pop()]
    return flags


def lab_session(lab: dict[str, Any]) -> str:
    return f"cross-model-{lab['label']}"


def _fields(model: dict[str, Any] | None) -> str | None:
    fields = (model or {}).get("request_fields")
    return json.dumps(fields, sort_keys=True, separators=(",", ":")) if fields else None


def lab_command(lab: dict[str, Any], match: Match) -> list[str]:
    preset = lab.get("mission_preset", "standard")
    by_key = {m["key"]: m for m in lab["models"]}
    fields = []
    for side, key in (("black", match.black_model_key), ("white", match.white_model_key)):
        if _fields(by_key.get(key)):
            fields += [f"--{side}-request-fields", _fields(by_key[key])]
    max_minutes = lab.get("max_minutes") or MISSION_SECONDS[preset] / 60 + 3
    # simulated clock (evaluation timing + game_time timer): only set when the
    # lab names them, so an older config keeps workshop.play's own defaults
    clock = []
    for key, flag in (("mission_clock", "--mission-clock"), ("mission_timer", "--mission-timer"),
                      ("commit_policy", "--commit-policy"), ("timeout_seconds", "--timeout")):
        if lab.get(key) is not None:
            clock += [flag, f"{lab[key]:g}" if isinstance(lab[key], (int, float)) else str(lab[key])]
    return [sys.executable, "-m", "workshop.play",
            "--black-provider", match.black_provider, "--black-model", match.black_model,
            "--white-provider", match.white_provider, "--white-model", match.white_model,
            "--region", lab.get("region", "us-east-1"), "--runtime", lab.get("runtime", "native"),
            *_temperature_flags(lab), *fields,
            "--black-personality", match.black, "--white-personality", match.white,
            "--mission-preset", preset, "--seed", str(match.seed),
            "--timing", lab.get("decision_timing", "live_async"), *clock, "--encounters", lab.get("encounters", "on"),
            "--lives", lab.get("lives", "on"), "--spawn", lab.get("spawn", "different_rooms"),
            "--items", lab.get("items", "contested"), "--puzzle", lab.get("puzzle", "riddle"),
            "--session", lab_session(lab), "--headless",
            "--max-decisions", "900", "--max-minutes", f"{max_minutes:g}"]


def simulated_clock(lab: dict[str, Any]) -> bool:
    """Model latency spends no mission time: the world and the shared countdown
    freeze while either spy's model thinks (evaluation timing), and the
    countdown follows game time, not the wall clock."""
    return (lab.get("decision_timing") == "evaluation" and lab.get("mission_clock", "simulation_time") == "simulation_time"
            and lab.get("mission_timer") == "game_time")


def preflight_command(lab: dict[str, Any], model: dict[str, Any]) -> list[str]:
    """One bounded probe request for one model (workshop.play --preflight-only)."""
    return [sys.executable, "-m", "workshop.play", "--provider", model["provider"],
            "--model", model["provider_model_id"], "--region", lab.get("region", "us-east-1"),
            "--runtime", lab.get("runtime", "native"), *_temperature_flags(lab),
            *(["--request-fields", _fields(model)] if _fields(model) else []),
            "--session", lab_session(lab), "--preflight-only"]


def code_fingerprint() -> str:
    """The game has no VCS here: a hash of the code and data a match runs on."""
    digest = hashlib.sha256()
    for root in ("src", "workshop/live", "data"):
        for path in sorted((PROJECT_ROOT / root).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json") and "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
                digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()[:16]


# -- shared runner ------------------------------------------------------------------------

def _load_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {m["name"]: m for m in json.loads(path.read_text()).get("matches", [])}


def _write_manifest(path: Path, args: argparse.Namespace, matches: list[Match]) -> None:
    # matches of this label that are not in the current schedule (other seeds) stay listed
    names = {m.name for m in matches}
    kept = [m for name, m in _load_manifest(path).items() if name not in names]
    body = {"label": args.label, "model": args.model, "region": args.region,
            "bedrock_temperature": args.bedrock_temperature, "mission_preset": args.mission_preset,
            "puzzle": args.puzzle, "personalities": list(args.personalities),
            "seeds": sorted(set(args.seeds) | {m["seed"] for m in kept}),
            "matches": kept + [asdict(m) for m in matches]}
    _atomic_write(path, body)


def _atomic_write(path: Path, body: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n")
    tmp.replace(path)


def _done(match: Match) -> bool:
    return match.status == "done" and match.run_dir is not None \
        and (PROJECT_ROOT / match.run_dir / "summary.json").is_file()


def _failure_reason(log_text: str, exit_code: int | None) -> str:
    for line in log_text.splitlines():
        if "[FAIL]" in line:
            return "preflight: " + line.split("[FAIL]", 1)[1].strip()[:300]
    for line in log_text.splitlines():
        if line.startswith("LIVE ERROR"):
            return line[:300]
    return f"exit code {exit_code}, no summary.json"


def _restore(m: Match, old: dict[str, Any]) -> None:
    for key in ("status", "run_dir", "exit_code", "wall_seconds", "retry_count", "failure_reason", "rerun_count",
                "previous_run_dirs", "excluded_reason"):
        if key in old:
            setattr(m, key, old[key])


def _stop_reason(summary_path: Path) -> str | None:
    try:
        return json.loads(summary_path.read_text()).get("stop_reason")
    except (OSError, json.JSONDecodeError):
        return None


def run_queue(todo: list[Match], *, build_command: Callable[[Match], list[str]], logs: Path,
              save: Callable[[], None], parallel: int, stagger_seconds: float, preflight_retries: int,
              require_match_over: bool = False) -> bool:
    """Runs the matches, `parallel` at a time; False if interrupted. With
    `require_match_over` (the lab) a match that stopped on a cap is failed."""
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    running: dict[str, tuple[subprocess.Popen, Match, float, Any]] = {}
    preflight_tries: dict[str, int] = {}
    queue = list(todo)
    try:
        while queue or running:
            while queue and len(running) < parallel:
                m = queue.pop(0)
                log = open(logs / f"{m.name}.log", "w")  # noqa: SIM115 - closed when the process ends
                proc = subprocess.Popen(build_command(m), cwd=PROJECT_ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
                m.status, m.run_dir, m.exit_code = "running", None, None
                running[m.name] = (proc, m, time.monotonic(), log)
                save()
                print(f"[START] {m.name} (seed {m.seed})", flush=True)
                time.sleep(stagger_seconds)
            for name, (proc, m, started, log) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                text = (logs / f"{m.name}.log").read_text(errors="replace")
                found = OUTPUTS_RE.findall(text)
                m.run_dir = found[-1] if found else None
                m.exit_code = proc.returncode
                m.wall_seconds = round(time.monotonic() - started, 1)
                finished = m.run_dir is not None and (PROJECT_ROOT / m.run_dir / "summary.json").is_file()
                m.status = "done" if proc.returncode == 0 and finished else "failed"
                m.failure_reason = None if m.status == "done" else _failure_reason(text, proc.returncode)
                if m.status == "done" and require_match_over:
                    # a match cut short (wall-time or decision cap) has no comparable result
                    stop = _stop_reason(PROJECT_ROOT / m.run_dir / "summary.json")
                    if stop != "match_over":
                        m.status, m.failure_reason = "failed", f"stopped before the end of the match: {stop}"
                if m.status == "failed":
                    m.retry_count += 1
                del running[name]
                save()
                print(f"[{m.status.upper()}] {m.name} exit={m.exit_code} {m.wall_seconds} s -> {m.run_dir}"
                      + (f" ({m.failure_reason})" if m.failure_reason else ""), flush=True)
                # the match never started (the one probe request failed): try it again later
                if m.status == "failed" and "LIVE PREFLIGHT FAIL" in text \
                        and preflight_tries.get(m.name, 0) < preflight_retries:
                    preflight_tries[m.name] = preflight_tries.get(m.name, 0) + 1
                    queue.append(m)
                    print(f"[RETRY] {m.name}: preflight failed, attempt {preflight_tries[m.name] + 1} queued",
                          flush=True)
            time.sleep(2.0)
    except KeyboardInterrupt:
        for proc, m, _, log in running.values():
            proc.terminate()
            m.status = "failed"
            m.failure_reason = "interrupted"
        for proc, m, _, log in running.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # a headless match does not always stop on SIGTERM
                proc.kill()
                proc.wait()
            log.close()
        save()
        return False
    return True


# -- entry points -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m workshop.scripts.tournament", description=__doc__.splitlines()[0])
    p.add_argument("--lab", help="cross-model lab config (workshop/labs/cross_model_full_match/config.json)")
    p.add_argument("--label", help="self-play: short name for this model's tournament, e.g. sonnet55")
    p.add_argument("--model", help="self-play: Bedrock model id / inference profile, both spies")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--bedrock-temperature", default="0", choices=("0", "provider-default"))
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--personalities", nargs="+", default=list(PERSONALITIES))
    p.add_argument("--mission-preset", default="standard", choices=("quick", "standard", "long"),
                   help="mission length: quick 150 s, standard 300 s (default), long 480 s")
    p.add_argument("--puzzle", default="riddle", choices=("riddle", "split", "none"))
    p.add_argument("--parallel", type=int, default=None, help="matches at once (default 4, or the lab's)")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="wall-clock cap per match (default: the mission length plus 3 minutes)")
    p.add_argument("--stagger-seconds", type=float, default=5.0, help="pause between two launches")
    p.add_argument("--preflight-retries", type=int, default=2,
                   help="re-queue a match whose preflight probe failed (it never started) this many times")
    p.add_argument("--max-retries", type=int, default=2, help="lab: a failed match is retried on later runs at most this often")
    p.add_argument("--rerun", action="append", default=[], metavar="MATCH", help="lab: run a finished match again")
    p.add_argument("--only", action="append", default=[], metavar="PAIR_OR_MATCH",
                   help="lab: run only this counterbalanced pair (e.g. 7303-saboteur, both orientations) or match; "
                        "repeatable - for a pilot")
    p.add_argument("--exclude", action="append", default=[], metavar="MATCH=REASON",
                   help="lab: take a match out of the analysis (kept on disk, reason recorded)")
    p.add_argument("--preflight", action="store_true", help="lab: one bounded probe per model, then stop")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.lab:
        return main_lab(args)
    if not (args.label and args.model and args.seeds):
        p.error("a self-play tournament needs --label, --model and --seeds (or use --lab)")
    args.parallel = args.parallel or 4
    args.personalities = tuple(args.personalities)
    if args.max_minutes is None:
        args.max_minutes = MISSION_SECONDS[args.mission_preset] / 60 + 3

    root = ensure_within_output_root(session_dir(f"tournament-{args.label}"))
    logs = root / "logs"
    manifest_path = root / "manifest.json"
    matches = schedule(args.personalities, args.seeds)
    if manifest_path.is_file():  # resuming must not mix two different setups under one label
        body = json.loads(manifest_path.read_text())
        clash = [k for k in ("model", "mission_preset", "puzzle", "bedrock_temperature")
                 if body.get(k) is not None and body.get(k) != getattr(args, k)]
        if clash:
            print(f"[TOURNAMENT] label {args.label!r} was run with a different "
                  + ", ".join(f"{k} ({body[k]})" for k in clash) + " - use another --label", file=sys.stderr)
            return 2
    previous = _load_manifest(manifest_path)
    for m in matches:  # resume: keep what an earlier run finished
        old = previous.get(m.name)
        if old and old.get("seed") == m.seed:
            _restore(m, old)
    todo = [m for m in matches if not _done(m)]
    print(f"[TOURNAMENT] {args.label}: {len(matches)} matches, {len(matches) - len(todo)} already done, "
          f"{len(todo)} to run, {args.parallel} at a time")
    if args.dry_run:
        for m in matches:
            print(f"  {m.name:42} seed={m.seed} {'done' if _done(m) else 'to run'}")
        return 0
    finished = run_queue(todo, build_command=lambda m: command(args, m), logs=logs,
                         save=lambda: _write_manifest(manifest_path, args, matches), parallel=args.parallel,
                         stagger_seconds=args.stagger_seconds, preflight_retries=args.preflight_retries)
    if not finished:
        print("[TOURNAMENT] interrupted - run the same command again to resume")
        return 130
    failed = [m.name for m in matches if m.status == "failed"]
    print(f"[TOURNAMENT] finished: {sum(_done(m) for m in matches)}/{len(matches)} done"
          + (f", failed: {', '.join(failed)}" if failed else "") + f" - manifest {display_path(manifest_path)}")
    return 0 if not failed else 1


def _lab_manifest(lab: dict[str, Any], matches: list[Match], fingerprint: str) -> dict[str, Any]:
    return {
        "kind": "cross_model_full_match",
        "experiment": lab.get("experiment"), "experiment_version": lab.get("experiment_version"),
        "phase": lab.get("phase"), "label": lab["label"], "pairing": lab["pairing"],
        "config_file": display_path(Path(lab["_config_path"])), "seeds_file": display_path(Path(lab["_seeds_path"])),
        "config": {k: v for k, v in lab.items() if not k.startswith("_")},
        "seed_panel": lab["_panel"],
        "game_version": fingerprint,
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matches": [asdict(m) for m in matches],
    }


def main_lab(args: argparse.Namespace) -> int:
    try:
        lab = load_lab(args.lab)
    except (LabConfigError, json.JSONDecodeError) as exc:
        print(f"[LAB] {exc}", file=sys.stderr)
        return 2
    parallel = args.parallel or int(lab.get("parallel", 4))
    root = ensure_within_output_root(session_dir(lab_session(lab)))
    logs = root / "logs"
    manifest_path = root / "manifest.json"
    fingerprint = code_fingerprint()
    matches = schedule_lab(lab)
    names = {m.name for m in matches}
    if manifest_path.is_file():
        body = json.loads(manifest_path.read_text())
        old_models = [(m["provider"], m["provider_model_id"]) for m in body.get("config", {}).get("models", [])]
        new_models = [(m["provider"], m["provider_model_id"]) for m in lab["models"]]
        if old_models and old_models != new_models:
            print(f"[LAB] label {lab['label']!r} was run with other models {old_models} - use another label",
                  file=sys.stderr)
            return 2
        for key in ("mission_preset", "puzzle", "pairing", "experiment_version", "decision_timing", "mission_clock",
                    "mission_timer", "commit_policy"):
            if body.get("config", {}).get(key) not in (None, lab.get(key)):
                print(f"[LAB] label {lab['label']!r} was run with another {key} - use another label", file=sys.stderr)
                return 2
        previous = _load_manifest(manifest_path)
        for m in matches:
            if m.name in previous:
                _restore(m, previous[m.name])
    for name in args.rerun:
        if name not in names:
            print(f"[LAB] --rerun: no match named {name!r}", file=sys.stderr)
            return 2
    pair_ids = {m.pair_id for m in matches}
    for name in args.only:
        if name not in names and name not in pair_ids:
            print(f"[LAB] --only: no pair or match named {name!r} (pairs look like 7303-saboteur)", file=sys.stderr)
            return 2
    exclusions = {}
    for item in args.exclude:
        name, _, reason = item.partition("=")
        if name not in names or not reason.strip():
            print(f"[LAB] --exclude needs MATCH=REASON with a known match, got {item!r}", file=sys.stderr)
            return 2
        exclusions[name] = reason.strip()
    for m in matches:
        if m.name in exclusions:
            m.status, m.excluded_reason = "excluded", exclusions[m.name]
        elif m.name in args.rerun and m.status == "done":  # an explicit request, never silent
            m.previous_run_dirs = [*m.previous_run_dirs, m.run_dir]
            m.rerun_count += 1
            m.status, m.run_dir = "pending", None
    todo = [m for m in matches if m.status not in ("done", "excluded") and not _done(m)
            and m.retry_count <= args.max_retries]
    if args.only:  # a pilot: a subset of pairs / matches, the rest stays pending
        todo = [m for m in todo if m.pair_id in args.only or m.name in args.only]
    blocked = [m for m in matches if m.status == "failed" and m.retry_count > args.max_retries]
    _print_lab_plan(lab, matches, todo, blocked, parallel, root)
    if args.only:
        print(f"  --only {', '.join(args.only)}: {len(todo)} match(es) to run now")
    save = lambda: _atomic_write(manifest_path, _lab_manifest(lab, matches, fingerprint))  # noqa: E731
    if args.dry_run:
        return 0
    root.mkdir(parents=True, exist_ok=True)
    save()
    if args.preflight:
        return _lab_preflight(lab, logs)
    finished = run_queue(todo, build_command=lambda m: lab_command(lab, m), logs=logs, save=save,
                         parallel=parallel, stagger_seconds=args.stagger_seconds,
                         preflight_retries=args.preflight_retries, require_match_over=True)
    if not finished:
        print("[LAB] interrupted - run the same command again to resume")
        return 130
    failed = [m.name for m in matches if m.status == "failed"]
    print(f"[LAB] finished: {sum(_done(m) for m in matches)}/{len(matches)} done"
          + (f", failed: {', '.join(failed)}" if failed else "") + f" - manifest {display_path(manifest_path)}")
    return 0 if not failed else 1


def _print_lab_plan(lab, matches, todo, blocked, parallel, root) -> None:
    first, second = lab["models"]
    seeds = lab["_seeds"]
    print(f"[LAB] {lab.get('experiment')} {lab.get('experiment_version')} phase {lab.get('phase')} - "
          f"label {lab['label']}")
    for m in lab["models"]:
        print(f"  model {m['key']:8} {m['display_name']} - provider {m['provider']}, "
              f"model id {m['provider_model_id']}, temperature {m['temperature']}"
              + (f", request fields {_fields(m)}" if _fields(m) else ""))
    print(f"  personalities: {', '.join(lab['personalities'])} (same personality on both sides)")
    print(f"  seeds ({len(seeds)}): " + ", ".join(
        f"{s['seed']} [{s.get('historical_side_advantage', 'unknown')}]" for s in lab["_panel"]["seeds"]))
    print(f"  orientations per condition: A = {first['key']} Black / {second['key']} White, "
          f"B = {second['key']} Black / {first['key']} White")
    print(f"  planned: {len(seeds)} seeds x {len(lab['personalities'])} personalities x 2 orientations = "
          f"{len(matches)} matches ({len({m.pair_id for m in matches})} counterbalanced pairs), "
          f"{2 * len(matches)} model instances (2 spies per match)")
    print(f"  mission {lab.get('mission_preset', 'standard')} ({MISSION_SECONDS[lab.get('mission_preset', 'standard')]:g} s), "
          f"puzzle {lab.get('puzzle', 'riddle')}, runtime {lab.get('runtime', 'native')}, {parallel} at a time")
    if simulated_clock(lab):
        print(f"  clock: simulated - timing evaluation, mission timer game_time, commit policy "
              f"{lab.get('commit_policy', 'barrier_answer_order')}: model latency spends no mission time "
              f"(latency is still recorded); wall-time cap {lab.get('max_minutes', 'default')} min per match")
    else:
        print(f"  clock: timing {lab.get('decision_timing', 'live_async')}, mission timer "
              f"{lab.get('mission_timer', 'wall_clock')} - model latency spends mission time")
    counts = {s: sum(1 for m in matches if m.status == s) for s in ("done", "failed", "excluded", "pending")}
    print(f"  status: {counts['done']} done, {counts['failed']} failed, {counts['excluded']} excluded, "
          f"{len(todo)} to run" + (f", {len(blocked)} over the retry limit" if blocked else ""))
    print(f"  output: {display_path(root)}")


def _lab_preflight(lab: dict[str, Any], logs: Path) -> int:
    """One probe request per configured model; no fallback, the real reason on failure."""
    logs.mkdir(parents=True, exist_ok=True)
    ok = True
    for model in lab["models"]:
        result = subprocess.run(preflight_command(lab, model), cwd=PROJECT_ROOT, capture_output=True, text=True,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=180)
        text = result.stdout + result.stderr
        (logs / f"preflight-{model['key']}.log").write_text(text)
        line = next((ln.strip() for ln in text.splitlines() if "model " + model["provider"] in ln), text.strip()[-300:])
        passed = result.returncode == 0 and "LIVE PREFLIGHT PASS" in text
        ok &= passed
        print(f"[PREFLIGHT] {model['key']} ({model['provider_model_id']}): {'PASS' if passed else 'FAIL'} - {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
