"""Stage 3 milestone 3: closed-loop B1 vs B2 surrogate matches on the Phase-A matrix. No provider call.
    python -m workshop.scripts.closed_loop_policy_eval --dry-run
    python -m workshop.scripts.closed_loop_policy_eval --pilot --headless
    python -m workshop.scripts.closed_loop_policy_eval --headless --resume
    python -m workshop.scripts.closed_loop_policy_eval --only 7101-opportunist-A --headless

Both spies are SurrogateControllers of the identities that played the Phase-A
match (same seed, personality, side assignment and game settings, read from the
frozen Phase-A run.json files). A match uses one action emulator on both sides
(B1 table-v1 or B2 logistic-v1) and the frozen target policy; the paired B1 and
B2 matches of a configuration share the environment seed and the policy RNG seed.

Execution: one process per match, headless, a fixed 1/60 s simulated step per
frame with no sleep (the frame clock is the only wall-clock input of the game
loop; with mission_timer=game_time and two surrogates the world advances by
exactly the step, as in evaluation mode, where the world is frozen while a model
thinks). Rendering is skipped except for the two screenshots. A network guard
refuses every connection inside a match process.

Outputs: workshop_runs/<session>/manifest.json and
workshop_runs/<session>/matches/r<replicate>/<emulator>/<configuration>/
(run.json, summary.json, match.json, telemetry_match01.jsonl, screenshots).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workshop.policy_emulator.closed_loop import DATASET, PHASE_A_DIR, phase_a_configurations
from workshop.runners.common import PROJECT_ROOT, WorkshopError, display_path, ensure_within_output_root, session_dir

EXPERIMENT_VERSION = "stage3-m3-closed-loop-v1"
DEFAULT_SESSION = "stage3-closed-loop-phase-a-v2"
DEFAULT_EMULATORS = ("table-v1", "logistic-v1")
EMULATOR_KEYS = {"table-v1": "table", "logistic-v1": "logistic", "tree-v1": "tree", "frequency-v1": "frequency"}
DEFAULT_TARGET = "target-v2.1"
PILOT_PERSONALITIES = ("opportunist", "saboteur")
MAX_DECISIONS = 100000          # per spy; a 300 s mission allows about 600
MAX_WALL_SECONDS = 1200.0       # safety stop per match (a stopped match is FAILED, not scored)
HASH_SEED = "0"                 # PYTHONHASHSEED of every match process


# -- the plan -------------------------------------------------------------------------------------

def policy_rng_seed(environment_seed: int, replicate: int) -> int:
    """Explicit and separate from the environment seed; the same for B1 and B2 of a configuration."""
    return environment_seed * 100 + replicate


def build_plan(configs: list[dict], emulators: tuple[str, ...], replicates: int, *, only: list[str] | None = None,
               pilot: bool = False) -> list[dict[str, Any]]:
    if pilot:
        first_seed = min(configs, key=lambda c: c["seed_index"])["seed"]
        configs = [c for c in configs if c["seed"] == first_seed and c["personality"] in PILOT_PERSONALITIES]
    if only:
        configs = [c for c in configs if any(fnmatch.fnmatch(c["config_id"], pat) or c["config_id"] == pat
                                             or c["phase_a_match"] == pat for pat in only)]
    plan = []
    for replicate in range(1, replicates + 1):
        for c in configs:
            for emulator in emulators:
                plan.append({**c, "replicate": replicate, "emulator": emulator,
                             "policy_rng_seed": policy_rng_seed(c["seed"], replicate),
                             "match_id": f"r{replicate}-{emulator}-{c['config_id']}",
                             "rel_dir": f"matches/r{replicate}/{emulator}/{c['config_id']}"})
    return plan


def match_config(entry: dict, target: str) -> dict[str, Any]:
    """The resolve_config values of one match: Phase-A conditions, two surrogates, one emulator."""
    return {**entry["conditions"], "black_controller": "surrogate", "white_controller": "surrogate",
            "black_policy": entry["black_policy"], "white_policy": entry["white_policy"],
            "policy_dataset": DATASET, "policy_emulator": EMULATOR_KEYS[entry["emulator"]], "policy_target": target,
            "surrogate_seed": entry["policy_rng_seed"]}


# -- one match (runs in its own process) ----------------------------------------------------------

class FixedStepClock:
    """Drop-in for pygame.time.Clock: every tick is exactly one 1/FPS step and never sleeps."""

    def __init__(self, fps: int) -> None:
        self.fps = fps
        self.ms = 1000.0 / fps
        self.ticks = 0

    def tick(self, framerate: int = 0) -> float:
        self.ticks += 1
        return self.ms

    def get_fps(self) -> float:
        return float(self.fps)


def _fast_runner_class():
    from src import constants
    from workshop.live.session import MatchRunner

    class FastMatchRunner(MatchRunner):
        """MatchRunner with a fixed simulated step, no frame sleep and no per-frame rendering."""

        def new_match(self) -> None:
            super().new_match()
            game = self.game
            if game.state.external_mission_clock:
                raise RuntimeError("fixed-step execution needs mission_timer=game_time (no wall-clock mission timer)")
            game.clock = FixedStepClock(constants.FPS)
            renderer, view = game.renderer, self.view
            self._real_render = type(renderer).render
            renderer.render = lambda *args, **kwargs: view.update(game)  # keeps the decision counts current
            self.monitor.check_overlay = lambda *args, **kwargs: None    # LLM overlay check, scans every event

        def screenshot(self, name: str) -> None:
            self._real_render(self.game.renderer, self.game.state, 0.0, None, None, self.game.bot_controller)
            super().screenshot(name)

    return FastMatchRunner


def _refuse_client(*args, **kwargs):
    raise RuntimeError("closed-loop matches never build a provider client")


def run_one(spec_path: Path) -> int:
    """Play one planned match into its directory; write run.json, summary.json and match.json."""
    from workshop.live.config import resolve_config
    from workshop.live.session import load_surrogate_policy, run_metadata, write_json, _code_fingerprint
    from workshop.runners.common import GUARD, install_guard

    spec = json.loads(spec_path.read_text())
    run_dir = ensure_within_output_root(Path(spec["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    install_guard()
    GUARD.activate([run_dir])
    started = time.monotonic()
    config = resolve_config(spec["config"])
    policies = {}
    for actor_id, actor in config.actors.items():
        lp = load_surrogate_policy(config, actor)
        policies[actor_id] = {"policy": actor.policy, "emulator": lp.emulator, "target": lp.target_version,
                              "emulator_artifact_hash": lp.metadata.get("content_hash"),
                              "target_artifact_hash": (lp.target_metadata or {}).get("content_hash"),
                              "coverage_artifact_hash": lp.coverage_hash}
    entry = spec["entry"]
    meta = run_metadata(config, session=spec["session"], run_id=entry["match_id"], headless=True,
                        preflight={"ok": True, "skipped": "no LLM spy: no provider is called", "surrogates": policies})
    meta.update({"kind": "closed_loop_surrogate_match", "command": "python -m workshop.scripts.closed_loop_policy_eval",
                 "experiment_version": EXPERIMENT_VERSION, "entry": entry, "execution": spec["execution"]})
    write_json(run_dir / "run.json", meta)
    runner = _fast_runner_class()(config, run_dir, headless=True, max_decisions=MAX_DECISIONS,
                                  max_wall_seconds=MAX_WALL_SECONDS, client_factory=_refuse_client)
    summary = runner.run()
    write_json(run_dir / "summary.json", summary)
    GUARD.deactivate()
    actors = summary.get("actors") or {}
    record = {
        "match_id": entry["match_id"], "experiment_version": EXPERIMENT_VERSION, "stop_reason": summary.get("stop_reason"),
        "wall_seconds": round(time.monotonic() - started, 2), "frames": runner.game.clock.ticks,
        "game_time_elapsed": summary.get("game_time_elapsed"),
        "simulated_seconds": round(runner.game.clock.ticks / runner.game.clock.fps, 3),
        "environment_seed": config.seed, "policy_rng_seed": entry["policy_rng_seed"],
        "emulator": entry["emulator"], "target_policy": spec["config"]["policy_target"],
        "game_code_hash": _code_fingerprint(), "phase_a_game_version": entry.get("phase_a_game_version"),
        "provider_calls": sum(int(a.get("provider_calls") or 0) for a in actors.values()),
        "network_attempts": list(GUARD.network_attempts),
        "unexpected_writes": [w for w in GUARD.unexpected_writes if w != f"open {os.devnull}"],
        "requests_after_terminal": {a: (v.get("terminal") or {}).get("requests_after_terminal") for a, v in actors.items()},
        "decisions": {a: v.get("decisions") for a, v in actors.items()},
        "lifecycle": {a: v.get("lifecycle") for a, v in actors.items()},
        "provenance": {a: (v.get("surrogate") or {}).get("provenance") for a, v in actors.items()},
        "artifacts": policies,
    }
    write_json(run_dir / "match.json", record)
    return 0 if summary.get("stop_reason") == "match_over" else 3


# -- the orchestrator ---------------------------------------------------------------------------

def _manifest_path(session: str) -> Path:
    return session_dir(session) / "manifest.json"


def load_manifest(session: str) -> dict[str, Any]:
    path = _manifest_path(session)
    return json.loads(path.read_text()) if path.is_file() else {"matches": {}}


def save_manifest(session: str, manifest: dict[str, Any]) -> None:
    path = ensure_within_output_root(_manifest_path(session))
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


MATCH_FILES = ("match.json", "summary.json", "run.json", "telemetry_match01.jsonl")


def check_match(session: str, entry: dict, manifest: dict) -> tuple[str, str | None]:
    """'valid', 'missing', 'corrupt' (rerun with --resume) or 'provider' (a hard failure), with the reason."""
    rec = manifest.get("matches", {}).get(entry["match_id"])
    run_dir = session_dir(session) / entry["rel_dir"]
    if rec is None:
        return ("corrupt", "a match directory without a manifest entry") if run_dir.exists() else ("missing", None)
    if rec.get("status") != "done":
        return "corrupt", f"manifest status {rec.get('status')!r}"
    for name in MATCH_FILES:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            return "corrupt", f"{name} missing or empty"
    try:
        match = json.loads((run_dir / "match.json").read_text())
    except ValueError:
        return "corrupt", "match.json is not valid JSON"
    if (match.get("provider_calls") or 0) != 0 or match.get("network_attempts"):
        return "provider", f"{match.get('provider_calls')} provider calls, {len(match.get('network_attempts') or [])} network attempts"
    expected = {"match_id": entry["match_id"], "stop_reason": "match_over", "emulator": entry["emulator"],
                "environment_seed": entry["seed"], "policy_rng_seed": entry["policy_rng_seed"]}
    wrong = [k for k, v in expected.items() if match.get(k) != v]
    if wrong:
        return "corrupt", "match.json differs from the plan: " + ", ".join(f"{k}={match.get(k)!r}" for k in wrong)
    return "valid", None


def is_done(session: str, entry: dict, manifest: dict) -> bool:
    return check_match(session, entry, manifest)[0] == "valid"


def plan_status(plan: list[dict], session: str) -> dict[str, Any]:
    """Read-only: which planned matches are valid, missing or need a rerun."""
    manifest = load_manifest(session)
    out: dict[str, Any] = {"valid": [], "missing": [], "corrupt": [], "provider": []}
    for entry in plan:
        state, reason = check_match(session, entry, manifest)
        out[state].append((entry["match_id"], reason))
    return out


def _launch(session: str, entry: dict, target: str, execution: dict) -> tuple[subprocess.Popen, Path, Any]:
    run_dir = session_dir(session) / entry["rel_dir"]
    if run_dir.exists():  # a failed or interrupted earlier attempt: start this match from scratch
        from workshop.runners.common import remove_within_output_root
        remove_within_output_root(run_dir)
    run_dir.mkdir(parents=True)
    spec_path = run_dir / "match_spec.json"
    spec_path.write_text(json.dumps({"session": session, "run_dir": str(run_dir), "entry": entry,
                                     "config": match_config(entry, target), "execution": execution}, indent=1) + "\n")
    env = {**os.environ, "PYTHONHASHSEED": HASH_SEED, "PYTHONDONTWRITEBYTECODE": "1", "SDL_VIDEODRIVER": "dummy",
           "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    log = open(run_dir / "process.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "workshop.scripts.closed_loop_policy_eval", "--run-one", str(spec_path)],
                            cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, run_dir, log


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_plan(plan: list[dict], *, session: str, target: str, parallel: int, resume: bool) -> dict[str, Any]:
    manifest = load_manifest(session)
    manifest.update({"kind": "stage3_closed_loop", "experiment_version": EXPERIMENT_VERSION, "dataset": DATASET,
                     "target_policy": target, "phase_a_dir": PHASE_A_DIR})
    manifest.setdefault("matches", {})
    pending, skipped, marked = [], 0, []
    for entry in plan:
        state, reason = check_match(session, entry, manifest)
        if state == "provider":
            raise WorkshopError(f"{entry['match_id']}: {reason} - a provider call or network attempt is a hard failure")
        if state == "valid":
            if not resume:
                raise WorkshopError(f"{entry['match_id']} already ran in session {session}; pass --resume to skip "
                                    "finished matches")
            skipped += 1
            continue
        if state == "corrupt":
            if not resume:
                raise WorkshopError(f"{entry['match_id']} has a partial or corrupt result ({reason}); pass --resume "
                                    "to rerun exactly that match")
            rec = manifest["matches"].setdefault(entry["match_id"], {"rel_dir": entry["rel_dir"]})
            rec.setdefault("previous_attempts", []).append({"status": rec.get("status"), "reason": reason,
                                                            "marked_utc": _now()})
            rec["status"] = "corrupt"
            marked.append((entry["match_id"], reason))
        pending.append(entry)
    execution = {"step": "fixed 1/60 s simulated step, no frame sleep", "rendering": "screenshots only",
                 "python_hash_seed": HASH_SEED, "max_decisions_per_spy": MAX_DECISIONS,
                 "max_wall_seconds": MAX_WALL_SECONDS, "network": "refused by the workshop guard"}
    manifest["execution"] = execution
    save_manifest(session, manifest)
    print(f"[CLOSED-LOOP] {len(plan)} planned, {skipped} valid and skipped, {len(pending)} to run "
          f"({len(marked)} of them partial / corrupt, marked in the manifest; {parallel} in parallel) -> "
          f"{display_path(session_dir(session))}", flush=True)
    for match_id, reason in marked:
        print(f"  [RERUN] {match_id}: {reason}", flush=True)
    running: list[tuple[dict, subprocess.Popen, Path, Any, float]] = []
    started_all, started_utc = time.monotonic(), _now()
    queue = list(pending)
    launched = 0
    try:
        while queue or running:
            while queue and len(running) < parallel:
                entry = queue.pop(0)
                proc, run_dir, log = _launch(session, entry, target, execution)
                launched += 1
                previous = (manifest["matches"].get(entry["match_id"]) or {}).get("previous_attempts")
                manifest["matches"][entry["match_id"]] = {"status": "running", "rel_dir": entry["rel_dir"],
                                                          "config_id": entry["config_id"], "emulator": entry["emulator"],
                                                          "replicate": entry["replicate"],
                                                          "policy_rng_seed": entry["policy_rng_seed"],
                                                          **({"previous_attempts": previous} if previous else {})}
                save_manifest(session, manifest)
                running.append((entry, proc, run_dir, log, time.monotonic()))
            time.sleep(0.5)
            still = []
            for entry, proc, run_dir, log, t0 in running:
                code = proc.poll()
                if code is None:
                    still.append((entry, proc, run_dir, log, t0))
                    continue
                log.close()
                rec = manifest["matches"][entry["match_id"]]
                match_file = run_dir / "match.json"
                result = json.loads(match_file.read_text()) if match_file.is_file() else {}
                ok = code == 0 and result.get("stop_reason") == "match_over"
                provider = (result.get("provider_calls") or 0) != 0 or bool(result.get("network_attempts"))
                rec.update({"status": "provider_or_network" if provider else "done" if ok else "failed", "exit_code": code,
                            "wall_seconds": round(time.monotonic() - t0, 1), "stop_reason": result.get("stop_reason"),
                            "provider_calls": result.get("provider_calls"),
                            "network_attempts": len(result.get("network_attempts") or []),
                            "lifecycle": result.get("lifecycle"), "decisions": result.get("decisions")})
                save_manifest(session, manifest)
                if provider:
                    raise WorkshopError(f"{entry['match_id']}: {result.get('provider_calls')} provider calls, "
                                        f"{len(result.get('network_attempts') or [])} network attempts - hard failure, "
                                        "the run is stopped")
                print(f"  [{'DONE' if ok else 'FAIL'}] {entry['match_id']} exit {code}, {rec['wall_seconds']} s wall, "
                      f"stop {result.get('stop_reason')}, decisions {result.get('decisions')}, "
                      f"lifecycle {result.get('lifecycle')}", flush=True)
            running = still
    finally:
        for _, proc, _, log, _ in running:  # only after a hard failure or an interrupt
            if proc.poll() is None:
                proc.terminate()
            log.close()
        manifest.setdefault("runs", []).append({
            "started_utc": started_utc, "finished_utc": _now(), "wall_seconds": round(time.monotonic() - started_all, 1),
            "planned": len(plan), "skipped_valid": skipped, "launched": launched,
            "rerun_corrupt": len(marked), "replicates": sorted({e["replicate"] for e in pending})})
        save_manifest(session, manifest)
    done = sum(1 for e in plan if is_done(session, e, manifest))
    print(f"[CLOSED-LOOP] {done}/{len(plan)} done in {time.monotonic() - started_all:.0f} s wall", flush=True)
    return manifest


def describe_plan(plan: list[dict], *, session: str, target: str, emulators: tuple[str, ...], replicates: int) -> dict:
    configs = sorted({e["config_id"] for e in plan})
    return {"configurations": len(configs), "emulators": list(emulators), "replicates": replicates, "matches": len(plan),
            "provider_calls": 0, "target_policy": target, "output": display_path(session_dir(session)),
            "seeds": sorted({e["seed"] for e in plan}), "personalities": sorted({e["personality"] for e in plan}),
            "orientations": sorted({e["orientation"] for e in plan})}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m workshop.scripts.closed_loop_policy_eval", description=__doc__.splitlines()[0])
    p.add_argument("--emulators", default=",".join(DEFAULT_EMULATORS), help="comma-separated (default table-v1,logistic-v1)")
    p.add_argument("--target-policy", default=DEFAULT_TARGET, help=f"frozen target policy (default {DEFAULT_TARGET})")
    p.add_argument("--replicates", type=int, default=1, help="policy RNG replicates (default 1; at most 3 are planned)")
    p.add_argument("--dry-run", action="store_true", help="print the plan; play nothing, write nothing")
    p.add_argument("--resume", action="store_true", help="skip matches that already finished in this session")
    p.add_argument("--only", action="append", metavar="CONFIGURATION", help="configuration id or glob, e.g. 7101-opportunist-A")
    p.add_argument("--pilot", action="store_true", help="the 8-match pilot: opportunist + saboteur, first seed, both sides")
    p.add_argument("--headless", action="store_true", help="headless (the only mode: fixed-step matches have no window)")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--session", default=DEFAULT_SESSION)
    p.add_argument("--run-one", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # progress lines reach a redirected log as they happen
    if args.run_one:
        return run_one(args.run_one)
    from workshop.policy_emulator.artifacts import ACTION_EMULATORS, TARGET_VERSIONS
    emulators = tuple(e.strip() for e in args.emulators.split(",") if e.strip())
    bad = [e for e in emulators if e not in ACTION_EMULATORS]
    if bad or not emulators:
        p.error(f"unknown emulator(s) {bad}; available: {', '.join(ACTION_EMULATORS)}")
    if args.target_policy not in TARGET_VERSIONS:
        p.error(f"unknown target policy {args.target_policy}; available: {', '.join(TARGET_VERSIONS)}")
    if not 1 <= args.replicates <= 3:
        p.error("--replicates must be 1, 2 or 3")
    plan = build_plan(phase_a_configurations(), emulators, args.replicates, only=args.only, pilot=args.pilot)
    if not plan:
        p.error("the selection matches no configuration")
    info = describe_plan(plan, session=args.session, target=args.target_policy, emulators=emulators, replicates=args.replicates)
    print(f"[CLOSED-LOOP] {info['configurations']} configurations x {len(emulators)} emulators x {args.replicates} "
          f"replicate(s) = {info['matches']} matches; provider calls: 0; target policy {args.target_policy}; "
          f"output {info['output']}")
    print(f"  seeds {info['seeds']}, personalities {info['personalities']}, orientations {info['orientations']}")
    if args.dry_run:
        status = plan_status(plan, args.session)
        by_rep = {r: sum(1 for e in plan if e["replicate"] == r) for r in sorted({e["replicate"] for e in plan})}
        missing_ids = {m for m, _ in status["missing"]}
        print(f"  in session {args.session}: {len(status['valid'])} valid (skipped by --resume), "
              f"{len(status['missing'])} missing, {len(status['corrupt'])} partial / corrupt (rerun by --resume), "
              f"{len(status['provider'])} with provider or network activity (hard failure)")
        for r, n in by_rep.items():
            miss = sum(1 for e in plan if e["replicate"] == r and e["match_id"] in missing_ids)
            print(f"  replicate {r}: {n} planned, {miss} missing")
        for match_id, reason in status["corrupt"] + status["provider"]:
            print(f"  [RERUN] {match_id}: {reason}")
        examples = [e for e in plan if e["match_id"] in missing_ids][:4] or plan[:4]
        for e in examples:
            print(f"  {e['match_id']}: Black {e['black_policy']} vs White {e['white_policy']}, environment seed "
                  f"{e['seed']}, policy RNG seed {e['policy_rng_seed']} (Phase A {e['phase_a_match']})")
        if len(missing_ids) > len(examples):
            print(f"  ... {len(missing_ids) - len(examples)} more missing")
        print("DRY RUN - nothing was played or written")
        return 0
    if not args.headless:
        p.error("closed-loop matches run headless only: pass --headless")
    run_plan(plan, session=args.session, target=args.target_policy, parallel=max(1, args.parallel), resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
