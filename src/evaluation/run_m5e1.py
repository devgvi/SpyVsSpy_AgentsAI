"""M5E.1: automated live-diagnostic entry point.

    python -m src.evaluation.run_m5e1

Runs, in order, everything M5E.1's spec asks for that needs a real
Ollama/Gemma4 call:

1. an Ollama health snapshot (spec section 20, before);
2. the synthetic S1/S2/S3 sequential-call harness (spec sections 14-19)
   for both native_tool_like and strands, 15 decisions each;
3. the live passive-vs-active gameplay matrix (spec section 41):
   native_tool_like and strands, each under a passive and an active
   (scripted_active_v1) opponent, on 2 shared seeds;
4. a second Ollama health snapshot (after).

Headless (SDL_VIDEODRIVER=dummy), matching every other evaluation entry
point. Writes everything under evaluation_runs/m5e1-long-horizon-stability-v1/
(experiment.json/episodes.jsonl/decisions.jsonl/synthetic_runs.jsonl/
ollama_health.json/summary.json/summary.md) and never touches any prior
M5D/M5E output directory (spec section 23).
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import json
import sys
from datetime import datetime, timezone

from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.ollama_health import snapshot_ollama_health
from src.evaluation.reporting import render_terminal
from src.evaluation.spec import ExperimentSpec, VariantSpec
from src.evaluation.synthetic_stability import PATTERNS, run_synthetic_pattern
from src.llm.telemetry import TelemetryLogger

EXPERIMENT_ID = "m5e1-long-horizon-stability-v1"
SEEDS = [401, 402]
MODEL = "gemma4:latest"
PROVIDER = "ollama"
BASE_URL = "http://localhost:11434"
LLM_TIMEOUT = 60.0
N_DECISIONS_TARGET = 15
MAX_RUNTIME_FAILURES = 20  # deliberately above N_DECISIONS_TARGET (spec section 4/42): a
# runtime-failure-budget abort must never cut a run short before it has had
# the chance to reach the historical decision-6-8 failure horizon and run
# on to the full 15-decision target.


def _build_gameplay_spec(output_root: str) -> ExperimentSpec:
    variants = []
    for runtime_id, runtime_name in (("A_native_tool_like", "native_tool_like"), ("B_strands", "strands")):
        for condition_id, opponent_type in (("passive", "passive"), ("active", "scripted_active")):
            variants.append(
                VariantSpec(
                    variant_id=f"{runtime_id}_{condition_id}",
                    agent_runtime=runtime_name,
                    provider=PROVIDER,
                    model=MODEL,
                    llm_timeout=LLM_TIMEOUT,
                    opponent_type=opponent_type,
                    opponent_profile="OPPORTUNIST",
                )
            )
    return ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        variants=variants,
        seeds=SEEDS,
        execution_mode="evaluation",
        max_episode_game_time=180.0,
        max_episode_wall_time=1800.0,
        max_decisions=N_DECISIONS_TARGET,
        max_runtime_failures=MAX_RUNTIME_FAILURES,
    )


def main(argv: list[str] | None = None) -> int:
    output_root = "evaluation_runs"
    output_dir = os.path.join(output_root, EXPERIMENT_ID)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== M5E.1 diagnostics starting {datetime.now(timezone.utc).isoformat()} ===")

    health_before = snapshot_ollama_health(base_url=BASE_URL, model_name=MODEL)
    print("[OLLAMA HEALTH - before]", json.dumps(health_before, default=str)[:400])

    # -- 1. synthetic S1/S2/S3, both runtimes -----------------------------
    synthetic_telemetry = TelemetryLogger(os.path.join(output_dir, "synthetic_decisions.jsonl"))
    synthetic_summaries = []
    synthetic_runs_path = os.path.join(output_dir, "synthetic_runs.jsonl")
    with open(synthetic_runs_path, "w", encoding="utf-8") as synthetic_runs_file:
        for runtime in ("native_tool_like", "strands"):
            for pattern in PATTERNS:
                print(f"[SYNTHETIC] {runtime} / {pattern} starting {datetime.now(timezone.utc).isoformat()}")
                try:
                    result = run_synthetic_pattern(
                        pattern=pattern,
                        runtime=runtime,
                        n_decisions=N_DECISIONS_TARGET,
                        provider=PROVIDER,
                        model=MODEL,
                        base_url=BASE_URL,
                        timeout=LLM_TIMEOUT,
                        telemetry=synthetic_telemetry,
                        episode_id=f"synthetic__{runtime}__{pattern}",
                        seed=SEEDS[0],
                        per_decision_wall_timeout=180.0,
                    )
                    record = result.to_dict()
                    record["error"] = None
                except Exception as exc:  # never let one synthetic pattern take the whole run down
                    record = {"pattern": pattern, "runtime": runtime, "decisions": [], "error": f"{type(exc).__name__}: {exc}"}
                synthetic_runs_file.write(json.dumps(record, default=str) + "\n")
                synthetic_runs_file.flush()
                synthetic_summaries.append(
                    {
                        "runtime": runtime,
                        "pattern": pattern,
                        "decisions_completed": len(record["decisions"]),
                        "error": record["error"],
                    }
                )
                print(f"[SYNTHETIC] {runtime} / {pattern} finished: {synthetic_summaries[-1]}")
    synthetic_telemetry.close()

    # -- 2. live passive-vs-active gameplay matrix -------------------------
    print(f"[GAMEPLAY MATRIX] starting {datetime.now(timezone.utc).isoformat()}")
    spec = _build_gameplay_spec(output_root)
    runner = ExperimentRunner(episode_runner=EpisodeRunner(poll_sleep=0.0), output_root=output_root)
    result = runner.run(spec)
    print(render_terminal(result))

    health_after = snapshot_ollama_health(base_url=BASE_URL, model_name=MODEL)
    print("[OLLAMA HEALTH - after]", json.dumps(health_after, default=str)[:400])
    with open(os.path.join(output_dir, "ollama_health.json"), "w", encoding="utf-8") as f:
        json.dump({"before": health_before, "after": health_after}, f, indent=2, default=str)

    with open(os.path.join(output_dir, "synthetic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(synthetic_summaries, f, indent=2, default=str)

    print(f"\nOutputs written under: {output_dir}/")
    print(f"=== M5E.1 diagnostics finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
