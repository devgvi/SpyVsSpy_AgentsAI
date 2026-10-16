"""M5D: ExperimentRunner - orchestrates one ExperimentSpec end to end:

    variant x seed expansion
    -> per-episode compatibility + run (EpisodeRunner)
    -> aggregate ExperimentResult
    -> machine-readable + human outputs

One episode's failure never stops the rest of the experiment (spec
section 33) - unless the whole ExperimentSpec itself is invalid, which
fails immediately, before any episode runs.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from src.evaluation.compatibility import CompatibilityStatus
from src.evaluation.episode_runner import EpisodeResult, EpisodeRunner, EpisodeStatus
from src.evaluation.reporting import write_report
from src.evaluation.results import ExperimentResult
from src.evaluation.spec import ExperimentSpec

DEFAULT_OUTPUT_ROOT = "evaluation_runs"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2.0)
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:
        pass
    return None


def build_config_snapshot(spec: ExperimentSpec) -> dict[str, Any]:
    """Spec section 7: recorded once per experiment. Never logs credentials
    - only structural config (schema versions, seeds, variant shapes,
    platform facts), nothing from llm_api_key/etc."""
    strands_sdk_version = None
    try:
        import importlib.metadata

        strands_sdk_version = importlib.metadata.version("strands-agents")
    except Exception:
        pass

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "prompt_version": spec.prompt_version,
        "observation_schema_version": spec.observation_schema_version,
        "memory_schema_version": spec.memory_schema_version,
        "action_schema_version": spec.action_schema_version,
        "map_version": spec.map_version,
        "execution_mode": spec.execution_mode,
        "seeds": list(spec.seeds),
        "variants": [v.to_dict() for v in spec.variants],
        "agent_sdk_strands_agents_version": strands_sdk_version,
    }


class ExperimentRunner:
    def __init__(self, episode_runner: EpisodeRunner | None = None, output_root: str = DEFAULT_OUTPUT_ROOT):
        self.episode_runner = episode_runner or EpisodeRunner()
        self.output_root = output_root

    def run(self, spec: ExperimentSpec, write_outputs: bool = True) -> ExperimentResult:
        if not spec.variants:
            raise ValueError(f"ExperimentSpec {spec.experiment_id!r} has no variants")
        if not spec.seeds:
            raise ValueError(f"ExperimentSpec {spec.experiment_id!r} has no seeds")

        started_at = datetime.now(timezone.utc).isoformat()
        output_dir = os.path.join(self.output_root, spec.experiment_id)
        telemetry_path = os.path.join(output_dir, "decisions.jsonl")
        if write_outputs:
            os.makedirs(output_dir, exist_ok=True)

        episodes: list[EpisodeResult] = []
        for episode_spec in spec.expand():
            try:
                result = self.episode_runner.run(episode_spec, telemetry_path=telemetry_path)
            except Exception as exc:  # pragma: no cover - EpisodeRunner.run already
                # isolates per-episode failures internally; this is one more
                # layer so a single bad episode can never take the rest of
                # the experiment down with it (spec section 33).
                result = EpisodeResult(
                    experiment_id=episode_spec.experiment_id,
                    variant_id=episode_spec.variant_id,
                    episode_id=episode_spec.episode_id,
                    seed=episode_spec.seed,
                    status=EpisodeStatus.FAILED,
                    compatibility_status=CompatibilityStatus.UNKNOWN_COMPATIBILITY_FAILURE,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            episodes.append(result)

        finished_at = datetime.now(timezone.utc).isoformat()
        experiment_result = ExperimentResult(
            experiment_id=spec.experiment_id,
            spec=spec,
            episodes=episodes,
            config_snapshot=build_config_snapshot(spec),
            started_at=started_at,
            finished_at=finished_at,
        )

        if write_outputs:
            experiment_result.write(output_dir)
            write_report(experiment_result, output_dir)

        return experiment_result
