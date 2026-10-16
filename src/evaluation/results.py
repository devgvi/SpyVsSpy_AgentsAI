"""M5D: ExperimentResult - aggregates EpisodeResults with the denominators
spec section 24 requires kept distinct: requested (every expanded episode)
vs supported (compatibility passed) vs completed (status COMPLETED) vs
successful (COMPLETED and mission_success). Never divide a success count
by `requested` when `supported` is what the number is actually over.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.evaluation.episode_runner import EpisodeResult
from src.evaluation.spec import ExperimentSpec


@dataclass
class VariantSummary:
    variant_id: str
    variant: dict[str, Any]
    requested: int
    supported: int
    completed: int
    successful: int
    compatibility_statuses: dict[str, int]
    episode_statuses: dict[str, int]
    episode_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant": self.variant,
            "requested": self.requested,
            "supported": self.supported,
            "completed": self.completed,
            "successful": self.successful,
            "success_rate_of_supported": (self.successful / self.supported) if self.supported else None,
            "success_rate_of_requested": (self.successful / self.requested) if self.requested else None,
            "compatibility_statuses": self.compatibility_statuses,
            "episode_statuses": self.episode_statuses,
            "episode_ids": self.episode_ids,
        }


@dataclass
class ExperimentResult:
    experiment_id: str
    spec: ExperimentSpec
    episodes: list[EpisodeResult]
    config_snapshot: dict[str, Any]
    started_at: str
    finished_at: str

    def variant_summaries(self) -> list[VariantSummary]:
        summaries = []
        for variant in self.spec.variants:
            variant_episodes = [e for e in self.episodes if e.variant_id == variant.variant_id]
            supported_episodes = [e for e in variant_episodes if e.compatibility_status.value == "SUPPORTED"]
            completed_episodes = [e for e in supported_episodes if e.status.value == "COMPLETED"]
            successful_episodes = [e for e in completed_episodes if e.mission_success]
            summaries.append(
                VariantSummary(
                    variant_id=variant.variant_id,
                    variant=variant.to_dict(),
                    requested=len(variant_episodes),
                    supported=len(supported_episodes),
                    completed=len(completed_episodes),
                    successful=len(successful_episodes),
                    compatibility_statuses=dict(Counter(e.compatibility_status.value for e in variant_episodes)),
                    episode_statuses=dict(Counter(e.status.value for e in variant_episodes)),
                    episode_ids=[e.episode_id for e in variant_episodes],
                )
            )
        return summaries

    def to_summary_dict(self) -> dict[str, Any]:
        summaries = self.variant_summaries()
        return {
            "experiment_id": self.experiment_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config_snapshot": self.config_snapshot,
            "spec": self.spec.to_dict(),
            # Spec section 24: these three denominators are never conflated.
            "requested": len(self.episodes),
            "supported": sum(1 for e in self.episodes if e.compatibility_status.value == "SUPPORTED"),
            "completed": sum(1 for e in self.episodes if e.status.value == "COMPLETED"),
            "successful": sum(1 for e in self.episodes if e.status.value == "COMPLETED" and e.mission_success),
            "by_variant": {s.variant_id: s.to_dict() for s in summaries},
        }

    def write(self, output_dir: str) -> dict[str, str]:
        """Writes experiment.json, episodes.jsonl, summary.json under
        `output_dir` (spec section 23's suggested layout). Returns the
        paths written. Decision-level telemetry already lives in the
        caller-provided telemetry file (usually the same directory's
        decisions.jsonl) - not duplicated here."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {
            "experiment": os.path.join(output_dir, "experiment.json"),
            "episodes": os.path.join(output_dir, "episodes.jsonl"),
            "summary": os.path.join(output_dir, "summary.json"),
        }
        with open(paths["experiment"], "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_id": self.experiment_id,
                    "spec": self.spec.to_dict(),
                    "config_snapshot": self.config_snapshot,
                    "started_at": self.started_at,
                    "finished_at": self.finished_at,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        with open(paths["episodes"], "w", encoding="utf-8") as f:
            for episode in self.episodes:
                f.write(json.dumps(episode.to_dict(), sort_keys=True, default=str) + "\n")
        with open(paths["summary"], "w", encoding="utf-8") as f:
            json.dump(self.to_summary_dict(), f, indent=2, sort_keys=True, default=str)
        return paths
