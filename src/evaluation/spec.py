"""M5D: ExperimentSpec - serializable, deterministic experiment configuration.

Central invariant (see M5D_CONTROLLED_AGENT_EVALUATION_HARNESS.md section 2):
same controlled game conditions, change only the intended experimental
variables. `VariantSpec` is the thing that varies; `ExperimentSpec` is the
fixed frame (seeds, limits, schema versions) everything else holds equal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from src.llm.agent_memory import AgentMemory
from src.llm_controller import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, PROMPT_VERSION

MAP_VERSION = "level_01"


@dataclass(frozen=True)
class VariantSpec:
    """One experimental configuration - which runtime/provider/model, and
    the settings needed to construct it. `variant_id` is stable and unique
    within an experiment: it's part of every episode's correlation id
    (spec section 6) and the join key `results.py` aggregates on.
    """

    variant_id: str
    agent_runtime: str  # "native" | "strands"
    provider: str = "ollama"
    model: str = "qwen:latest"
    llm_base_url: str | None = None
    llm_timeout: float | None = None
    # Descriptive/recorded only for M5D: NativeAgentAdapter/StrandsAgentAdapter
    # both fix temperature=0.0 internally for deterministic comparison: this
    # field exists so a config snapshot states that explicitly rather than
    # leaving it implicit, not because per-variant temperature is wired
    # through yet.
    temperature: float = 0.0
    # White Spy's own profile - only meaningful when `agent_runtime` is
    # itself a scripted controller (M5E's "scripted reference" variant);
    # ignored for every LLM-driven variant.
    agent_profile: str = "OPPORTUNIST"
    # M5E: "passive" (the pre-M5E default: HumanController with no real
    # keyboard input, Black Spy never acts) or "scripted_active" (Black
    # Spy driven by ScriptedBotController - see Game's black_controller
    # param). Wired through EpisodeRunner -> Game(black_controller=...).
    opponent_type: str = "passive"
    # M5E: the scripted opponent's own AgentProfile name (bot_controller.
    # AGENT_PROFILES key) - only meaningful when opponent_type is
    # "scripted_active".
    opponent_profile: str = "OPPORTUNIST"
    # M5E.1.1: optional context-window/residency controls (spec sections
    # 3-4, 16), wired through EpisodeRunner -> Game(num_ctx=...,
    # keep_alive=...). Both None by default - every pre-M5E.1.1 config
    # gets byte-identical behavior (Ollama's own defaults).
    num_ctx: int | None = None
    keep_alive: str | None = None
    # M5E.3: Bedrock's own region concept (spec section 5), wired through
    # EpisodeRunner -> Game(llm_region=...). None by default - meaningless
    # for every non-Bedrock provider, which resolves it however it always
    # has.
    llm_region: str | None = None
    # M5F.1: White Spy's own personality profile (spec section 13) - only
    # meaningful for an LLM-driven `agent_runtime`; None ("default") is
    # byte-identical pre-M5F.1 behavior. Wired through EpisodeRunner ->
    # Game(white_personality=...).
    personality_profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariantSpec":
        return cls(**data)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    variants: tuple[VariantSpec, ...]
    seeds: tuple[int, ...]
    execution_mode: str = "evaluation"
    max_episode_game_time: float = 180.0
    max_episode_wall_time: float = 180.0
    max_decisions: int = 8
    max_runtime_failures: int = 5
    prompt_version: str = PROMPT_VERSION
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    memory_schema_version: str = AgentMemory.MEMORY_SCHEMA_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION
    map_version: str = MAP_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "variants": [v.to_dict() for v in self.variants],
            "seeds": list(self.seeds),
            "execution_mode": self.execution_mode,
            "max_episode_game_time": self.max_episode_game_time,
            "max_episode_wall_time": self.max_episode_wall_time,
            "max_decisions": self.max_decisions,
            "max_runtime_failures": self.max_runtime_failures,
            "prompt_version": self.prompt_version,
            "observation_schema_version": self.observation_schema_version,
            "memory_schema_version": self.memory_schema_version,
            "action_schema_version": self.action_schema_version,
            "map_version": self.map_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        kwargs = dict(data)
        kwargs["variants"] = tuple(VariantSpec.from_dict(v) for v in data["variants"])
        kwargs["seeds"] = tuple(data["seeds"])
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ExperimentSpec":
        return cls.from_dict(json.loads(text))

    def expand(self) -> list["EpisodeSpec"]:
        """Deterministic variant x seed expansion: variants outer loop,
        seeds inner loop, always the same stable order for the same spec -
        so re-running the same ExperimentSpec produces the same episode_id
        sequence (spec section 3: "variant x seed expansion must be
        deterministic")."""
        episodes: list[EpisodeSpec] = []
        for variant in self.variants:
            for seed in self.seeds:
                episode_id = f"{self.experiment_id}__{variant.variant_id}__seed{seed}"
                episodes.append(
                    EpisodeSpec(
                        experiment_id=self.experiment_id,
                        episode_id=episode_id,
                        variant=variant,
                        seed=seed,
                        execution_mode=self.execution_mode,
                        max_episode_game_time=self.max_episode_game_time,
                        max_episode_wall_time=self.max_episode_wall_time,
                        max_decisions=self.max_decisions,
                        max_runtime_failures=self.max_runtime_failures,
                        prompt_version=self.prompt_version,
                        observation_schema_version=self.observation_schema_version,
                        memory_schema_version=self.memory_schema_version,
                        action_schema_version=self.action_schema_version,
                        map_version=self.map_version,
                    )
                )
        return episodes


@dataclass(frozen=True)
class EpisodeSpec:
    """One fully-resolved (variant, seed) pair - everything EpisodeRunner
    needs to run exactly one episode, with nothing left ambiguous. This is
    the correlation root: experiment_id + episode_id + seed + variant_id
    (spec section 6)."""

    experiment_id: str
    episode_id: str
    variant: VariantSpec
    seed: int
    execution_mode: str
    max_episode_game_time: float
    max_episode_wall_time: float
    max_decisions: int
    max_runtime_failures: int
    prompt_version: str
    observation_schema_version: str
    memory_schema_version: str
    action_schema_version: str
    map_version: str
    # M6A.2 spec sections 25, 45: mirrors `duel_runner.py`'s own
    # `max_validation_failures` (established in M5F.3B.1) - a separate,
    # more generous cap for cleanly-repaired-or-fallen-back MODEL
    # VALIDATION failures, distinct from the `max_runtime_failures` cap
    # above (now checked against genuine `provider_failure_count`, not
    # the combined `fallback_count` - see `EpisodeRunner.run_episode`'s
    # own real-bug-fix comment). `None` (every pre-M6A.2 caller's
    # implicit value) means "no separate validation cap" - byte-identical
    # to every existing caller's prior behavior.
    max_validation_failures: int | None = None

    @property
    def variant_id(self) -> str:
        return self.variant.variant_id
