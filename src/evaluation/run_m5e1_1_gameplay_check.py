"""M5E.1.1 spec section 39: post-diagnosis gameplay confirmation.

    python -m src.evaluation.run_m5e1_1_gameplay_check

Runs one native_tool_like and one Strands active-opponent episode at the
recommended `num_ctx` (see CHANGELOG.md's M5E.1.1 entry for how 16384 was
selected from the P0 matrix + wrapper-comparison evidence), to confirm
whether provider-level stabilization transfers back to real gameplay.
Appends to the same `evaluation_runs/m5e1_1-ollama-provider-stability-v1/`
directory the main M5E.1.1 diagnostic run already wrote to (a distinct
`experiment_id` inside it, never overwriting the provider/wrapper data).
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys

from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.reporting import render_terminal
from src.evaluation.spec import ExperimentSpec, VariantSpec

RECOMMENDED_NUM_CTX = 16384
EXPERIMENT_ID = "m5e1_1-gameplay-check-v1"
SEED = 421  # fresh, distinct from every prior milestone's seed sets


def main(argv: list[str] | None = None) -> int:
    spec = ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        variants=[
            VariantSpec(
                variant_id="A_native_tool_like_num_ctx_16384",
                agent_runtime="native_tool_like",
                provider="ollama",
                model="gemma4:latest",
                llm_timeout=60.0,
                opponent_type="scripted_active",
                opponent_profile="OPPORTUNIST",
                num_ctx=RECOMMENDED_NUM_CTX,
            ),
            VariantSpec(
                variant_id="B_strands_num_ctx_16384",
                agent_runtime="strands",
                provider="ollama",
                model="gemma4:latest",
                llm_timeout=60.0,
                opponent_type="scripted_active",
                opponent_profile="OPPORTUNIST",
                num_ctx=RECOMMENDED_NUM_CTX,
            ),
        ],
        seeds=[SEED],
        execution_mode="evaluation",
        max_episode_game_time=180.0,
        max_episode_wall_time=1800.0,
        max_decisions=15,
        max_runtime_failures=20,
    )
    runner = ExperimentRunner(episode_runner=EpisodeRunner(poll_sleep=0.0), output_root="evaluation_runs")
    result = runner.run(spec)
    print(render_terminal(result))
    print(f"\nOutputs written under: evaluation_runs/{EXPERIMENT_ID}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
