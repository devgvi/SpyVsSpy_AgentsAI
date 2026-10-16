"""M5D: automated, non-interactive experiment entry point.

    python -m src.evaluation.run --config experiments/gemma4_runtime_baseline.json

Headless by default (episode_runner.py sets SDL_VIDEODRIVER=dummy on
import) - no rendering, no interactive input, safe for CI or a terminal
with no display.
"""

from __future__ import annotations

import argparse
import sys

from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.reporting import render_terminal
from src.evaluation.spec import ExperimentSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an M5D controlled agent evaluation experiment.")
    parser.add_argument("--config", required=True, help="Path to an ExperimentSpec JSON file.")
    parser.add_argument("--output-root", default="evaluation_runs", help="Directory under which experiment_id/ is written.")
    parser.add_argument("--no-write", action="store_true", help="Compute results without writing output files.")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        spec = ExperimentSpec.from_json(f.read())

    runner = ExperimentRunner(output_root=args.output_root)
    result = runner.run(spec, write_outputs=not args.no_write)

    print(render_terminal(result))
    if not args.no_write:
        print(f"\nOutputs written under: {args.output_root}/{spec.experiment_id}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
