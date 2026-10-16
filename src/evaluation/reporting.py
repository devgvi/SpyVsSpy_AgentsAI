"""M5D: concise human-readable report over an ExperimentResult (spec
section 23: "summary.md or terminal report"). Descriptive statistics and
raw counts only - no significance claims from a small sample (spec
section 30).
"""

from __future__ import annotations

import os

from src.evaluation.results import ExperimentResult


def render_markdown(result: ExperimentResult) -> str:
    summary = result.to_summary_dict()
    lines = [
        f"# Experiment: {result.experiment_id}",
        "",
        f"Started: {result.started_at}  ",
        f"Finished: {result.finished_at}  ",
        "",
        f"Requested: {summary['requested']} | Supported: {summary['supported']} | "
        f"Completed: {summary['completed']} | Successful: {summary['successful']}",
        "",
        "_Descriptive statistics from a small sample only - no significance "
        "claims (M5D spec section 30)._",
        "",
    ]
    for variant_id, v in summary["by_variant"].items():
        lines.append(f"## {variant_id}")
        lines.append("")
        lines.append(
            f"- runtime: `{v['variant']['agent_runtime']}` | provider: `{v['variant']['provider']}` | "
            f"model: `{v['variant']['model']}`"
        )
        lines.append(
            f"- requested={v['requested']} supported={v['supported']} "
            f"completed={v['completed']} successful={v['successful']}"
        )
        if v["supported"]:
            lines.append(
                f"- success rate of supported: {v['successful']}/{v['supported']} "
                f"({v['success_rate_of_supported']:.0%})"
            )
        else:
            lines.append("- success rate of supported: N/A (no supported episodes)")
        lines.append(f"- compatibility statuses: {v['compatibility_statuses']}")
        lines.append(f"- episode statuses: {v['episode_statuses']}")
        lines.append("")
    return "\n".join(lines)


def render_terminal(result: ExperimentResult) -> str:
    return render_markdown(result)


def write_report(result: ExperimentResult, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
    return path
