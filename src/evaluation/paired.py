"""M5D.1: paired per-seed comparison between two variants that share the
same seed set - controls for differences in generated world difficulty
(spec section 18: "side-by-side per-seed results ... This controls for
differences in generated world difficulty").

Pure over an ExperimentResult's already-computed episodes/metrics - no new
gameplay, no aggregation logic duplicated from results.py.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.results import ExperimentResult

DEFAULT_FIELDS: tuple[str, ...] = (
    "mission_success",
    "mission_progress_ratio",
    "decisions_requested",
    "first_pass_valid",
    "repair_attempts",
    "fallbacks",
    "final_valid_llm_actions",
)


def paired_seed_rows(
    result: ExperimentResult,
    variant_a_id: str,
    variant_b_id: str,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> list[dict[str, Any]]:
    """One row per seed shared by both variants: seed, then each variant's
    episode status/game_time/wall_time and the requested metric fields.
    Only seeds present (as SUPPORTED episodes) on *both* sides are included -
    an incompatible variant contributes no rows, by construction never
    entering this comparison (spec section 4: "D must never enter gameplay
    metrics").
    """
    by_seed_a = {e.seed: e for e in result.episodes if e.variant_id == variant_a_id and e.compatibility_status.value == "SUPPORTED"}
    by_seed_b = {e.seed: e for e in result.episodes if e.variant_id == variant_b_id and e.compatibility_status.value == "SUPPORTED"}
    shared_seeds = sorted(set(by_seed_a) & set(by_seed_b))

    rows = []
    for seed in shared_seeds:
        ep_a, ep_b = by_seed_a[seed], by_seed_b[seed]
        row: dict[str, Any] = {
            "seed": seed,
            f"{variant_a_id}.status": ep_a.status.value,
            f"{variant_b_id}.status": ep_b.status.value,
            f"{variant_a_id}.game_time": ep_a.game_time,
            f"{variant_b_id}.game_time": ep_b.game_time,
            f"{variant_a_id}.wall_time": ep_a.wall_time,
            f"{variant_b_id}.wall_time": ep_b.wall_time,
        }
        for field_name in fields:
            row[f"{variant_a_id}.{field_name}"] = ep_a.metrics.get(field_name)
            row[f"{variant_b_id}.{field_name}"] = ep_b.metrics.get(field_name)
        rows.append(row)
    return rows


def render_paired_markdown_table(
    result: ExperimentResult,
    variant_a_id: str,
    variant_b_id: str,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> str:
    rows = paired_seed_rows(result, variant_a_id, variant_b_id, fields)
    if not rows:
        return f"_No shared supported seeds between `{variant_a_id}` and `{variant_b_id}`._"

    columns = ["seed", f"{variant_a_id}.status", f"{variant_b_id}.status"]
    for field_name in fields:
        columns.append(f"{variant_a_id}.{field_name}")
        columns.append(f"{variant_b_id}.{field_name}")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)
