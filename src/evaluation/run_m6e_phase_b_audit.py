"""M6E Phase B (spec sections 28, 60): static token audit, computed
entirely from network-free condition transforms - no live calls. Produces
`static_token_metrics.json`, `context_compaction_audit.md`,
`fact_provenance.json`, `snapshot_manifest.json`,
`compaction_condition_manifest.json`.

    python -m src.evaluation.run_m6e_phase_b_audit
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src.evaluation.context_compaction import CONDITIONS, apply_condition, representation_hash, token_estimate
from src.evaluation.m6e_scenarios import SCENARIO_FAMILIES, build_all_scenarios, build_compact_observation_payload
from src.evaluation.request_fact_model import duplicate_fact_table, extract_request_facts, request_fact_hash

OUT_DIR = os.path.join("evaluation_runs", "m6e-context-compaction-v1")
_HISTORY_FIELDS = ("recent_interactions", "recent_failures", "recent_events", "recent_decisions")


def run() -> dict:
    scenarios = build_all_scenarios()
    condition_table = []
    fact_provenance = {}
    snapshot_manifest = {}

    for family in SCENARIO_FAMILIES:
        scenario = scenarios[family]
        obs = build_compact_observation_payload(scenario)
        mem = scenario.memory_payload
        snapshot_manifest[family] = {"scenario_id": scenario.scenario_id}

        for condition in CONDITIONS:
            result = apply_condition(condition, obs, mem)
            facts = extract_request_facts(result.observation_payload, result.memory_payload)
            dup_table = duplicate_fact_table(facts)
            text = result.to_json()
            history_items = sum(len(result.memory_payload.get(f, [])) for f in _HISTORY_FIELDS)

            row = {
                "scenario": family, "condition": condition,
                "lossless": condition != "C3A_LAST_3" and condition != "C3B_LAST_1",
                "fact_count": len(facts), "unique_fact_count": len({f.semantic_signature() for f in facts}),
                "duplicate_fact_count": len(dup_table), "history_items": history_items,
                "characters": len(text), "estimated_tokens": token_estimate(text),
                "request_fact_hash": request_fact_hash(facts), "representation_hash": representation_hash(text),
            }
            condition_table.append(row)
            fact_provenance.setdefault(family, {})[condition] = {
                "facts": [{"fact_id": f.fact_id, "category": f.category, "source_sections": list(f.source_sections), "chronological": f.chronological} for f in facts],
                "duplicate_fact_table": dup_table,
            }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "static_token_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(condition_table, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "fact_provenance.json"), "w", encoding="utf-8") as f:
        json.dump(fact_provenance, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "snapshot_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot_manifest, f, indent=2, default=str)

    from src.evaluation.context_compaction import DEDUPLICATION_RULES

    with open(os.path.join(OUT_DIR, "compaction_condition_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "conditions": list(CONDITIONS),
            "deduplication_rules": list(DEDUPLICATION_RULES),
            "c1_removed_representation_keys": ["available_actions_text"],
            "c3_history_fields": list(_HISTORY_FIELDS),
        }, f, indent=2, default=str)

    _write_audit_md(condition_table)
    return {"rows": len(condition_table)}


def _write_audit_md(condition_table: list[dict]) -> None:
    lines = [
        "# M6E Phase B — Static Token Audit\n",
        "Spec sections 28, 60. Computed entirely network-free from the real",
        "`compact_observation_for_model()` + `AgentMemory.to_compact_dict()`",
        "shapes, before any live call.\n",
        "| Scenario | Condition | Lossless? | Facts | Unique | Duplicates | History items | Characters | Est. tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in condition_table:
        lines.append(
            f"| {row['scenario']} | {row['condition']} | {row['lossless']} | {row['fact_count']} | "
            f"{row['unique_fact_count']} | {row['duplicate_fact_count']} | {row['history_items']} | "
            f"{row['characters']} | {row['estimated_tokens']} |"
        )
    lines.append("")
    lines.append("## C1 compaction ratio vs C0 (characters)\n")
    by_scenario: dict[str, dict[str, int]] = {}
    for row in condition_table:
        by_scenario.setdefault(row["scenario"], {})[row["condition"]] = row["characters"]
    lines.append("| Scenario | C0 chars | C1 chars | Reduction |")
    lines.append("|---|---|---|---|")
    for scenario, vals in by_scenario.items():
        c0, c1 = vals.get("C0_FULL_CONTEXT"), vals.get("C1_STRUCTURAL_COMPACT")
        if c0 and c1:
            pct = (1 - c1 / c0) * 100
            lines.append(f"| {scenario} | {c0} | {c1} | {pct:.1f}% |")

    with open(os.path.join(OUT_DIR, "context_compaction_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    result = run()
    print(f"[DONE] Phase B static audit: {result}")
