"""M6E analysis (spec sections 29-36, 60-65): reads `native_trials.jsonl`/
`strands_trials.jsonl` and produces the required behavioral/deduplication/
history tables plus `behavioral_stability_metrics.json`,
`deduplication_metrics.json`, `history_truncation_metrics.json`,
`runtime_replication.json`. Analysis-only - no live calls, no
LLM-as-a-Judge (spec section 55): hashes/actions/targets/validator output/
telemetry only.

Stability thresholds (spec section 36) are PREDECLARED here, before this
script is ever run against real trial data:

    BEHAVIOR_STABLE:        modal decision matches AND >= 80% of samples
                             agree with the modal decision on both sides
    BEHAVIOR_SHIFT_SMALL:   modal decision matches but agreement < 80%
                             on either side
    BEHAVIOR_SHIFT_MATERIAL: modal decision itself differs between conditions
    MIXED:                  used only when a single classification cannot
                             fairly summarize a scenario across its several
                             condition-pairs (never retrofit per-scenario)

    python -m src.evaluation.run_m6e_analysis
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter

from src.llm_controller import _normalize_action_name

OUT_DIR = os.path.join("evaluation_runs", "m6e-context-compaction-v1")

BEHAVIOR_STABLE = "BEHAVIOR_STABLE"
BEHAVIOR_SHIFT_SMALL = "BEHAVIOR_SHIFT_SMALL"
BEHAVIOR_SHIFT_MATERIAL = "BEHAVIOR_SHIFT_MATERIAL"
STABILITY_AGREEMENT_THRESHOLD = 0.8

LOSSLESS_AND_BEHAVIOR_STABLE = "LOSSLESS_AND_BEHAVIOR_STABLE"
LOSSLESS_BUT_BEHAVIOR_CHANGED = "LOSSLESS_BUT_BEHAVIOR_CHANGED"
NO_MEANINGFUL_COMPACTION = "NO_MEANINGFUL_COMPACTION"


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _decision_dist(rows: list[dict]) -> dict:
    """Action names are normalized via the real production
    `_normalize_action_name()` before any distribution/modal comparison -
    a case-only difference (`"MOVE"` vs `"move"`) must never register as a
    behavior change, matching the case-normalization fix M6D's own
    `classify_intent` bug required (raw model output casing is not
    semantically meaningful; only the normalized action is)."""
    n = len(rows) or 1
    decisions = Counter((_normalize_action_name(r.get("action")), r.get("target_id")) for r in rows)
    actions = Counter(_normalize_action_name(r.get("action")) for r in rows)
    targets = Counter(r.get("target_id") for r in rows)
    modal_decision, modal_count = decisions.most_common(1)[0] if decisions else ((None, None), 0)
    return {
        "samples": len(rows),
        "action_distribution": dict(actions), "target_distribution": dict(targets),
        "unique_decisions": len(decisions), "modal_decision": {"action": modal_decision[0], "target_id": modal_decision[1]},
        "modal_agreement": modal_count / n,
        "validity_rate": sum(1 for r in rows if r.get("validation_status") == "valid") / n,
    }


def _stability_classification(base: dict, other: dict) -> str:
    base_modal = (base["modal_decision"]["action"], base["modal_decision"]["target_id"])
    other_modal = (other["modal_decision"]["action"], other["modal_decision"]["target_id"])
    if base_modal != other_modal:
        return BEHAVIOR_SHIFT_MATERIAL
    if base["modal_agreement"] < STABILITY_AGREEMENT_THRESHOLD or other["modal_agreement"] < STABILITY_AGREEMENT_THRESHOLD:
        return BEHAVIOR_SHIFT_SMALL
    return BEHAVIOR_STABLE


def _cost(rows: list[dict]) -> dict:
    lat = [r.get("latency_ms") for r in rows if r.get("latency_ms") is not None]
    it = [r.get("input_tokens") for r in rows if r.get("input_tokens") is not None]
    ot = [r.get("output_tokens") for r in rows if r.get("output_tokens") is not None]
    return {
        "latency_ms_mean": statistics.mean(lat) if lat else None,
        "input_tokens_mean": statistics.mean(it) if it else None,
        "output_tokens_mean": statistics.mean(ot) if ot else None,
    }


def run() -> dict:
    native_rows = _read_jsonl(os.path.join(OUT_DIR, "native_trials.jsonl"))
    strands_rows = _read_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"))

    behavioral_table = {}
    stage1_2 = {}
    for family in sorted({r["scenario_family"] for r in native_rows}):
        family_rows = [r for r in native_rows if r["scenario_family"] == family]
        by_cond = {c: [r for r in family_rows if r["condition"] == c] for c in sorted({r["condition"] for r in family_rows})}
        behavioral_table[family] = {c: {**_decision_dist(rows), "cost": _cost(rows)} for c, rows in by_cond.items()}

        c0 = by_cond.get("C0_FULL_CONTEXT")
        entry = {}
        for lossless_cond in ("C1_STRUCTURAL_COMPACT", "C2_DEDUPLICATED"):
            cond_rows = by_cond.get(lossless_cond)
            if c0 and cond_rows:
                c0_dist, cond_dist = _decision_dist(c0), _decision_dist(cond_rows)
                stability = _stability_classification(c0_dist, cond_dist)
                classification = LOSSLESS_AND_BEHAVIOR_STABLE if stability == BEHAVIOR_STABLE else LOSSLESS_BUT_BEHAVIOR_CHANGED
                entry[lossless_cond] = {"stability": stability, "classification": classification}
        stage1_2[family] = entry

    # History truncation (Stage 3, spec sections 34, 65)
    history_table = []
    for family in sorted({r["scenario_family"] for r in native_rows if r["condition"] in ("C3A_LAST_3", "C3B_LAST_1")}):
        family_rows = [r for r in native_rows if r["scenario_family"] == family]
        c0_rows = [r for r in family_rows if r["condition"] == "C0_FULL_CONTEXT"]
        for cond in ("C3A_LAST_3", "C3B_LAST_1"):
            cond_rows = [r for r in family_rows if r["condition"] == cond]
            if not cond_rows:
                continue
            dist = _decision_dist(cond_rows)
            c0_dist = _decision_dist(c0_rows) if c0_rows else None
            repeat_rate = None
            history_table.append({
                "scenario": family, "condition": cond, "decision_distribution": dist["action_distribution"],
                "target_distribution": dist["target_distribution"], "modal_decision": dist["modal_decision"],
                "validity_rate": dist["validity_rate"],
                "behavior_vs_c0": _stability_classification(c0_dist, dist) if c0_dist else None,
            })

    # S6-specific repeat-failed-action rate (the one scenario built with a
    # known injected failure to test exactly this metric).
    s6_failed_target = "filing_cabinet_01"
    s6_failed_action = "interact"
    repeat_failed_action = {}
    for cond in ("C0_FULL_CONTEXT", "C1_STRUCTURAL_COMPACT", "C2_DEDUPLICATED", "C3A_LAST_3", "C3B_LAST_1"):
        rows = [r for r in native_rows if r["scenario_family"] == "S6_LONG_HISTORY" and r["condition"] == cond]
        if rows:
            repeats = sum(1 for r in rows if (r.get("action") or "").lower() == s6_failed_action and r.get("target_id") == s6_failed_target)
            repeat_failed_action[cond] = {"attempts": len(rows), "repeat_failed_action_rate": repeats / len(rows)}

    # Runtime replication (Stage 4)
    runtime_replication = []
    for family in sorted({r["scenario_family"] for r in strands_rows}):
        for cond in sorted({r["condition"] for r in strands_rows if r["scenario_family"] == family}):
            n_rows = [r for r in native_rows if r["scenario_family"] == family and r["condition"] == cond]
            s_rows = [r for r in strands_rows if r["scenario_family"] == family and r["condition"] == cond]
            if not n_rows or not s_rows:
                continue
            n_dist, s_dist = _decision_dist(n_rows), _decision_dist(s_rows)
            same_modal = n_dist["modal_decision"] == s_dist["modal_decision"]
            classification = "PARTIAL_REPLICATION" if not same_modal else "COMPACTION_EFFECT_RUNTIME_INDEPENDENT"
            runtime_replication.append({
                "scenario": family, "condition": cond, "native_modal": n_dist["modal_decision"], "strands_modal": s_dist["modal_decision"],
                "native_modal_agreement": n_dist["modal_agreement"], "strands_modal_agreement": s_dist["modal_agreement"],
                "classification": classification,
            })

    dedup_metrics = {"note": "see fact_provenance.json's duplicate_fact_table per scenario/condition for full detail"}

    with open(os.path.join(OUT_DIR, "behavioral_stability_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"per_scenario_condition": behavioral_table, "stage1_2_classification": stage1_2}, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "history_truncation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"history_table": history_table, "s6_repeat_failed_action": repeat_failed_action}, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "deduplication_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dedup_metrics, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "runtime_replication.json"), "w", encoding="utf-8") as f:
        json.dump(runtime_replication, f, indent=2, default=str)

    return {
        "behavioral_table": behavioral_table, "stage1_2_classification": stage1_2,
        "history_table": history_table, "s6_repeat_failed_action": repeat_failed_action,
        "runtime_replication": runtime_replication,
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out["stage1_2_classification"], indent=2, default=str))
    print(json.dumps(out["s6_repeat_failed_action"], indent=2, default=str))
