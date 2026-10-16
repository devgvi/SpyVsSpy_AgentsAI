"""M6F Phase C analysis (spec sections 29-35, 60-63): reads
`native_trials.jsonl`/`strands_trials.jsonl` (live S0/S1/S3 rows from
`run_m6f_phase_c.py` + reused S2 rows from `run_m6f_repair_reuse.py`) and
produces the required scenario/first-difference/divergence tables plus
`runtime_divergence_metrics.json` and `cost_metrics.json`. Analysis-only -
no live calls, no model judging (spec section 53: hashes/structured
actions/validator output/telemetry only).

    python -m src.evaluation.run_m6f_analysis
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict

OUT_DIR = os.path.join("evaluation_runs", "m6f-runtime-interoperability-v1")

# Canonical request hashes for S0/S1/S3, computed once from
# `scenario_manifest.json` (built by m6f_scenarios.py) - live trial rows
# for these three families already carry their own `canonical_request_hash`
# field directly (see run_m6f_phase_c.py), so this is only needed to cross
# check S2's reused rows, which predate agent_contract.py and therefore
# carry `repair_fact_hash`/`repair_representation_hash` instead.


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


def _behavioral_metrics(rows: list[dict], action_key: str, target_key: str, valid_key: str) -> dict:
    n = len(rows) or 1
    actions = Counter(r.get(action_key) for r in rows)
    targets = Counter(r.get(target_key) for r in rows)
    decisions = {(r.get(action_key), r.get(target_key)) for r in rows}
    return {
        "attempts": len(rows),
        "action_distribution": dict(actions),
        "target_distribution": dict(targets),
        "modal_action": actions.most_common(1)[0][0] if actions else None,
        "modal_target": targets.most_common(1)[0][0] if targets else None,
        "unique_decisions": len(decisions),
        "validity_rate": sum(1 for r in rows if r.get(valid_key)) / n,
    }


def _cost_metrics(rows: list[dict], get_latency, get_in_tok, get_out_tok) -> dict:
    latencies = [x for x in (get_latency(r) for r in rows) if x is not None]
    in_toks = [x for x in (get_in_tok(r) for r in rows) if x is not None]
    out_toks = [x for x in (get_out_tok(r) for r in rows) if x is not None]
    return {
        "n": len(rows),
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "input_tokens_mean": statistics.mean(in_toks) if in_toks else None,
        "output_tokens_mean": statistics.mean(out_toks) if out_toks else None,
        "total_tokens_mean": (statistics.mean(in_toks) + statistics.mean(out_toks)) if in_toks and out_toks else None,
    }


def run() -> dict:
    native_rows = _read_jsonl(os.path.join(OUT_DIR, "native_trials.jsonl"))
    strands_rows = _read_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"))

    scenario_table: dict[str, dict] = {}
    first_difference_table: list[dict] = []
    divergence: dict[str, str] = {}

    for family in ("S0_ORDINARY_DECISION", "S1_MEMORY_DECISION", "S3_TOOL_ACTION_DECISION"):
        n_rows = [r for r in native_rows if r.get("scenario_family") == family]
        s_rows = [r for r in strands_rows if r.get("scenario_family") == family]
        if not n_rows or not s_rows:
            continue
        n_hashes = {r.get("canonical_request_hash") for r in n_rows}
        s_hashes = {r.get("canonical_request_hash") for r in s_rows}
        canonical_match = n_hashes == s_hashes and len(n_hashes) == 1

        n_metrics = _behavioral_metrics(
            n_rows, action_key=None, target_key=None, valid_key="validation_status",
        )
        # normalized_action is a nested dict - re-derive distributions from it directly.
        def _dist(rows):
            actions = Counter((r.get("normalized_action") or {}).get("action") for r in rows)
            targets = Counter((r.get("normalized_action") or {}).get("target_id") for r in rows)
            decisions = {((r.get("normalized_action") or {}).get("action"), (r.get("normalized_action") or {}).get("target_id")) for r in rows}
            valid = sum(1 for r in rows if r.get("validation_status") == "valid") / (len(rows) or 1)
            return {
                "attempts": len(rows), "action_distribution": dict(actions), "target_distribution": dict(targets),
                "modal_action": actions.most_common(1)[0][0] if actions else None,
                "modal_target": targets.most_common(1)[0][0] if targets else None,
                "unique_decisions": len(decisions), "validity_rate": valid,
            }

        n_behavior = _dist(n_rows)
        s_behavior = _dist(s_rows)
        n_cost = _cost_metrics(
            n_rows, lambda r: (r.get("provider_metadata") or {}).get("latency_ms"),
            lambda r: (r.get("provider_metadata") or {}).get("input_tokens"),
            lambda r: (r.get("provider_metadata") or {}).get("output_tokens"),
        )
        s_cost = _cost_metrics(
            s_rows, lambda r: (r.get("provider_metadata") or {}).get("latency_ms"),
            lambda r: (r.get("provider_metadata") or {}).get("input_tokens"),
            lambda r: (r.get("provider_metadata") or {}).get("output_tokens"),
        )

        behavior_matches = n_behavior["action_distribution"] == s_behavior["action_distribution"]

        if not canonical_match:
            classification = "HARNESS_PARITY_FAILURE"
            first_diff_layer = "canonical_request"
        elif n_behavior["modal_action"] == s_behavior["modal_action"] and behavior_matches:
            classification = "NO_RUNTIME_DIVERGENCE"
            first_diff_layer = "none"
        else:
            n_repr_hashes = {r.get("runtime_representation_hash") for r in n_rows}
            s_repr_hashes = {r.get("runtime_representation_hash") for r in s_rows}
            if n_repr_hashes != s_repr_hashes:
                classification = "RUNTIME_REPRESENTATION_DIVERGENCE"
                first_diff_layer = "runtime_representation"
            else:
                classification = "RAW_MODEL_OUTPUT_DIVERGENCE"
                first_diff_layer = "raw_model_output"

        divergence[family] = classification
        scenario_table[family] = {
            "canonical_request_hash_native": sorted(n_hashes), "canonical_request_hash_strands": sorted(s_hashes),
            "canonical_request_match": canonical_match,
            "native": {**n_behavior, "cost": n_cost}, "strands": {**s_behavior, "cost": s_cost},
        }
        first_difference_table.append({
            "scenario": family, "first_differing_layer": first_diff_layer,
            "native_representation_hash": sorted({r.get("runtime_representation_hash") for r in n_rows}),
            "strands_representation_hash": sorted({r.get("runtime_representation_hash") for r in s_rows}),
            "behavior_diverged": not behavior_matches,
            "classification": classification,
        })

    # S2 repair interoperability (spec section 62) - reused M6D.2 data.
    repair_table = []
    for family in ("INVALID_TRAP_TYPE", "MISSING_REQUIRED_FIELD"):
        for contract in ("N0_CURRENT_PRODUCTION", "N2_NEUTRAL_STRUCTURED"):
            n_rows = [r for r in native_rows if r.get("scenario_family") == "S2_REPAIR_DECISION" and r.get("failure_family") == family and r.get("contract") == contract]
            s_rows = [r for r in strands_rows if r.get("scenario_family") == "S2_REPAIR_DECISION" and r.get("failure_family") == family and r.get("contract") == contract]
            if not n_rows and not s_rows:
                continue
            for runtime, rows in (("native", n_rows), ("strands", s_rows)):
                if not rows:
                    continue
                n = len(rows)
                retained = sum(1 for r in rows if r.get("intent_category") in ("SAME_ACTION_SAME_TARGET", "SAME_ACTION_DIFFERENT_TARGET")) / n
                valid = sum(1 for r in rows if r.get("repair_valid")) / n
                actions = Counter(r.get("final_action") for r in rows)
                repr_hashes = sorted({r.get("repair_representation_hash") for r in rows})
                repair_table.append({
                    "failure_family": family, "contract": contract, "runtime": runtime, "attempts": n,
                    "validity_rate": valid, "same_action_retention": retained,
                    "action_distribution": dict(actions), "repair_representation_hash": repr_hashes,
                })

    result = {
        "scenario_table": scenario_table,
        "first_difference_table": first_difference_table,
        "divergence_classification": divergence,
        "repair_interoperability_table": repair_table,
    }
    with open(os.path.join(OUT_DIR, "runtime_divergence_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    cost_result = {
        family: {"native": scenario_table[family]["native"]["cost"], "strands": scenario_table[family]["strands"]["cost"]}
        for family in scenario_table
    }
    with open(os.path.join(OUT_DIR, "cost_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(cost_result, f, indent=2, default=str)

    return result


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str)[:4000])
