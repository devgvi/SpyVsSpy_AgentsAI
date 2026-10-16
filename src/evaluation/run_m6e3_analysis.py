"""M6E.3 analysis (spec sections 25-27, 35-37, 58-63): reads
`native_window_trials.jsonl`, computes the horizon curve, window
classification, and S1/S0 control analysis. Analysis-only - no live
calls, no LLM-as-a-Judge.

    python -m src.evaluation.run_m6e3_analysis
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter

from src.evaluation.history_window import WINDOW_ORDER
from src.llm_controller import _normalize_action_name

OUT_DIR = os.path.join("evaluation_runs", "m6e3-memory-horizon-v1")

NO_HISTORY_WINDOW_EFFECT = "NO_HISTORY_WINDOW_EFFECT"
MONOTONIC_HISTORY_EFFECT = "MONOTONIC_HISTORY_EFFECT"
THRESHOLD_HISTORY_EFFECT = "THRESHOLD_HISTORY_EFFECT"
NON_MONOTONIC_HISTORY_EFFECT = "NON_MONOTONIC_HISTORY_EFFECT"

TV_MATERIAL_THRESHOLD = 0.3  # predeclared, consistent with M6E.2's own convention


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


def _decision_dist(rows: list[dict]) -> Counter:
    return Counter((_normalize_action_name(r.get("action")), r.get("target_id")) for r in rows)


def _tv_distance(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    na, nb = sum(a.values()) or 1, sum(b.values()) or 1
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def _modal(dist: Counter):
    return dist.most_common(1)[0] if dist else (None, 0)


def _cost(rows: list[dict]) -> dict:
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    it = [r["input_tokens"] for r in rows if r.get("input_tokens") is not None]
    ot = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
    ch = [r["characters"] for r in rows if r.get("characters") is not None]
    return {
        "latency_ms_mean": statistics.mean(lat) if lat else None,
        "input_tokens_mean": statistics.mean(it) if it else None,
        "output_tokens_mean": statistics.mean(ot) if ot else None,
        "characters_mean": statistics.mean(ch) if ch else None,
    }


def analyze_scenario(rows: list[dict], family: str) -> dict:
    windows_present = [w for w in WINDOW_ORDER if any(r["window"] == w for r in rows if r["scenario_family"] == family)]
    per_window = {}
    for w in windows_present:
        cell = [r for r in rows if r["scenario_family"] == family and r["window"] == w]
        dist = _decision_dist(cell)
        modal, modal_count = _modal(dist)
        n = len(cell) or 1
        repeat_vals = [r["repeats_known_failure"] for r in cell if r.get("repeats_known_failure") is not None]
        per_window[w] = {
            "samples": len(cell), "action_distribution": dict(Counter(a for a, t in dist.elements())),
            "target_distribution": dict(Counter(t for a, t in dist.elements())),
            "modal_decision": {"action": modal[0], "target_id": modal[1]} if modal else None,
            "modal_rate": modal_count / n, "validity_rate": sum(1 for r in cell if r["validation_status"] == "valid") / n,
            "repeat_known_failure_rate": (sum(repeat_vals) / len(repeat_vals)) if repeat_vals else None,
            "cost": _cost(cell),
        }

    # Boundary detection between adjacent tested windows (spec section 27).
    boundaries = []
    for a, b in zip(windows_present, windows_present[1:]):
        da, db = _decision_dist([r for r in rows if r["scenario_family"] == family and r["window"] == a]), _decision_dist([r for r in rows if r["scenario_family"] == family and r["window"] == b])
        tv = _tv_distance(da, db)
        ra, rb = per_window[a]["repeat_known_failure_rate"], per_window[b]["repeat_known_failure_rate"]
        repeat_diff = abs(ra - rb) if ra is not None and rb is not None else None
        material = tv >= TV_MATERIAL_THRESHOLD or (repeat_diff is not None and repeat_diff >= TV_MATERIAL_THRESHOLD)
        boundaries.append({"window_a": a, "window_b": b, "tv_distance": tv, "repeat_rate_diff": repeat_diff, "candidate_boundary": material})

    any_boundary = any(b["candidate_boundary"] for b in boundaries)
    if not any_boundary:
        classification = NO_HISTORY_WINDOW_EFFECT
    else:
        material_indices = [i for i, b in enumerate(boundaries) if b["candidate_boundary"]]
        classification = MONOTONIC_HISTORY_EFFECT if len(material_indices) == 1 else NON_MONOTONIC_HISTORY_EFFECT

    horizon_curve = [{"window": w, "repeat_known_failure_rate": per_window[w]["repeat_known_failure_rate"], "modal_decision": per_window[w]["modal_decision"]} for w in windows_present]

    return {"per_window": per_window, "boundaries": boundaries, "classification": classification, "horizon_curve": horizon_curve}


def _runtime_replication(native_rows: list[dict], strands_rows: list[dict]) -> list[dict]:
    out = []
    for family in sorted({r["scenario_family"] for r in strands_rows}):
        for window in sorted({r["window"] for r in strands_rows if r["scenario_family"] == family}):
            n_rows = [r for r in native_rows if r["scenario_family"] == family and r["window"] == window]
            s_rows = [r for r in strands_rows if r["scenario_family"] == family and r["window"] == window]
            if not n_rows or not s_rows:
                continue
            n_modal, s_modal = _modal(_decision_dist(n_rows))[0], _modal(_decision_dist(s_rows))[0]
            tv = _tv_distance(_decision_dist(n_rows), _decision_dist(s_rows))
            classification = "HISTORY_EFFECT_RUNTIME_INDEPENDENT" if n_modal == s_modal else ("PARTIAL_REPLICATION" if tv < 0.7 else "NO_REPLICATION")
            out.append({"scenario": family, "window": window, "native_modal": n_modal, "strands_modal": s_modal, "tv_distance": tv, "classification": classification})
    return out


def run() -> dict:
    rows = _read_jsonl(os.path.join(OUT_DIR, "native_window_trials.jsonl"))
    strands_rows = _read_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"))
    families = sorted({r["scenario_family"] for r in rows})
    results = {family: analyze_scenario(rows, family) for family in families}

    with open(os.path.join(OUT_DIR, "horizon_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({fam: r["horizon_curve"] for fam, r in results.items()}, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "boundary_analysis.json"), "w", encoding="utf-8") as f:
        json.dump({fam: {"boundaries": r["boundaries"], "classification": r["classification"]} for fam, r in results.items()}, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "cost_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({fam: {w: r["per_window"][w]["cost"] for w in r["per_window"]} for fam, r in results.items()}, f, indent=2, default=str)

    runtime_replication = _runtime_replication(rows, strands_rows)
    with open(os.path.join(OUT_DIR, "runtime_replication.json"), "w", encoding="utf-8") as f:
        json.dump(runtime_replication, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    out = run()
    for fam, r in out.items():
        print(f"=== {fam} === classification={r['classification']}")
        for w, m in r["per_window"].items():
            print(f"  {w}: modal={m['modal_decision']} repeat_rate={m['repeat_known_failure_rate']} valid={m['validity_rate']}")
        for b in r["boundaries"]:
            print(f"  boundary {b['window_a']}->{b['window_b']}: tv={b['tv_distance']:.2f} repeat_diff={b['repeat_rate_diff']} candidate={b['candidate_boundary']}")
