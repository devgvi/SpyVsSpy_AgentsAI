"""M6E.2 analysis (spec sections 24-27, 34-35, 62-66): reads
`native_trials.jsonl`, computes main effects, pairwise interactions, and
the three-way interaction. Analysis-only - no live calls, no
LLM-as-a-Judge (spec section 57): hashes/action signatures/distribution
metrics/telemetry only.

Predeclared before computing any interaction statistic (spec section 27):
a total-variation distance of >= 0.3 between two decision distributions
(over normalized (action, target_id) pairs) counts as a material
difference; below that, distributions are treated as descriptively the
same. This is a fixed, round, non-data-fit threshold applied uniformly
everywhere below - never adjusted per cell to fit a story.

    python -m src.evaluation.run_m6e2_analysis
"""

from __future__ import annotations

import json
import os
from collections import Counter

from src.evaluation.m6e2_factorial import (
    A0_SINGLE_COPY,
    A1_DOUBLE_COPY,
    B0_NATURAL_LOCATION,
    B1_AUXILIARY_LOCATION,
    C0_FULL_FORMAT,
    C1_COMPACT_FORMAT,
    FACTORIAL_CELLS,
)
from src.llm_controller import _normalize_action_name

OUT_DIR = os.path.join("evaluation_runs", "m6e2-representation-interaction-matrix-v1")
TV_MATERIAL_THRESHOLD = 0.3  # predeclared, spec section 27

NO_EFFECT = "NO_EFFECT"
MATERIAL_EFFECT = "MATERIAL_EFFECT"


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
    return dist.most_common(1)[0][0] if dist else None


def _stringify_dist(dist: Counter) -> dict:
    return {f"{a}|{t}": n for (a, t), n in dist.items()}


def _cell_rows(rows: list[dict], family: str, cell: tuple) -> list[dict]:
    cond = FACTORIAL_CELLS[cell]
    return [r for r in rows if r["scenario_family"] == family and r["condition"] == cond]


def _pool(rows: list[dict], family: str, cells: list[tuple]) -> list[dict]:
    out = []
    for c in cells:
        out.extend(_cell_rows(rows, family, c))
    return out


def analyze_family(rows: list[dict], family: str) -> dict:
    all_cells = list(FACTORIAL_CELLS.keys())
    cell_dists = {cell: _decision_dist(_cell_rows(rows, family, cell)) for cell in all_cells}
    def _modal_dict(dist: Counter):
        m = _modal(dist)
        return {"action": m[0], "target_id": m[1]} if m else None

    eight_cell_table = {
        FACTORIAL_CELLS[cell]: {
            "cell": list(cell), "samples": sum(cell_dists[cell].values()), "modal_decision": _modal_dict(cell_dists[cell]),
            "action_distribution": dict(Counter(a for a, t in cell_dists[cell].elements())),
            "target_distribution": dict(Counter(t for a, t in cell_dists[cell].elements())),
        }
        for cell in all_cells
    }

    def cells_where(levels: dict) -> list:
        return [c for c in all_cells if all(c[i] == v for i, v in levels.items())]

    def dist_for(levels: dict) -> Counter:
        d = Counter()
        for c in cells_where(levels):
            d.update(cell_dists[c])
        return d

    # Main effects (spec section 24): marginal distributions per level.
    main_effects = {}
    for axis, (lvl0, lvl1) in (("A", (A0_SINGLE_COPY, A1_DOUBLE_COPY)), ("B", (B0_NATURAL_LOCATION, B1_AUXILIARY_LOCATION)), ("C", (C0_FULL_FORMAT, C1_COMPACT_FORMAT))):
        idx = {"A": 0, "B": 1, "C": 2}[axis]
        d0, d1 = dist_for({idx: lvl0}), dist_for({idx: lvl1})
        tv = _tv_distance(d0, d1)
        main_effects[axis] = {
            "level0": lvl0, "level1": lvl1, "dist_level0": _stringify_dist(d0), "dist_level1": _stringify_dist(d1),
            "modal0": _modal(d0), "modal1": _modal(d1), "tv_distance": tv,
            "classification": MATERIAL_EFFECT if (tv >= TV_MATERIAL_THRESHOLD or _modal(d0) != _modal(d1)) else NO_EFFECT,
        }

    # Pairwise interactions (spec section 25): effect of X at each level of Y.
    def effect_tv(fixed_idx, fixed_val, vary_idx, lvl0, lvl1):
        d0 = dist_for({fixed_idx: fixed_val, vary_idx: lvl0})
        d1 = dist_for({fixed_idx: fixed_val, vary_idx: lvl1})
        return _tv_distance(d0, d1), _modal(d0), _modal(d1)

    pairwise = {}
    axis_levels = {"A": (A0_SINGLE_COPY, A1_DOUBLE_COPY), "B": (B0_NATURAL_LOCATION, B1_AUXILIARY_LOCATION), "C": (C0_FULL_FORMAT, C1_COMPACT_FORMAT)}
    axis_idx = {"A": 0, "B": 1, "C": 2}
    for fixed_axis, vary_axis in (("A", "C"), ("B", "C"), ("A", "B")):
        f0, f1 = axis_levels[fixed_axis]
        v0, v1 = axis_levels[vary_axis]
        tv_at_f0, m0a, m0b = effect_tv(axis_idx[fixed_axis], f0, axis_idx[vary_axis], v0, v1)
        tv_at_f1, m1a, m1b = effect_tv(axis_idx[fixed_axis], f1, axis_idx[vary_axis], v0, v1)
        contrast = tv_at_f1 - tv_at_f0
        interaction_present = abs(contrast) >= TV_MATERIAL_THRESHOLD or ((m0a != m0b) != (m1a != m1b))
        pairwise[f"{fixed_axis}x{vary_axis}"] = {
            f"effect_of_{vary_axis}_at_{fixed_axis}0": {"tv": tv_at_f0, "modal_level0": m0a, "modal_level1": m0b},
            f"effect_of_{vary_axis}_at_{fixed_axis}1": {"tv": tv_at_f1, "modal_level0": m1a, "modal_level1": m1b},
            "interaction_contrast": contrast, "classification": "INTERACTION" if interaction_present else "NO_INTERACTION",
        }

    # Three-way interaction (spec section 26): only after pairwise reported -
    # does the A x C interaction contrast itself differ across B0 vs B1?
    def ac_contrast_at(b_level):
        d_a0c0 = cell_dists[(A0_SINGLE_COPY, b_level, C0_FULL_FORMAT)]
        d_a0c1 = cell_dists[(A0_SINGLE_COPY, b_level, C1_COMPACT_FORMAT)]
        d_a1c0 = cell_dists[(A1_DOUBLE_COPY, b_level, C0_FULL_FORMAT)]
        d_a1c1 = cell_dists[(A1_DOUBLE_COPY, b_level, C1_COMPACT_FORMAT)]
        return _tv_distance(d_a1c0, d_a1c1) - _tv_distance(d_a0c0, d_a0c1)

    ac_at_b0, ac_at_b1 = ac_contrast_at(B0_NATURAL_LOCATION), ac_contrast_at(B1_AUXILIARY_LOCATION)
    three_way_contrast = ac_at_b1 - ac_at_b0
    three_way = {
        "ACxB_contrast_at_B0_natural": ac_at_b0, "ACxB_contrast_at_B1_auxiliary": ac_at_b1,
        "three_way_contrast": three_way_contrast,
        "needed": abs(three_way_contrast) >= TV_MATERIAL_THRESHOLD,
        "remaining_ambiguity": "each underlying cell has only n=5 - three-way estimates are the noisiest in this design and reported descriptively, not as a confirmed effect size",
    }

    return {"eight_cell_table": eight_cell_table, "main_effects": main_effects, "pairwise_interactions": pairwise, "three_way_interaction": three_way}


def _runtime_replication(native_rows: list[dict], strands_rows: list[dict]) -> list[dict]:
    out = []
    for family in sorted({r["scenario_family"] for r in strands_rows}):
        for cond in sorted({r["condition"] for r in strands_rows if r["scenario_family"] == family}):
            n_rows = [r for r in native_rows if r["scenario_family"] == family and r["condition"] == cond]
            s_rows = [r for r in strands_rows if r["scenario_family"] == family and r["condition"] == cond]
            if not n_rows or not s_rows:
                continue
            n_dist, s_dist = _decision_dist(n_rows), _decision_dist(s_rows)
            n_modal, s_modal = _modal(n_dist), _modal(s_dist)
            tv = _tv_distance(n_dist, s_dist)
            classification = "INTERACTION_RUNTIME_INDEPENDENT" if n_modal == s_modal else ("PARTIAL_REPLICATION" if tv < 0.7 else "NO_REPLICATION")
            out.append({"scenario": family, "condition": cond, "native_modal": n_modal, "strands_modal": s_modal, "tv_distance": tv, "classification": classification})
    return out


def _cost_metrics(rows: list[dict]) -> dict:
    import statistics

    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        by_cond.setdefault(f"{r['scenario_family']}/{r['condition']}", []).append(r)
    out = {}
    for k, cell_rows in by_cond.items():
        lat = [r["latency_ms"] for r in cell_rows if r.get("latency_ms") is not None]
        it = [r["input_tokens"] for r in cell_rows if r.get("input_tokens") is not None]
        ot = [r["output_tokens"] for r in cell_rows if r.get("output_tokens") is not None]
        ch = [r["characters"] for r in cell_rows if r.get("characters") is not None]
        out[k] = {
            "n": len(cell_rows),
            "latency_ms_mean": statistics.mean(lat) if lat else None,
            "input_tokens_mean": statistics.mean(it) if it else None,
            "output_tokens_mean": statistics.mean(ot) if ot else None,
            "characters_mean": statistics.mean(ch) if ch else None,
        }
    return out


def run() -> dict:
    rows = _read_jsonl(os.path.join(OUT_DIR, "native_trials.jsonl"))
    strands_rows = _read_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"))
    families = sorted({r["scenario_family"] for r in rows})
    results = {family: analyze_family(rows, family) for family in families}

    runtime_replication = _runtime_replication(rows, strands_rows)
    with open(os.path.join(OUT_DIR, "runtime_replication.json"), "w", encoding="utf-8") as f:
        json.dump(runtime_replication, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "cost_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_cost_metrics(rows), f, indent=2, default=str)

    with open(os.path.join(OUT_DIR, "main_effects.json"), "w", encoding="utf-8") as f:
        json.dump({fam: r["main_effects"] for fam, r in results.items()}, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "pairwise_interactions.json"), "w", encoding="utf-8") as f:
        json.dump({fam: r["pairwise_interactions"] for fam, r in results.items()}, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "three_way_interaction.json"), "w", encoding="utf-8") as f:
        json.dump({fam: r["three_way_interaction"] for fam, r in results.items()}, f, indent=2, default=str)

    s2 = results.get("S2_KNOWN_LOCKED_OBJECT", {})
    s2_causal = {
        "eight_cell_table": s2.get("eight_cell_table"),
        "location_still_dominant": s2.get("main_effects", {}).get("B", {}).get("classification") == MATERIAL_EFFECT,
        "format_depends_on_copy_count": s2.get("pairwise_interactions", {}).get("AxC", {}).get("classification") == "INTERACTION",
        "location_alters_format_effect": s2.get("pairwise_interactions", {}).get("BxC", {}).get("classification") == "INTERACTION",
        "three_way_needed": s2.get("three_way_interaction", {}).get("needed"),
    }
    with open(os.path.join(OUT_DIR, "s2_factorial_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(s2_causal, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    out = run()
    for fam, r in out.items():
        print(f"=== {fam} ===")
        for axis, m in r["main_effects"].items():
            print(f"  main[{axis}]: {m['classification']} (tv={m['tv_distance']:.2f}, modal0={m['modal0']}, modal1={m['modal1']})")
        for pair, p in r["pairwise_interactions"].items():
            print(f"  pairwise[{pair}]: {p['classification']} (contrast={p['interaction_contrast']:.2f})")
        print(f"  three_way: needed={r['three_way_interaction']['needed']} contrast={r['three_way_interaction']['three_way_contrast']:.2f}")
