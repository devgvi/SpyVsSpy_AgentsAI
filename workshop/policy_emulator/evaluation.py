"""3D / M2 - LOGGED_STATE_REPLAY: offline policy fidelity, no game step, no provider call.

recorded observation -> features -> surrogate probabilities (masked by the
recorded legal actions) -> compared with the recorded LLM intended action.

Two separate test suites (never pooled):

A. leave-one-seed-out on CLEAN full-match decisions: for each Phase-A seed,
   train on the other seeds and test on the held-out seed (a match has one
   seed, so no match is ever on both sides). Every emulator (B0-B3) uses the
   very same folds, test records and policy identities; `fold_definitions`
   persists them.
B. controlled snapshots (Stage 2): policies trained on every CLEAN full-match
   decision, evaluated on the frozen snapshots against the Stage-2 samples.

Metrics are kept apart on purpose: action agreement, distribution quality,
opportunity-normalized behaviour, calibration, target fidelity, coverage and
controlled-snapshot signatures. There is no overall fidelity score.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Callable

from workshop.policy_emulator import schema
from workshop.policy_emulator.coverage import PolicyCoverageEstimator
from workshop.policy_emulator.policies import TargetPolicy, legal_action_set

EPS = 1e-6
B0, B1, B2, B3 = "frequency-v1", "table-v1", "logistic-v1", "tree-v1"
DEFAULT_EMULATORS = (B0, B1)

# opportunity -> (condition on features, action counted)
OPPORTUNITIES: dict[str, tuple[Callable[[dict], bool], str]] = {
    "P(place_trap | legal)": (lambda f: f["place_trap_legal"], "place_trap"),
    "P(inspect | legal)": (lambda f: f["inspect_legal"], "inspect"),
    "P(search | opportunity)": (lambda f: f["search_legal"], "interact"),
    "P(approach | alert)": (lambda f: f["opponent_alert_present"], "approach_opponent"),
    "P(avoid | alert)": (lambda f: f["opponent_alert_present"], "avoid_opponent"),
}

SIGNATURES = [
    # id, snapshot, personality, Stage-2 category, higher model (Stage 2), lower model, Stage-2 note
    ("A1 opportunist", "encounter_equal_relaxed", "opportunist", "AVOID", "gpt61sol", "sonnet55", "GPT avoids, Sonnet continues"),
    ("A1 saboteur", "encounter_equal_relaxed", "saboteur", "APPROACH", "gpt61sol", "sonnet55", "GPT approaches, Sonnet continues"),
    ("A3 racer", "encounter_self_leads", "racer", "APPROACH", "sonnet55", "gpt61sol", "Sonnet approached 5/10, GPT 0/10"),
    ("A4 racer", "encounter_self_trails", "racer", "AVOID", "sonnet55", "gpt61sol", "Sonnet avoided 10/10, GPT 6/10"),
    ("B1 racer", "inspect_optional_relaxed", "racer", "INSPECT", "sonnet55", "gpt61sol", "Sonnet 1/10, GPT 0/10 (not decisive)"),
]
NO_DIFFERENCE = 0.05


def _build(emulator: str, rows: list[dict], configs: dict[str, dict] | None):
    from workshop.policy_emulator.artifacts import build_action_model
    return build_action_model(emulator, rows, (configs or {}).get(emulator))


# -- replay ---------------------------------------------------------------------------------------

def replay_record(policy, target: TargetPolicy | None, row: dict,
                  coverage: PolicyCoverageEstimator | None = None) -> dict[str, Any]:
    payload = row["observation"]
    legal = legal_action_set(payload)
    pred = policy.predict(row["features"], legal)
    real = row["intended_action"]
    out = {"decision_id": row.get("decision_id"), "policy_key": row["policy_key"], "seed": row.get("seed"),
           "match_id": row.get("match_id"), "real": real, "real_target": row.get("intended_target"),
           "real_category": row.get("category"), "probs": pred.probs, "top": pred.top_action,
           "p_real": pred.probs.get(real, 0.0), "masking_changed_top": pred.masking_changed_top,
           "backoff_level": pred.backoff_level, "sample_count": pred.sample_count, "bucket": pred.matched_bucket,
           "confidence": pred.confidence, "explanation": pred.explanation, "features": row["features"]}
    if coverage is not None:
        out["coverage"] = coverage.assess(row["features"], legal)
    if target is not None and real is not None:
        choice = target.choose(real, payload, row["features"])
        targets = [t for t in schema.legal_targets(payload).get(real, []) if t is not None]
        out["target"] = {"n_targets": len(targets), "p_real_target": choice.distribution.get(row.get("intended_target"), 0.0),
                         "uniform_p": 1.0 / len(targets) if targets else 1.0,
                         "real_class": row.get("target_class"),
                         "top_class": max(choice.target_class_probs, key=choice.target_class_probs.get)
                         if choice.target_class_probs else None,
                         "source": choice.source, "fallback": choice.fallback,
                         "in_legal_set": all(t in targets or t is None for t in choice.distribution)}
    return out


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence, base 2 (0 = identical, 1 = disjoint)."""
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a):
        return sum(a.get(k, 0.0) * math.log2(a.get(k, 0.0) / m[k]) for k in keys if a.get(k, 0.0) > 0)
    return max(0.0, 0.5 * kl(p) + 0.5 * kl(q))


def _metrics(results: list[dict]) -> dict[str, Any]:
    if not results:
        return {"n": 0}
    n = len(results)
    top1 = sum(1 for r in results if r["top"] == r["real"]) / n
    log_loss = -sum(math.log(max(r["p_real"], EPS)) for r in results) / n
    brier = sum(sum((p - (1.0 if a == r["real"] else 0.0)) ** 2 for a, p in r["probs"].items())
                + (0.0 if r["real"] in r["probs"] else 1.0) for r in results) / n
    labels = sorted({r["real"] for r in results} | {r["top"] for r in results})
    f1s = []
    for label in labels:
        tp = sum(1 for r in results if r["top"] == label and r["real"] == label)
        fp = sum(1 for r in results if r["top"] == label and r["real"] != label)
        fn = sum(1 for r in results if r["top"] != label and r["real"] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    real_dist = Counter(r["real"] for r in results)
    mean_pred: dict[str, float] = defaultdict(float)
    for r in results:
        for a, p in r["probs"].items():
            mean_pred[a] += p / n
    return {"n": n, "top1": round(top1, 4), "macro_f1": round(sum(f1s) / len(f1s), 4), "log_loss": round(log_loss, 4),
            "brier": round(brier, 4), "jsd": round(jsd({a: c / n for a, c in real_dist.items()}, mean_pred), 4),
            "masking_changed_top": sum(1 for r in results if r["masking_changed_top"]),
            "backoff_levels": dict(sorted(Counter(r["backoff_level"] for r in results).items()))}


def opportunity_rates(results: list[dict]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, (cond, action) in OPPORTUNITIES.items():
        pool = [r for r in results if cond(r["features"])]
        n = len(pool)
        real = sum(1 for r in pool if r["real"] == action) / n if n else None
        sur = sum(r["probs"].get(action, 0.0) for r in pool) / n if n else None
        out[name] = {"n": n, "real": round(real, 4) if n else None, "surrogate": round(sur, 4) if n else None,
                     "abs_error": round(abs(real - sur), 4) if n else None,
                     "real_count": sum(1 for r in pool if r["real"] == action)}
    return out


def calibration(results: list[dict], bins: int = 10, actions: tuple[str, ...] | None = None,
                where: Callable[[dict], bool] | None = None) -> list[dict]:
    """Per predicted-probability bin: mean predicted vs observed frequency, over every
    (decision, legal action) pair (or only the given actions)."""
    acc = [[0, 0.0, 0] for _ in range(bins)]
    for r in results:
        if where is not None and not where(r):
            continue
        for a, p in r["probs"].items():
            if actions is not None and a not in actions:
                continue
            b = min(int(p * bins), bins - 1)
            acc[b][0] += 1
            acc[b][1] += p
            acc[b][2] += 1 if r["real"] == a else 0
    return [{"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": n, "mean_predicted": round(s / n, 4),
             "observed": round(k / n, 4)} for i, (n, s, k) in enumerate(acc) if n]


def ece(bins: list[dict], middle_only: bool = False) -> float | None:
    """Expected calibration error over the bins (pair-weighted); middle = 0.2-0.8."""
    chosen = [b for b in bins if not middle_only or 0.2 <= float(b["bin"].split("-")[0]) < 0.8]
    total = sum(b["n"] for b in chosen)
    return round(sum(b["n"] * abs(b["mean_predicted"] - b["observed"]) for b in chosen) / total, 4) if total else None


CALIBRATION_VIEWS = {
    "all legal actions": {},
    "AVOID and APPROACH when alerted": {"actions": ("avoid_opponent", "approach_opponent"),
                                        "where": lambda r: r["features"]["opponent_alert_present"]},
    "MOVE and INTERACT (common classes)": {"actions": ("move", "interact")},
}


def target_fidelity(results: list[dict]) -> dict[str, Any]:
    rows = [r for r in results if r.get("target")]
    multi = [r for r in rows if r["target"]["n_targets"] > 1]
    by_action: dict[str, list] = defaultdict(list)
    for r in multi:
        by_action[r["real"]].append(r)

    def block(pool):
        n = len(pool)
        return {"n": n,
                "class_top1": round(sum(1 for r in pool if r["target"]["top_class"] == r["target"]["real_class"]) / n, 4) if n else None,
                "mean_p_real_target": round(sum(r["target"]["p_real_target"] for r in pool) / n, 4) if n else None,
                "uniform_baseline": round(sum(r["target"]["uniform_p"] for r in pool) / n, 4) if n else None,
                "fallbacks": sum(1 for r in pool if r["target"]["fallback"]),
                "sources": dict(Counter(r["target"]["source"].split(":")[0] + (":bucket" if "|" in r["target"]["source"] else "")
                                        for r in pool))}
    return {"single_target_decisions": len(rows) - len(multi), "multi_target": block(multi),
            "by_action": {a: block(v) for a, v in sorted(by_action.items())},
            "all_in_legal_set": all(r["target"]["in_legal_set"] for r in rows),
            "real_class_counts": dict(Counter(r["target"]["real_class"] for r in multi).most_common())}


# -- A. leave-one-seed-out ---------------------------------------------------------------------

def fold_definitions(clean: list[dict]) -> list[dict[str, Any]]:
    """The leave-one-seed-out folds, persisted so every emulator uses exactly these."""
    folds = []
    for seed in sorted({r["seed"] for r in clean}):
        test = sorted(r["decision_id"] for r in clean if r["seed"] == seed)
        train_matches = {r["match_id"] for r in clean if r["seed"] != seed}
        test_matches = {r["match_id"] for r in clean if r["seed"] == seed}
        folds.append({"held_out_seed": seed, "test_decisions": len(test), "train_decisions": len(clean) - len(test),
                      "test_decision_ids_sha256": hashlib.sha256("\n".join(test).encode()).hexdigest()[:16],
                      "test_decision_ids": test, "test_matches": sorted(test_matches),
                      "match_leakage": sorted(train_matches & test_matches)})
    return folds


def leave_one_seed_out(clean: list[dict], policy_keys: list[str], *, emulators: tuple[str, ...] = DEFAULT_EMULATORS,
                       configs: dict[str, dict] | None = None, targets: bool = True, coverage: bool = True
                       ) -> dict[str, Any]:
    from workshop.policy_emulator.targets import TargetPolicyV2

    folds_def = fold_definitions(clean)
    by_id = {r["decision_id"]: r for r in clean}
    folds, results = [], {e: [] for e in emulators}
    target_recs: dict[str, list[dict]] = {"target-v1": [], "target-v2": []}
    for fold in folds_def:
        seed = fold["held_out_seed"]
        test_ids = set(fold["test_decision_ids"])
        test = [by_id[i] for i in fold["test_decision_ids"]]
        train = [r for r in clean if r["decision_id"] not in test_ids]
        entry = {"held_out_seed": seed, "train": len(train), "test": len(test), "match_leakage": fold["match_leakage"],
                 "train_matches": len({r["match_id"] for r in train}), "test_matches": len(fold["test_matches"]),
                 "test_decision_ids_sha256": fold["test_decision_ids_sha256"], "per_emulator": {}}
        per_policy = {}
        for key in policy_keys:
            mine_train = [r for r in train if r["policy_key"] == key]
            mine_test = [r for r in test if r["policy_key"] == key]
            if not mine_train or not mine_test:
                continue
            tv1 = TargetPolicy().fit(mine_train, train) if targets else None
            cov = PolicyCoverageEstimator().fit(mine_train) if coverage else None
            per_policy[key] = (mine_train, mine_test, tv1, cov)
            if targets:
                tv2 = TargetPolicyV2().fit(mine_train, train)
                for version, tp in (("target-v1", tv1), ("target-v2", tv2)):
                    target_recs[version].extend(_target_records(tp, mine_test))
        for emulator in emulators:
            fold_results = []
            for key, (mine_train, mine_test, tv1, cov) in per_policy.items():
                action_policy = _build(emulator, mine_train, configs)
                fold_results.extend(replay_record(action_policy, tv1 if emulator == B1 else None, r, cov) for r in mine_test)
            entry["per_emulator"][emulator] = {"overall": _metrics(fold_results),
                                               "per_policy": {k: _metrics([r for r in fold_results if r["policy_key"] == k])
                                                              for k in policy_keys}}
            results[emulator].extend(fold_results)
        folds.append(entry)
    seeds = [f["held_out_seed"] for f in folds_def]
    summary: dict[str, Any] = {}
    for emulator, res in results.items():
        summary[emulator] = {
            "overall": _metrics(res),
            "per_policy": {k: _metrics([r for r in res if r["policy_key"] == k]) for k in policy_keys},
            "opportunity": opportunity_rates(res),
            "opportunity_per_policy": {k: opportunity_rates([r for r in res if r["policy_key"] == k]) for k in policy_keys},
            "opportunity_per_seed": {s: opportunity_rates([r for r in res if r["seed"] == s]) for s in seeds},
            "jsd_by_bucket": {
                "alert": _metrics([r for r in res if r["features"]["opponent_alert_present"]]).get("jsd"),
                "no alert": _metrics([r for r in res if not r["features"]["opponent_alert_present"]]).get("jsd"),
                "inspect legal": _metrics([r for r in res if r["features"]["inspect_legal"]]).get("jsd"),
                "place_trap legal": _metrics([r for r in res if r["features"]["place_trap_legal"]]).get("jsd"),
                "mission complete": _metrics([r for r in res if r["features"]["mission_complete"]]).get("jsd"),
            },
            "jsd_policy_seed": {k: {s: _metrics([r for r in res if r["policy_key"] == k and r["seed"] == s]).get("jsd")
                                    for s in seeds} for k in policy_keys},
            "calibration": {name: calibration(res, **view) for name, view in CALIBRATION_VIEWS.items()},
            "by_coverage": _by_coverage(res) if coverage else {},
        }
        summary[emulator]["ece"] = {name: {"all bins": ece(b), "middle bins (0.2-0.8)": ece(b, True)}
                                    for name, b in summary[emulator]["calibration"].items()}
    if B1 in results:
        b1 = results[B1]
        summary["calibration"] = summary[B1]["calibration"]
        summary["target_fidelity"] = target_fidelity(b1)
        summary["target_fidelity_per_policy"] = {k: target_fidelity([r for r in b1 if r["policy_key"] == k])["multi_target"]
                                                 for k in policy_keys}
    if targets:
        summary["target_comparison"] = {v: _target_blocks(recs) for v, recs in target_recs.items()}
    summary["coverage_distribution"] = dict(Counter(r["coverage"]["coverage"] for r in next(iter(results.values()))
                                                    if r.get("coverage"))) if coverage and results else {}
    return {"seeds": seeds, "folds": folds, "fold_definitions": folds_def, "summary": summary, "_results": results}


def _target_records(policy, rows: list[dict]) -> list[dict]:
    recs = []
    for r in rows:
        action, real = r["intended_action"], r.get("intended_target")
        if action is None:
            continue
        payload = r["observation"]
        legal = [t for t in schema.legal_targets(payload).get(action, []) if t is not None]
        choice = policy.choose(action, payload, r["features"])
        dist = choice.distribution
        top = max(dist, key=lambda t: (dist[t], -legal.index(t) if t in legal else 0)) if dist else None
        top_class = max(choice.target_class_probs, key=choice.target_class_probs.get) if choice.target_class_probs else None
        recs.append({"policy_key": r["policy_key"], "action": action, "real_class": r.get("target_class"),
                     "n_targets": len(legal), "p_real": dist.get(real, 0.0), "exact": top == real,
                     "class_top": top_class == r.get("target_class"), "uniform": 1.0 / len(legal) if legal else 1.0,
                     "legal": all(t in legal or t is None for t in dist), "decision_id": r["decision_id"],
                     "real_target": real, "top_target": top, "seed": r["seed"]})
    return recs


def _target_block(recs: list[dict]) -> dict[str, Any]:
    n = len(recs)
    if not n:
        return {"n": 0}
    return {"n": n, "class_top1": round(sum(x["class_top"] for x in recs) / n, 4),
            "exact_top1": round(sum(x["exact"] for x in recs) / n, 4),
            "mean_p_real": round(sum(x["p_real"] for x in recs) / n, 4),
            "log_loss": round(-sum(math.log(max(x["p_real"], 1e-6)) for x in recs) / n, 4),
            "uniform_baseline": round(sum(x["uniform"] for x in recs) / n, 4),
            "legal_target_rate": round(sum(x["legal"] for x in recs) / n, 4)}


def _target_blocks(recs: list[dict]) -> dict[str, Any]:
    multi = [x for x in recs if x["n_targets"] > 1]
    out = {"all decisions": _target_block(recs), "with a choice of target": _target_block(multi),
           "single legal target": sum(1 for x in recs if x["n_targets"] <= 1)}
    out["by_action"] = {a: {**_target_block([x for x in recs if x["action"] == a]),
                            "with_choice": sum(1 for x in recs if x["action"] == a and x["n_targets"] > 1)}
                        for a in sorted({x["action"] for x in recs})}
    out["by_class"] = {c: _target_block([x for x in multi if x["real_class"] == c])
                       for c in sorted({str(x["real_class"]) for x in multi})}
    out["by_policy"] = {k: _target_block([x for x in multi if x["policy_key"] == k])
                        for k in sorted({x["policy_key"] for x in multi})}
    out["_records"] = recs
    return out


def _by_coverage(results: list[dict]) -> dict[str, Any]:
    out = {}
    for label in ("HIGH", "MEDIUM", "LOW", "OOD"):
        pool = [r for r in results if r.get("coverage", {}).get("coverage") == label]
        out[label] = {"n": len(pool), **({k: v for k, v in _metrics(pool).items() if k in ("top1", "log_loss")} if pool else {}),
                      "mean_confidence": round(sum(r["confidence"] for r in pool) / len(pool), 4) if pool else None}
    bands = (("confidence >= 0.8", lambda c: c >= 0.8), ("0.5 <= confidence < 0.8", lambda c: 0.5 <= c < 0.8),
             ("confidence < 0.5", lambda c: c < 0.5))
    out["quadrants"] = {f"{name} | coverage {label}": _quad([r for r in results if test(r["confidence"])
                                                            and r.get("coverage", {}).get("coverage") == label])
                        for name, test in bands for label in ("HIGH", "MEDIUM", "LOW", "OOD")}
    return out


def _quad(pool: list[dict]) -> dict[str, Any]:
    return {"n": len(pool), "top1": round(sum(1 for r in pool if r["top"] == r["real"]) / len(pool), 4) if pool else None}


def grid_search(clean: list[dict], policy_keys: list[str], emulator: str, grid: list[dict]) -> list[dict[str, Any]]:
    """Every configuration on the same leave-one-seed-out folds (no target / coverage)."""
    out = []
    for cfg in grid:
        res = leave_one_seed_out(clean, policy_keys, emulators=(emulator,), configs={emulator: cfg}, targets=False,
                                 coverage=False)
        s = res["summary"][emulator]
        per_seed = {f["held_out_seed"]: f["per_emulator"][emulator]["overall"]["log_loss"] for f in res["folds"]}
        out.append({"config": cfg, "overall": s["overall"], "per_seed_log_loss": per_seed,
                    "opportunity_mae": round(sum(v["abs_error"] for v in s["opportunity"].values()) / len(s["opportunity"]), 4)})
    return out


def failure_cases(results: list[dict], limit: int = 8, others: dict[str, list[dict]] | None = None) -> dict[str, list[dict]]:
    def brief(r):
        f = r["features"]
        top3 = sorted(r["probs"].items(), key=lambda x: -x[1])[:3]
        cov = r.get("coverage") or {}
        return {"policy_key": r["policy_key"], "seed": r["seed"], "match_id": r["match_id"],
                "decision_id": r["decision_id"], "real": r["real"], "real_target": r["real_target"],
                "surrogate_top": r["top"], "p_real": float(f"{r['p_real']:.3g}"),
                "top3": [(a, float(f"{p:.3g}")) for a, p in top3], "backoff_level": r["backoff_level"],
                "sample_count": r["sample_count"], "confidence": round(r["confidence"], 3),
                "coverage": cov.get("coverage"), "coverage_reasons": cov.get("reasons", [])[:2],
                "explanation": r.get("explanation"),
                "state": {k: f[k] for k in ("current_room", "time_remaining_seconds", "hearts", "opponent_alert_present",
                                            "opponent_distance_bucket", "life_balance", "near_object_type", "inspect_legal",
                                            "search_legal", "place_trap_legal", "mission_complete", "recent_action_1",
                                            "room_unsearched_count", "objectives_remaining_count")},
                "target": r.get("target")}
    wrong = sorted((r for r in results if r["top"] != r["real"]), key=lambda r: (-r["confidence"], r["decision_id"]))
    low = sorted(results, key=lambda r: (r["confidence"], r["decision_id"]))
    masked = [r for r in results if r["masking_changed_top"]]
    target_fail = sorted((r for r in results if r.get("target") and r["target"]["n_targets"] > 1
                          and r["target"]["top_class"] != r["target"]["real_class"]),
                         key=lambda r: (r["target"]["p_real_target"], r["decision_id"]))
    rare = sorted((r for r in results if r["real"] in ("place_trap", "disarm", "wait", "approach_opponent")),
                  key=lambda r: (r["p_real"], r["decision_id"]))
    extrapolating = sorted((r for r in results if (r.get("coverage") or {}).get("coverage") in ("LOW", "OOD")
                            and r["top"] != r["real"]), key=lambda r: (-r["confidence"], r["decision_id"]))
    out = {"rare action underweighted": [brief(r) for r in rare[:limit]],
           "confident wrong prediction": [brief(r) for r in wrong[:limit]],
           "low coverage / extrapolation": [brief(r) for r in extrapolating[:limit]],
           "legal mask changes top action": [brief(r) for r in masked[:limit]],
           "low confidence": [brief(r) for r in low[:limit]]}
    if any(r.get("target") for r in results):
        out["target failure"] = [brief(r) for r in target_fail[:limit]]
    if others:
        index = {name: {r["decision_id"]: r for r in res} for name, res in others.items()}
        disagree = []
        for r in results:
            tops = {name: idx[r["decision_id"]]["top"] for name, idx in index.items() if r["decision_id"] in idx}
            if len(set(tops.values()) | {r["top"]}) > 1:
                disagree.append((r, tops))
        disagree.sort(key=lambda x: (-max(x[0]["confidence"], *(index[n][x[0]["decision_id"]]["confidence"]
                                                                 for n in x[1])), x[0]["decision_id"]))
        out["model disagreement"] = [{**brief(r), "tops": tops} for r, tops in disagree[:limit]]
    return out


# -- B. controlled snapshots -------------------------------------------------------------------

def category_distribution(policy, target, row: dict, snapshot: dict) -> tuple[dict[str, float], dict]:
    """Surrogate probability of each Stage-2 category in this snapshot (action x target)."""
    from workshop.scripts.behavior_grid_snapshots import CATEGORIES, categorize

    payload = row["observation"]
    pred = policy.predict(row["features"], legal_action_set(payload))
    dist = {c: 0.0 for c in CATEGORIES[snapshot["family"]]}
    for action, p in pred.probs.items():
        choice = target.choose(action, payload, row["features"])
        for t, q in choice.distribution.items():
            dist[categorize(snapshot, action, t)] = dist.get(categorize(snapshot, action, t), 0.0) + p * q
    return dist, pred.as_dict()


def controlled_validation(controlled: list[dict], trained: dict[str, tuple], snapshots: dict[str, dict],
                          coverage: dict[str, PolicyCoverageEstimator] | None = None) -> dict[str, Any]:
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in controlled:
        by_cell[(r["snapshot_id"], r["personality"], r["model"])].append(r)
    cells = {}
    for (snapshot_id, personality, model), rows in sorted(by_cell.items()):
        key = schema.policy_key(model, personality)
        if key not in trained:
            continue
        clean = [r for r in rows if r["clean"]]
        real = Counter(r["stage2_category"] for r in clean)
        n = len(clean)
        surrogate, pred = category_distribution(*trained[key][:2], rows[0], snapshots[snapshot_id])
        real_dist = {c: real.get(c, 0) / n for c in surrogate} if n else {}
        cov = coverage[key].assess(rows[0]["features"], legal_action_set(rows[0]["observation"])) \
            if coverage and key in coverage else None
        cells[f"{snapshot_id}|{personality}|{model}"] = {
            "snapshot_id": snapshot_id, "code": snapshots[snapshot_id]["code"], "personality": personality, "model": model,
            "n_clean": n, "real": {c: round(v, 4) for c, v in real_dist.items()},
            "surrogate": {c: round(v, 4) for c, v in surrogate.items()},
            "tvd": round(0.5 * sum(abs(real_dist.get(c, 0) - surrogate.get(c, 0)) for c in surrogate), 4) if n else None,
            "prediction": pred, "coverage": cov}
    signatures = []
    for sig_id, snapshot_id, personality, cat, hi, lo, note in SIGNATURES:
        a, b = cells.get(f"{snapshot_id}|{personality}|{hi}"), cells.get(f"{snapshot_id}|{personality}|{lo}")
        if not a or not b:
            continue
        d_real = round(a["real"].get(cat, 0) - b["real"].get(cat, 0), 4)
        d_sur = round(a["surrogate"].get(cat, 0) - b["surrogate"].get(cat, 0), 4)
        if abs(d_sur) < NO_DIFFERENCE:
            verdict = "NO_DIFFERENCE"
        else:
            verdict = "SAME_DIRECTION" if (d_sur > 0) == (d_real > 0) else "OPPOSITE_DIRECTION"
        signatures.append({"id": sig_id, "snapshot_id": snapshot_id, "personality": personality, "category": cat,
                           "higher": hi, "lower": lo, "stage2_note": note,
                           "real": {hi: a["real"].get(cat), lo: b["real"].get(cat), "n": [a["n_clean"], b["n_clean"]]},
                           "surrogate": {hi: a["surrogate"].get(cat), lo: b["surrogate"].get(cat)},
                           "d_real": d_real, "d_surrogate": d_sur, "verdict": verdict,
                           "coverage": {hi: (a["coverage"] or {}).get("coverage"), lo: (b["coverage"] or {}).get("coverage")},
                           "buckets": {hi: (a["prediction"]["matched_bucket"], a["prediction"]["backoff_level"],
                                            a["prediction"]["sample_count"]),
                                       lo: (b["prediction"]["matched_bucket"], b["prediction"]["backoff_level"],
                                            b["prediction"]["sample_count"])}})
    tvds = [c["tvd"] for c in cells.values() if c["tvd"] is not None]
    return {"cells": cells, "signatures": signatures,
            "mean_tvd": round(sum(tvds) / len(tvds), 4) if tvds else None,
            "cells_tvd_over_0_5": sum(1 for t in tvds if t > 0.5),
            "coverage_distribution": dict(Counter((c["coverage"] or {}).get("coverage") for c in cells.values()))}


def train_all(clean: list[dict], policy_keys: list[str], exclude_seed: int | None = None, *,
              emulator: str = B1, configs: dict[str, dict] | None = None, target: str = "target-v1") -> dict[str, tuple]:
    from workshop.policy_emulator.artifacts import build_target

    pool = [r for r in clean if exclude_seed is None or r["seed"] != exclude_seed]
    out = {}
    for k in policy_keys:
        mine = [r for r in pool if r["policy_key"] == k]
        out[k] = (_build(emulator, mine, configs), build_target(target, mine, pool), PolicyCoverageEstimator().fit(mine))
    return out
