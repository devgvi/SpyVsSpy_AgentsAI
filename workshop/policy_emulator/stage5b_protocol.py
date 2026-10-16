"""Stage 5B - call-free closed-loop surrogate selection: protocol, seeds, plan and the match process
(Archive/STAGE5B_CALL_FREE_CLOSED_LOOP_SURROGATE_SELECTION.md sections 2-7, 21-25, 32).

Freeze order (each artifact hashed and written once):
    workshop/policies/stage5/stage5b_artifact_verification.json  every frozen artifact re-verified (spec 3; STOP on drift)
    workshop/policies/stage5/stage5b_seed_selection.json         6 fresh structural seeds + the Stage-5C reservation
    workshop/policies/stage5/stage5b_selection_protocol.json     metrics, references, tolerances, decision rule
    workshop_runs/stage5b-closed-loop/                           864 call-free matches (B2-v1 / B2-v2 through R0)
    workshop/policies/stage5/stage5b_decision.json               the decision, frozen before the protected regression
    workshop_runs/stage5b-protected-regression/                  Stage-4C configurations, regression-only

Both arms run through the same plumbing: the frozen M4D match process (confirmatory_trial.run_one) with the frozen
HybridController and router R0 (never escalates, the delegate is a fake legal client that is never asked). The only
difference between the arms is the action model handed to the controller (`arm_base`): the frozen B2-v1 artifact or
the frozen B2-v2 candidate; B1, target-v2.1, coverage-v1, the engine and R0 are the frozen artifacts in both. The
frozen R2.1 router runs in shadow on the very same router inputs (`install_shadow`): its result is logged on every
hybrid_decision event and never returned to the controller, so it cannot change the executed action.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from workshop.policy_emulator import stage5a_promotion as pr
from workshop.policy_emulator.artifacts import content_hash
from workshop.runners.common import PROJECT_ROOT, WorkshopError

EXPERIMENT_ID = "stage5b-call-free-closed-loop-v1"
SESSION = "stage5b-closed-loop"
REGRESSION_SESSION = "stage5b-protected-regression"
SMOKE_SESSION = "stage5b-smoke"
STAGE5_DIR = pr.STAGE5_DIR
VERIFICATION_PATH = STAGE5_DIR / "stage5b_artifact_verification.json"
SEED_SELECTION_PATH = STAGE5_DIR / "stage5b_seed_selection.json"
PROTOCOL_PATH = STAGE5_DIR / "stage5b_selection_protocol.json"
DECISION_PATH = STAGE5_DIR / "stage5b_decision.json"
REGRESSION_PATH = STAGE5_DIR / "stage5b_protected_regression.json"
LEVELS = ("level_01", "level_02", "level_03")
PERSONALITIES = ("cautious", "opportunist", "racer", "saboteur")
ORIENTATIONS = ("A", "B")
REPLICATES = (1, 2, 3)
SEEDS_NEEDED = 6
SEED_RANGE = (9013, 9099)              # Stage-5B candidates, ascending
STAGE5C_RESERVED = (9200, 9499)        # untouched: never validated, played or inspected in Stage 5B
ORDER_SEED = 20261008                  # the seeded B2-v1 / B2-v2 launch order within each pair
PARALLEL = 8
MISSION_LIMIT_S = 300.0
CALLS_PER_ESCALATION = 1.009           # Stage-4C provider calls per escalation (spec 21), descriptive only
MATERIAL = {"relative": 0.10, "absolute": 0.01}
PHASE_A_HASH = "sha256:90a0f39349ab509a"
B2V2_MANIFEST_HASH = "sha256:42e007a6a260fe5d"
B2V2_DATA_HASH = "sha256:f284935d6796a9a6"
R21_HASH = "sha256:216f3c9a3c4b3f78"
R0_HASH = "sha256:81fcece16925c290"
STAGE4_MANIFEST_HASH = "sha256:99188f109e84e777"
SHADOW_SIGNALS = ("repetition_loop", "strong_b1_b2_disagreement")
CANDIDATES = {
    "b2v1": {"label": "B2-v1", "artifact": "workshop/policies/phase-a-v2/<model>/<personality>/logistic-v1",
             "training_dataset_hash": PHASE_A_HASH, "description": "the frozen Stage-3 B2 (Phase-A CLEAN)"},
    "b2v2": {"label": "B2-v2 (C1)", "artifact": "workshop/policies/stage5/b2-v2-candidate/<model>/<personality>/logistic-v1",
             "training_dataset_hash": B2V2_DATA_HASH, "manifest_hash": B2V2_MANIFEST_HASH,
             "description": "the frozen Stage-5A candidate C1 (Phase-A CLEAN + promoted REAL hard examples, unweighted)"},
}
# seeds of earlier milestones outside Stage 3 / 4 / 5A that sit near the candidate range (never reused)
OTHER_MILESTONE_SEEDS = {9187: "src/evaluation/run_m5f3b1.py", 9500: "run_m6a2.py", 9501: "run_m6a2.py",
                         9502: "run_m6a2.py", 9503: "run_m6a2.py", 9601: "run_m6b1.py", 9602: "run_m6b1.py",
                         9603: "run_m6b1.py", 9604: "run_m6b1.py", 9605: "run_m6b1.py",
                         9801: "run_m6b2_episode_confirmation.py", 9802: "run_m6b2_episode_confirmation.py"}


def _now() -> str:
    return pr._now()


def _sha(path: Path) -> str:
    return pr._sha(path)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _freeze(path: Path, body: dict[str, Any], key: str) -> dict[str, Any]:
    return pr._freeze(path, body, key)


def _load_frozen(path: Path, key: str) -> dict[str, Any]:
    return pr._load_frozen(path, key)


# -- spec 3: frozen artifacts -------------------------------------------------------------------------------------

def verify_artifacts() -> dict[str, Any]:
    """Every frozen artifact the comparison depends on, recomputed from disk. Byte hashes only: no content of any
    protected Stage-4C file is analysed here."""
    from workshop.policy_emulator import router as rt
    from workshop.policy_emulator import schema
    from workshop.policy_emulator import stage3_release as rel
    from workshop.policy_emulator import stage4_interface as si
    from workshop.policy_emulator import stage4_release as r4
    from workshop.policy_emulator import stage4c_protocol as cp
    from workshop.policy_emulator import stage5a_training as tr
    from workshop.policy_emulator import trajectory_v2_candidate as tv
    s3 = _json(PROJECT_ROOT / "workshop/policies/phase-a-v2/stage3-release/freeze_manifest.json")
    s4 = _json(r4.FREEZE_MANIFEST_PATH)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, ok: bool, detail: Any) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    p3 = rel.verify_manifest(s3)
    check("stage3_release_manifest", not p3, {"problems": p3, "manifest_hash": s3["manifest_hash"]})
    p4 = r4.verify_manifest()
    check("stage4_release_manifest", not p4 and s4["manifest_hash"] == STAGE4_MANIFEST_HASH,
          {"problems": p4, "manifest_hash": s4["manifest_hash"]})
    for em, name in (("table-v1", "B1"), ("logistic-v1", "B2-v1"), ("target-v2.1", "target-v2.1"), ("coverage-v1", "coverage-v1")):
        bad = []
        for key in schema.all_policy_keys():
            model, personality = schema.parse_policy_key(key)
            base = PROJECT_ROOT / "workshop/policies/phase-a-v2" / model / personality / em
            ref = s3["policy_artifacts"][key][em]["files"]
            bad += [f"{key}/{em}/{n}" for n in ("model.json", "metadata.json") if _sha(base / n) != ref[n]]
        check(name, not bad, {"identities": len(schema.all_policy_keys()), "mismatched": bad})
    fz = tr.verify_freeze()
    check("B2-v2 (C1) manifest", fz["ok"] and fz["manifest_hash"] == B2V2_MANIFEST_HASH and fz["candidate"] == "C1", fz)
    r21 = rt.protocol_for("R2.1")["router_hash"]
    check("R2.1", r21 == R21_HASH == s4["router_r21"]["router_hash"], {"router_hash": r21})
    r0 = rt.protocol_for("R0")["router_hash"]
    check("R0 (execution router, hybrid-router-v1)", r0 == R0_HASH == s3["router_v1"]["router_hash"], {"router_hash": r0})
    groups = si.semantic_groups()
    for g in ("feature schema", "observation schema", "action schema", "personality prompts", "B1 (table-v1)",
              "B2 (logistic-v1)", "target-v2.1", "coverage-v1", "router-v2 / R2.1", "mission-time semantics",
              "scoring rules", "lock / key / safe mechanics", "model IDs"):
        check(f"semantic group: {g}", groups[g]["hash"] == s4["semantic_groups"][g],
              {"now": groups[g]["hash"], "stage4_release": s4["semantic_groups"][g]})
    l1 = _json(PROJECT_ROOT / "workshop/policies/stage4/levels/level_01.json")
    check("Level 1", _sha(PROJECT_ROOT / "data/level_01.json") == "sha256:" + l1["level_file_hash"].split(":")[-1],
          {"file": "data/level_01.json", "provenance": "workshop/policies/stage4/levels/level_01.json"})
    for lv in ("level_02", "level_03"):
        ref = s4["levels"][lv]
        check(f"Level {lv[-1]}", _sha(PROJECT_ROOT / ref["level_file"]) == ref["level_file_sha256"],
              {"file": ref["level_file"], "stage4_release": ref["level_file_sha256"]})
    man = tv.load_manifest()
    v2_ok = content_hash({k: v for k, v in man.items() if k not in ("manifest_hash", "frozen_utc")}) == man["manifest_hash"]
    check("trajectory-v2 manifest", v2_ok and man["manifest_hash"] == s4["metrics"]["trajectory_v2"]["manifest_hash"]
          and man["source_code"]["sha256"] == tv.source_hash() == s4["metrics"]["trajectory_v2"]["source_sha256"],
          {"manifest_hash": man["manifest_hash"], "source_sha256": tv.source_hash()})
    check("historical trajectory metric", cp.historical_metric_hash() == s4["metrics"]["historical_trajectory"]["hash"],
          {"hash": cp.historical_metric_hash()})
    check("state-visitation metric", cp.state_metric_hash() == s4["metrics"]["state_visitation"]["hash"],
          {"hash": cp.state_metric_hash()})
    s5 = {"promotion_policy": pr.load_policy()["policy_hash"], "split": pr.load_split()["split_hash"],
          "data": pr.load_data_manifest()["data_hash"], "selection_protocol_5a": tr.load_protocol()["protocol_hash"]}
    check("Stage-5A frozen artifacts (TRAIN / DEV / REGRESSION sets unmodified)",
          s5 == {"promotion_policy": "sha256:8504fba41f9e2914", "split": "sha256:014fb58caeae5034",
                 "data": B2V2_DATA_HASH, "selection_protocol_5a": "sha256:de48900d83464e4e"}, s5)
    ok = all(c["ok"] for c in checks.values())
    return {"version": "stage5b-artifact-verification-v1", "ok": ok, "checks": checks,
            "failed": [k for k, v in checks.items() if not v["ok"]]}


def write_verification() -> dict[str, Any]:
    v = verify_artifacts()
    if not v["ok"]:
        raise WorkshopError(f"STOP: frozen artifacts differ: {v['failed']}")
    return _freeze(VERIFICATION_PATH, v, "verification_hash")


def load_verification() -> dict[str, Any]:
    return _load_frozen(VERIFICATION_PATH, "verification_hash")


# -- spec 5 / 32: fresh seeds ---------------------------------------------------------------------------------------

def used_seeds() -> dict[int, str]:
    """Every seed of Stage 3, Stage 4 and Stage 5A (and nearby earlier milestones) with where it was used."""
    from workshop.policy_emulator import stage4c_protocol as cp
    out: dict[int, str] = {s: "Stage 3 / Stage 4A / Stage 4B (stage4c_protocol.used_seeds chain)" for s in cp.used_seeds()}
    for v in cp.load_seed_selection()["levels"].values():
        for s in v["accepted"]:
            out[s] = "Stage 4C (confirmatory)"
    split = pr.load_split()
    for role in ("train", "development", "regression_only"):
        for src, seeds in split[role].items():
            for s in seeds:
                out.setdefault(s, f"Stage 5A {role} ({src})")
    for s, where in OTHER_MILESTONE_SEEDS.items():
        out.setdefault(s, f"earlier milestone ({where})")
    return dict(sorted(out.items()))


def scan_sessions(seeds: list[int]) -> dict[str, Any]:
    """Every recorded match (run.json under workshop_runs/) whose seed is one of `seeds`: must be none."""
    hits, files = [], 0
    for p in sorted((PROJECT_ROOT / "workshop_runs").rglob("run.json")):
        files += 1
        try:
            d = _json(p)
        except ValueError:
            continue
        seed = (d.get("entry") or {}).get("seed")
        if seed is None and isinstance(d.get("configuration"), dict):
            seed = d["configuration"].get("seed")
        if seed in seeds:
            hits.append(str(p.relative_to(PROJECT_ROOT)))
    return {"run_files_scanned": files, "matches_with_a_selected_seed": hits}


def level_id(level: str) -> str:
    """The id inside the level file stage4_level_select loads for `level` (Level 1 is the original "embassy")."""
    from workshop.policy_emulator import stage4_level_select as ls
    return _json(PROJECT_ROOT / ls.LEVELS[level])["id"]


def select_seeds() -> dict[str, Any]:
    """Ascending from 9013: a seed is accepted only if the frozen nine-check structural validator passes on all
    three levels. No policy, controller or outcome is involved; the world is built and never stepped."""
    from workshop.policy_emulator import fresh_seeds as fs
    from workshop.policy_emulator import stage4_protocol as sp
    used = used_seeds()
    log, accepted, order = [], [], 0
    for seed in range(SEED_RANGE[0], SEED_RANGE[1] + 1):
        if len(accepted) == SEEDS_NEEDED:
            break
        if seed in used:
            log.append({"seed": seed, "skipped": f"used: {used[seed]}"})
            continue
        order += 1
        levels = {}
        for lv in LEVELS:
            r = sp.validate_seed(lv, seed)
            levels[lv] = {"ok": r["ok"], "level_id_loaded": r["level_id_loaded"],
                          "checks": {k: v["ok"] for k, v in r["checks"].items()},
                          "failed": [k for k, v in r["checks"].items() if not v["ok"]]}
        ok = all(v["ok"] and v["level_id_loaded"] == level_id(lv) for lv, v in levels.items())
        reason = None if ok else "; ".join(f"{lv}: {', '.join(v['failed']) or 'wrong level loaded'}"
                                           for lv, v in levels.items() if not v["ok"] or v["level_id_loaded"] != level_id(lv))
        log.append({"order": order, "seed": seed, "ok": ok, "decision": "accepted" if ok else "rejected", "reason": reason,
                    "level_compatibility": {lv: v["ok"] for lv, v in levels.items()}, "levels": levels})
        if ok:
            accepted.append(seed)
    if len(accepted) < SEEDS_NEEDED:
        raise WorkshopError(f"STOP: only {len(accepted)} structurally valid fresh seeds in {SEED_RANGE}")
    if any(STAGE5C_RESERVED[0] <= s <= STAGE5C_RESERVED[1] for s in accepted):
        raise WorkshopError("STOP: a Stage-5B seed falls in the Stage-5C reservation")
    return {"version": "stage5b-seeds-v1", "experiment": EXPERIMENT_ID, "candidate_range": list(SEED_RANGE),
            "needed": SEEDS_NEEDED, "levels": list(LEVELS), "order": "ascending; accepted only if valid on every level",
            "validator": {"function": "stage4_protocol.validate_seed(level, seed) -> fresh_seeds.validate_seed",
                          "checks": list(fs.CHECKS), "conditions": "the Phase-A template conditions (stage4_protocol.template_conditions)"},
            "forbidden_criteria": list(fs.FORBIDDEN), "excluded_seeds": {str(k): v for k, v in used.items()},
            "accepted": accepted, "rejected": {str(r["seed"]): r["reason"] for r in log if r.get("decision") == "rejected"},
            "candidates_checked": order, "log": log, "session_scan": scan_sessions(accepted),
            "registry": {"stage5b_call_free_development": accepted,
                         "stage5c_reserved": {"range": list(STAGE5C_RESERVED),
                                              "rule": "every seed in the range not used by an earlier milestone; selected "
                                                      "only by the Stage-5C preregistration",
                                              "validated_in_stage5b": False, "played_in_stage5b": False,
                                              "inspected_in_stage5b": False},
                         "stage5a_train_dev_regression": {r: pr.load_split()[r] for r in ("train", "development", "regression_only")}}}


def write_seed_selection() -> dict[str, Any]:
    load_verification()
    sel = select_seeds()
    if sel["session_scan"]["matches_with_a_selected_seed"]:
        raise WorkshopError(f"STOP: a selected seed was already played: {sel['session_scan']['matches_with_a_selected_seed'][:3]}")
    return _freeze(SEED_SELECTION_PATH, sel, "selection_hash")


def load_seed_selection() -> dict[str, Any]:
    return _load_frozen(SEED_SELECTION_PATH, "selection_hash")


# -- spec 23: tolerances from historical B2-v1 replicate variability -------------------------------------------------

GUARDS = {  # metric: (direction "higher is worse" / "lower is worse", floor kind)
    "timeout_rate": "higher is worse", "mission_completion_rate": "lower is worse", "escape_rate": "lower is worse",
    "repetition_4plus_share": "higher is worse", "low_ood_share": "higher is worse",
    "trajectory_v2_distance": "higher is worse", "state_visitation_tvd": "higher is worse",
}
EXTRA_TOLERANCES = ("repetition_10plus_share", "historical_trajectory_distance")
FLOORS = {"repetition_4plus_share": 0.01, "repetition_10plus_share": 0.01, "low_ood_share": 0.01,
          "trajectory_v2_distance": 0.02, "state_visitation_tvd": 0.005, "historical_trajectory_distance": 0.02}
RATE_GUARDS = ("timeout_rate", "mission_completion_rate", "escape_rate")
TOLERANCE_RULE = ("tolerance = max(2 x the between-replicate SD of the metric under B2-v1 (Stage-4A B2 arm, 3 replicates "
                  "of 48 configurations per level, the same level x seed x personality x orientation shape), floor); "
                  "for the rates the floor is 2 x the binomial SD of a difference of two independent Stage-5B arms at "
                  "the Stage-4A rate, sqrt(2 p (1 - p) / n) with n = spies per arm in the scope; catastrophic = 2 x "
                  "tolerance")


def historical_variability() -> dict[str, Any]:
    """Per metric and scope: the Stage-4A B2 (B2-v1) value of each replicate, their SD, the floor and the tolerance.
    Historical B2-v1 data and frozen REAL references only: no B2-v2 output exists when this is computed."""
    from workshop.policy_emulator import stage4_analysis as an
    from workshop.policy_emulator import stage5b_analysis as sa
    d = an.load()
    for s in d["spies"]:
        s["level"] = an.level_of(s)
    refs = sa.references()
    scale, v2s = sa.scales()
    per: dict[str, dict[str, list[float]]] = {}
    for r in REPLICATES:
        rows = {}
        for lv in LEVELS:
            spies = [s for s in d["spies"] if s["level"] == lv and s["replicate"] == r]
            rows[lv] = {**sa.behaviour_metrics(spies), **sa.reference_distances(lv, refs[lv], spies, spies, scale, v2s)["a"]}
        pooled_spies = [s for s in d["spies"] if s["replicate"] == r]
        rows["pooled"] = {**sa.behaviour_metrics(pooled_spies),
                          **{k: statistics.mean(rows[lv][k] for lv in LEVELS) for k in sa.DISTANCE_KEYS}}
        for scope, vals in rows.items():
            for k, v in vals.items():
                per.setdefault(scope, {}).setdefault(k, []).append(v)
    n_spies = {lv: len(PERSONALITIES) * len(ORIENTATIONS) * SEEDS_NEEDED * len(REPLICATES) * 2 for lv in LEVELS}
    n_spies["pooled"] = sum(n_spies.values())
    out: dict[str, Any] = {}
    for scope, metrics in per.items():
        out[scope] = {}
        for k in list(GUARDS) + list(EXTRA_TOLERANCES):
            vals = metrics[k]
            sd = statistics.stdev(vals)
            if k in RATE_GUARDS:
                p = statistics.mean(vals)
                floor = 2 * (2 * p * (1 - p) / n_spies[scope]) ** 0.5
            else:
                floor = FLOORS[k]
            tol = max(2 * sd, floor)
            out[scope][k] = {"replicates": [round(v, 6) for v in vals], "sd": round(sd, 6), "floor": round(floor, 6),
                             "tolerance": round(tol, 4), "catastrophic": round(2 * tol, 4),
                             "direction": GUARDS.get(k, "higher is worse")}
    return {"source": f"workshop_runs/{an.sp.SESSION} (Stage-4A B2 arm, B2-v1, seeds 9001-9006, 3 replicates)",
            "references": sa.REFERENCE_TEXT, "spies_per_arm": n_spies, "rule": TOLERANCE_RULE, "scopes": out}


PROTOCOL_TEXT: dict[str, Any] = {
    "version": "stage5b-selection-protocol-v1",
    "spec": "Archive/STAGE5B_CALL_FREE_CLOSED_LOOP_SURROGATE_SELECTION.md",
    "comparison": "B2-v1 vs the frozen B2-v2 candidate C1 only; no training, no new candidate, no router change",
    "execution": {"controller": "frozen HybridController, router R0 (never escalates), fake legal delegate never asked",
                  "difference_between_arms": "only the action model handed to the controller; B1 (table-v1), target-v2.1, "
                                             "coverage-v1, engine, levels, barrier and R0 are the frozen artifacts in both",
                  "shadow": "the frozen R2.1 router evaluated on the same RouterInputs as R0 at every decision; logged on the "
                            "hybrid_decision event, never returned to the controller",
                  "providers": "0 calls; offline guard in scrub mode in every process"},
    "matrix": {"levels": list(LEVELS), "seeds": SEEDS_NEEDED, "personalities": list(PERSONALITIES),
               "orientations": list(ORIENTATIONS), "replicates": list(REPLICATES), "per_candidate": 432, "total": 864},
    "rng": {"environment_seed": "the base seed (engine, placement, puzzle, spawn, fight / disarm / commit RNGs)",
            "policy_rng_seed": "seed x 100 + replicate", "per_decision": "random.Random(f'surrogate:{policy_rng_seed}:"
                                                                         "{actor_id}:{policy_key}:{decision_index}')",
            "pairing": "B2-v1 and B2-v2 of a (level, seed, personality, orientation, replicate) share every seed; the "
                       "emulator is not part of any RNG key (common random numbers; trajectories may diverge)",
            "python_hash_seed": "0", "launch_order": f"pairs in plan order; within a pair a random.Random({ORDER_SEED}) coin"},
    "primary": {"quantity": "closed-loop shadow R2.1 escalation rate = would-escalate decisions / all decisions, each "
                            "candidate on its own trajectories, pooled over every decision of the arm; a decision would "
                            "escalate when the frozen router.route('R2.1', inputs) returns escalate (no budget cap in shadow)",
                "scopes": ["pooled", "level", "personality", "model identity", "orientation"],
                "differences": ["absolute: B2-v2 - B2-v1", "relative reduction: (B2-v1 - B2-v2) / B2-v1"],
                "material_reduction": {**MATERIAL, "rule": "relative reduction >= 10% AND absolute reduction >= 1 pp, pooled"}},
    "trigger_decomposition": "disagreement only / repetition only / both among shadow escalations; counterfactual "
                             "single-trigger rates (disagreement signal rate, repetition signal rate) and their overlap",
    "disagreement_floor_rule": {
        "A": "B2-v2 reduces both signal rates (each by >= 10% relative)",
        "B": "B2-v2 reduces the repetition rate by >= 10% relative and the disagreement rate by less (disagreement is the floor)",
        "C": "the disagreement rate rises by >= 10% relative (B2-v2 shifts into more disagreement states)",
        "D": "neither signal rate falls by >= 10% relative",
        "precedence": "C > A > B > D", "status": "diagnostic only; R2.1 is not changed"},
    "guards": {k: v for k, v in GUARDS.items()},
    "guard_scope": "pooled over the three levels (rates over all spies; distances = mean of the three per-level distances)",
    "repetition_rule": "repetition pathology reduced or not worsened: pooled and per-level B2-v2 - B2-v1 <= tolerance "
                       "for the shares of decisions in same-action runs of 4+ and of 10+",
    "level_rule": "no level shows a catastrophic regression: on each level every guard metric worse by at most 2 x its "
                  "level tolerance, and the level's shadow escalation not materially higher (the mirror of the material "
                  "threshold: +10% relative and +1 pp)",
    "behaviour_improvement": "B2-v2 closer to the historical REAL reference than B2-v1 by more than the tolerance on the "
                             "pooled trajectory-v2 or state-visitation diagnostic, with no guard failing",
    "decision": {"A": "closed-loop viable: material pooled reduction AND no guard fails AND repetition rule AND level rule",
                 "B": "escalation reduction survives but behaviour regresses: material reduction and any of guard / "
                      "repetition / level rule failing",
                 "C": "behaviour improves but the escalation reduction disappears: no material reduction, guards and level "
                      "rule pass, behaviour improvement",
                 "D": "neither survives closed loop (otherwise)", "precedence": "A > B > C > D",
                 "recommendations": {"A": "proceed to Stage 5C preregistration",
                                     "B": "do not run provider-backed confirmation; analyze the regression",
                                     "C": "B2-v2 may be behaviorally useful, but the active-learning objective did not reduce "
                                          "provider demand enough",
                                     "D": "stop B2-v2 promotion; reconsider the training strategy"}},
    "references": {"level_01": "M4D REAL (seeds 8001-8004, 32 configurations, 8 identities)",
                   "level_02 / level_03": "Stage-4B REAL (seeds 9007-9008, cautious + saboteur, 4 identities)",
                   "status": "historical-reference diagnostics, unmatched configurations, not confirmatory",
                   "not_used": "Stage-4C REAL (protected, regression-only)",
                   "trajectory_v2": "frozen manifest scales; B2-v1 and B2-v2 evaluated jointly (symmetric support = REAL x "
                                    "B2-v1 x B2-v2 identities)",
                   "state_visitation": "frozen cl.state_visitation mean TVD; arms restricted to the reference identities",
                   "historical_trajectory": "frozen cl.trajectory with the Phase-A scale (secondary)"},
    "projection": {"primary": "would-escalate decisions / total decisions",
                   "secondary": f"shadow escalations x {CALLS_PER_ESCALATION} (Stage-4C calls per escalation), descriptive only"},
    "protected_regression": {"configurations": "the 64 Stage-4C configurations (Levels 2 / 3, seeds 9009-9012), replicate 1",
                             "arms": "B2-v1 and B2-v2 through the same Stage-5B plumbing; the B2-v1 run must reproduce the "
                                     "frozen Stage-4C R0 arm decision for decision",
                             "timing": "after stage5b_decision.json is frozen; read once",
                             "rule": "the level rule (catastrophic regression) on Stage-4C; never changes B2-v2, the "
                                     "thresholds, the decision rule or the decision",
                             "label": "protected regression-only"},
    "stage5c": f"seeds {STAGE5C_RESERVED[0]}-{STAGE5C_RESERVED[1]} reserved, untouched",
}


def write_protocol() -> dict[str, Any]:
    ver, seeds = load_verification(), load_seed_selection()
    from workshop.runners.common import session_dir
    if (session_dir(SESSION) / "manifest.json").is_file():
        raise WorkshopError("matches exist already; the selection protocol is frozen before any Stage-5B match")
    body = {**PROTOCOL_TEXT, "tolerances": historical_variability(), "verification_hash": ver["verification_hash"],
            "seed_selection_hash": seeds["selection_hash"], "seeds": seeds["accepted"],
            "b2v2_manifest_hash": B2V2_MANIFEST_HASH}
    return _freeze(PROTOCOL_PATH, json.loads(json.dumps(body, default=str)), "protocol_hash")


def load_protocol() -> dict[str, Any]:
    body = _load_frozen(PROTOCOL_PATH, "protocol_hash")
    if {k: body[k] for k in PROTOCOL_TEXT} != json.loads(json.dumps(PROTOCOL_TEXT)):
        raise WorkshopError("the Stage-5B protocol in code differs from the frozen protocol")
    return body


# -- spec 6 / 7: the plan ----------------------------------------------------------------------------------------------

def configurations(seeds: list[int], levels: tuple[str, ...] = LEVELS) -> list[dict[str, Any]]:
    from workshop.policy_emulator import confirmatory as cf
    out = []
    for level in levels:
        for i, seed in enumerate(seeds, 1):
            for personality in PERSONALITIES:
                for orientation in ORIENTATIONS:
                    t = cf.phase_a_template(personality, orientation)
                    out.append({"config_id": f"{level}-{seed}-{personality}-{orientation}", "level": level, "seed": seed,
                                "seed_index": i, "personality": personality, "orientation": orientation,
                                "black_policy": t["black_policy"], "white_policy": t["white_policy"],
                                "conditions": {**t["conditions"], "seed": seed, "level": level},
                                "phase_a_template": t["config_id"]})
    return out


def arm_entry(c: dict[str, Any], candidate: str, replicate: int, session: str = SESSION) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    cid = c["config_id"]
    return {**c, "experiment": EXPERIMENT_ID, "candidate": candidate, "candidate_label": CANDIDATES[candidate]["label"],
            "arm": "r0", "router": "R0", "shadow_router": "R2.1", "replicate": replicate,
            "policy_rng_seed": c["seed"] * 100 + replicate, "pair_id": f"r{replicate}-{cid}",
            "match_id": f"{candidate}-r{replicate}-{cid}", "rel_dir": f"matches/{candidate}/r{replicate}/{c['level']}/{cid}",
            "baseline": {"session": session, "rel_dir": f"no-plain-b2-run/{cid}"}}


def plan(seeds: list[int] | None = None, *, levels: tuple[str, ...] = LEVELS, replicates: tuple[int, ...] = REPLICATES,
         session: str = SESSION) -> list[dict[str, Any]]:
    """Every match, pairs in (replicate, level, seed, personality, orientation) order; a seeded coin orders the two
    candidates of each pair."""
    seeds = seeds if seeds is not None else load_seed_selection()["accepted"]
    rng = random.Random(ORDER_SEED)
    out = []
    for r in replicates:
        for c in configurations(seeds, levels):
            pair = [arm_entry(c, k, r, session) for k in CANDIDATES]
            if rng.random() < 0.5:
                pair.reverse()
            out += pair
    return out


# -- the arms (the only difference: the action model) -------------------------------------------------------------------

_B2V2: dict[str, Any] = {}


def candidate_dir(key: str) -> Path:
    from workshop.policy_emulator import schema
    from workshop.policy_emulator import stage5a_training as tr
    model, personality = schema.parse_policy_key(key)
    return tr.FREEZE_DIR / model / personality


def b2v2_action(key: str) -> tuple[Any, dict[str, Any]]:
    """The frozen B2-v2 action model of one identity, checked against the candidate manifest."""
    if key not in _B2V2:
        from workshop.policy_emulator.artifacts import _load
        cand = _json(candidate_dir(key).parent.parent / "candidate.json")
        act, meta = _load(candidate_dir(key), "action", "logistic-v1", key, None)
        if meta["content_hash"] != cand["identities"][key] or meta.get("training_dataset_hash") != B2V2_DATA_HASH:
            raise WorkshopError(f"STOP: B2-v2 {key} differs from the frozen candidate manifest")
        _B2V2[key] = (act, meta)
    return _B2V2[key]


def arm_base(candidate: str, base: Any) -> Any:
    """B2-v1: the frozen LoadedPolicy unchanged. B2-v2: the same LoadedPolicy (target-v2.1, coverage-v1 and their
    metadata untouched) with only the action model and its metadata replaced."""
    if candidate == "b2v1":
        return base
    act, meta = b2v2_action(base.policy_key)
    return dataclasses.replace(base, action_policy=act, metadata=meta)


def bound_protocol(candidate: str, key: str, base: Any, protocol: dict[str, Any]) -> dict[str, Any]:
    """The router protocol the controller checks the artifacts against. R0's rules and router hash are unchanged; for
    B2-v2 only the identity's base-emulator hash is bound to the candidate (the protocol artifact has no candidate axis)."""
    if candidate == "b2v1":
        return protocol
    out = copy.deepcopy(protocol)
    out["identities"][key]["base_emulator_hash"] = base.metadata["content_hash"]
    return out


def make_builder(candidate: str):
    """LiveGame._build_hybrid_agent, line for line, with `arm_base` / `bound_protocol` applied to the base."""
    from workshop.live import session as live

    def _build_hybrid_agent(self, spec, delegate, episode_id: str):
        from workshop.policy_emulator import router
        from workshop.policy_emulator.hybrid import HybridController, lab_pricing
        base = live.load_surrogate_policy(self.live_config, spec, emulator=router.BASE_EMULATOR, target=router.TARGET_POLICY)
        b1 = live.load_surrogate_policy(self.live_config, spec, emulator=router.DIAGNOSTIC_EMULATOR, target=router.TARGET_POLICY)
        protocol = router.protocol_for(self.live_config.hybrid_router, self.live_config.policy_dataset)
        base = arm_base(candidate, base)
        protocol = bound_protocol(candidate, spec.policy, base, protocol)
        BINDINGS[spec.actor_id] = {"policy": spec.policy, "candidate": candidate, "emulator": base.emulator,
                                   "action_content_hash": base.metadata["content_hash"],
                                   "training_dataset_hash": base.metadata.get("training_dataset_hash"),
                                   "target_hash": (base.target_metadata or {}).get("content_hash"),
                                   "coverage_hash": base.coverage_hash, "b1_hash": b1.metadata["content_hash"],
                                   "router_hash": protocol["router_hash"],
                                   "bound_base_emulator_hash": protocol["identities"][spec.policy]["base_emulator_hash"]}
        seed = self.live_config.surrogate_seed if self.live_config.surrogate_seed is not None else self.state.seed
        return HybridController(spec.actor_id, base, b1, delegate, router_id=self.live_config.hybrid_router,
                                protocol=protocol, budget=self.hybrid_budget, telemetry=self.telemetry,
                                episode_id=episode_id, seed=seed, dataset=self.live_config.policy_dataset,
                                environment_seed=self.state.seed, provider_name=spec.provider, model_name=spec.model,
                                pricing=lab_pricing())
    return _build_hybrid_agent


BINDINGS: dict[str, dict[str, Any]] = {}
SHADOW: dict[str, Any] = {"last": None, "calls": 0, "r0_results_changed": 0}


def shadow_route(original):
    """router.route with the frozen R2.1 evaluated in shadow on the same inputs. The R0 result object is returned
    unchanged; the shadow result is only remembered for the event fields."""
    def route(router_id, x, *, budget_available: bool = True):
        res = original(router_id, x, budget_available=budget_available)
        if router_id == "R0":
            sh = original("R2.1", x, budget_available=True)
            SHADOW["last"] = sh
            SHADOW["calls"] += 1
        return res
    route.__wrapped__ = original
    return route


def shadow_fields(original):
    """HybridController._router_fields plus the shadow R2.1 result of the decision being recorded."""
    def _router_fields(self, result, b1_probs, *, source):
        out = original(self, result, b1_probs, source=source)
        sh = SHADOW["last"]
        if sh is not None:
            out.update({"shadow_router": "R2.1", "shadow_router_hash": R21_HASH, "shadow_escalate": sh.escalate,
                        "shadow_reasons": list(sh.reasons),
                        "shadow_signals": {k: bool(sh.signals[k]) for k in SHADOW_SIGNALS},
                        "shadow_b1_b2_tvd": sh.b1_b2_tvd, "shadow_repetition_count": sh.repetition_count})
            SHADOW["last"] = None
        return out
    _router_fields.__wrapped__ = original
    return _router_fields


def install(candidate: str) -> None:
    """Bind the arm and the shadow router in this match process (process-local monkeypatches of frozen code)."""
    from workshop.live import session as live
    from workshop.policy_emulator import hybrid
    from workshop.policy_emulator import router as rt
    if candidate not in CANDIDATES:
        raise WorkshopError(candidate)
    live.LiveGame._build_hybrid_agent = make_builder(candidate)
    if not hasattr(rt.route, "__wrapped__"):
        rt.route = shadow_route(rt.route)
    if not hasattr(hybrid.HybridController._router_fields, "__wrapped__"):
        hybrid.HybridController._router_fields = shadow_fields(hybrid.HybridController._router_fields)


def run_one(spec_path: Path, isolation) -> int:
    """One match in this process (the offline guard is already active): level, arm, shadow, then the frozen M4D
    match process; afterwards the guard record, the arm bindings and the shadow counters."""
    from workshop.policy_emulator import stage4_level_select as ls
    from workshop.policy_emulator import stage4c_protocol as cp
    from workshop.runners.common import ensure_within_output_root
    from workshop.scripts import confirmatory_trial as ct
    spec = _json(spec_path)
    entry = spec["entry"]
    run_dir = ensure_within_output_root(Path(spec["run_dir"]))
    ls.select(entry["level"])
    ct.MAX_WALL_SECONDS = cp.MAX_WALL_SECONDS
    install(entry["candidate"])
    error = None
    try:
        code = ct.run_one(spec_path)
    except Exception as exc:  # recorded; the queue sees a failed match
        code, error = 9, f"{type(exc).__name__}: {exc}"
    from workshop.policy_emulator.stage4_analysis import rooms_seen
    rooms = rooms_seen(run_dir) if (run_dir / "telemetry_match01.jsonl").is_file() else None
    record = ls.loaded_record(entry["level"])
    iso = isolation()
    pr._write_json(run_dir / "stage5b_guard.json", {
        "candidate": entry["candidate"], "exit_code": code, "error": error, "level_select": record, "rooms_seen": rooms,
        "bindings": BINDINGS, "shadow_calls": SHADOW["calls"], "isolation": iso, "checked_utc": _now()})
    if code == 0 and not record["ok"]:
        return 8
    if code == 0 and not iso["clean"]:
        return 7
    return code


def load_decision_or_stop() -> dict[str, Any]:
    if not DECISION_PATH.is_file():
        raise WorkshopError("STOP: the protected Stage-4C regression runs only after stage5b_decision.json is frozen")
    return _load_frozen(DECISION_PATH, "decision_hash")


# -- per-match integrity ----------------------------------------------------------------------------------------------

def match_integrity(run_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """The frozen milestone-4B match checks (no provider; B2 recomputed with the arm's own action model) + the R0
    guarantees + the Stage-5B arm, shadow and isolation checks."""
    from unittest import mock
    from workshop.policy_emulator import hybrid_pilot as hp
    from workshop.policy_emulator import stage4_analysis as an
    guard = _json(run_dir / "stage5b_guard.json")
    cand = entry["candidate"]

    def policy(key: str):
        from workshop.policy_emulator import router as rt
        from workshop.policy_emulator.artifacts import load_policy
        return arm_base(cand, load_policy("phase-a-v2", key, emulator=rt.BASE_EMULATOR, target=rt.TARGET_POLICY))
    with mock.patch.object(hp, "_policy", policy):
        base = hp.match_integrity(run_dir, entry, real_provider=False, exp=None, ledger=None)
    checks = dict(base["checks"])
    events = hp._jsonl(run_dir / "telemetry_match01.jsonl")
    match = _json(run_dir / "match.json")
    hyb = [e for e in events if e.get("event_type") == "hybrid_decision"]
    kinds = Counter(e.get("event_type") for e in events)
    iso = guard["isolation"]
    checks["zero_provider"] = {"ok": not match.get("provider_calls") and not match.get("fake_client_calls")
                               and not match.get("llm_decisions") and kinds.get("provider_call", 0) == 0
                               and kinds.get("llm_decision", 0) == 0 and match.get("fake_client") is True
                               and not iso["offline_guard"]["provider_clients_constructed"],
                               "detail": {"provider_calls": match.get("provider_calls"), "fake_client_calls": match.get("fake_client_calls"),
                                          "llm_decisions": match.get("llm_decisions")}}
    checks["router_r0_never_escalates"] = {"ok": match.get("router") == "R0" and bool(hyb) and all(
        e.get("decision_source") == "surrogate" and e.get("router_id") == "R0" and not e.get("router_escalated")
        and not e.get("router_reasons") for e in hyb), "detail": {"decisions": len(hyb)}}
    want = CANDIDATES[cand]["training_dataset_hash"]
    bound = guard["bindings"]
    checks["arm_bound"] = {"ok": set(bound) == {"black_spy", "white_spy"} and all(
        b["candidate"] == cand and b["training_dataset_hash"] == want and b["router_hash"] == R0_HASH
        and b["bound_base_emulator_hash"] == b["action_content_hash"] for b in bound.values())
        and all(e.get("training_dataset_hash") == want for e in hyb), "detail": bound}
    bad = [e.get("decision_id") for e in hyb if e.get("shadow_router") != "R2.1"
           or e.get("shadow_reasons") != [s for s in SHADOW_SIGNALS if (e.get("router_signals") or {}).get(s)]
           or e.get("shadow_escalate") != bool(e.get("shadow_reasons"))
           or e.get("shadow_b1_b2_tvd") != e.get("b1_b2_tvd")]
    checks["shadow_r21_on_every_decision"] = {"ok": not bad and guard["shadow_calls"] == len(hyb),
                                              "detail": {"decisions": len(hyb), "shadow_calls": guard["shadow_calls"],
                                                         "inconsistent": bad[:5]}}
    sel = guard.get("level_select") or {}
    checks["level_loaded"] = {"ok": bool(sel.get("ok")) and sel.get("expected_level") == entry["level"], "detail": sel}
    rooms = set(guard.get("rooms_seen") or [])
    checks["rooms_within_level"] = {"ok": rooms <= set(an._level_rooms(entry["level"])), "detail": sorted(rooms)}
    checks["isolation_clean"] = {"ok": bool(iso["clean"]), "detail": {k: iso[k] for k in ("network_attempts", "credential_reads")}}
    return {"match_id": entry["match_id"], "candidate": cand, "ok": all(c["ok"] for c in checks.values()), "checks": checks}


__all__ = ["verify_artifacts", "write_verification", "load_verification", "used_seeds", "select_seeds",
           "write_seed_selection", "load_seed_selection", "historical_variability", "write_protocol", "load_protocol",
           "configurations", "arm_entry", "plan", "arm_base", "bound_protocol", "make_builder", "shadow_route",
           "shadow_fields", "install", "run_one", "match_integrity", "CANDIDATES", "SESSION", "REGRESSION_SESSION"]
