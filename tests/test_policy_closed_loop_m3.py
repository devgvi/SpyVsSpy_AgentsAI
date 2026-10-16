"""Stage 3 milestone 3 - closed-loop B1 vs B2 evaluation (zero provider calls).
Spec: Archive/STAGE3_MILESTONE3_CLOSED_LOOP_EVALUATION.md, section 39. The earlier
Stage-3 tests (tests/test_policy_emulator.py, tests/test_policy_models_m2.py) still apply."""
from __future__ import annotations

import hashlib
import json
import socket
import time

import pytest

from tests.workshop_live_helpers import LegalMoveClient  # noqa: F401  (dummy SDL drivers)
from src import constants
from workshop.policy_emulator import closed_loop as cl
from workshop.policy_emulator import dataset as ds
from workshop.policy_emulator import schema
from workshop.policy_emulator.artifacts import TARGET_VERSIONS, load_policy
from workshop.policy_emulator.surrogate import SurrogateController
from workshop.policy_emulator.targets import TargetPolicyV2, TargetPolicyV21
from workshop.runners.common import GUARD, PROJECT_ROOT, WorkshopError, install_guard
from workshop.scripts import closed_loop_policy_eval as runner

DATASET = "phase-a-v2"
WHITE = constants.WHITE_SPY_ID


@pytest.fixture(scope="module")
def configs():
    return cl.phase_a_configurations()


@pytest.fixture(scope="module")
def clean():
    return ds.load(DATASET, "decisions_clean.jsonl")


# -- the plan ---------------------------------------------------------------------------------------

def test_the_96_match_plan_reproduces_the_phase_a_matrix(configs):
    plan = runner.build_plan(configs, runner.DEFAULT_EMULATORS, 1)
    assert len(configs) == 48 and len(plan) == 96
    assert {c["seed"] for c in configs} == {7101, 7202, 7303, 7422, 7435, 7457}
    assert {c["personality"] for c in configs} == {"racer", "cautious", "saboteur", "opportunist"}
    assert {c["orientation"] for c in configs} == {"A", "B"}
    manifest = json.loads((PROJECT_ROOT / cl.PHASE_A_DIR / "manifest.json").read_text())
    by_name = {m["name"]: m for m in manifest["matches"]}
    for c in configs:
        m = by_name[c["phase_a_match"]]
        assert c["black_policy"] == f"{schema.MODELS[m['black_model_key']]['name']}:{m['black']}"
        assert c["white_policy"] == f"{schema.MODELS[m['white_model_key']]['name']}:{m['white']}"
        assert c["conditions"]["seed"] == m["seed"]
    per_config = {}
    for e in plan:
        per_config.setdefault(e["config_id"], []).append(e["emulator"])
    assert all(sorted(v) == ["logistic-v1", "table-v1"] for v in per_config.values())
    assert len({e["match_id"] for e in plan}) == 96 and len({e["rel_dir"] for e in plan}) == 96


def test_the_8_match_pilot_gate(configs):
    pilot = runner.build_plan(configs, runner.DEFAULT_EMULATORS, 1, pilot=True)
    assert len(pilot) == 8
    assert {e["seed"] for e in pilot} == {7101}
    assert {e["personality"] for e in pilot} == {"opportunist", "saboteur"}
    assert {e["orientation"] for e in pilot} == {"A", "B"}
    assert {e["emulator"] for e in pilot} == {"table-v1", "logistic-v1"}


def test_b1_and_b2_get_the_same_environment_seeds_and_target_policy(configs):
    plan = runner.build_plan(configs, runner.DEFAULT_EMULATORS, 2)
    by_key = {}
    for e in plan:
        by_key.setdefault((e["replicate"], e["config_id"]), []).append(runner.match_config(e, runner.DEFAULT_TARGET))
    for (replicate, _), pair in by_key.items():
        a, b = pair
        assert a["policy_emulator"] != b["policy_emulator"]
        strip = lambda c: {k: v for k, v in c.items() if k != "policy_emulator"}  # noqa: E731
        assert strip(a) == strip(b)
        assert a["policy_target"] == b["policy_target"] == "target-v2.1"
        assert a["surrogate_seed"] == runner.policy_rng_seed(a["seed"], replicate) != a["seed"]
        assert a["black_controller"] == a["white_controller"] == "surrogate"
        assert a["decision_timing"] == "evaluation" and a["mission_timer"] == "game_time"
    seeds = {(e["config_id"], e["replicate"]): e["policy_rng_seed"] for e in plan}
    assert seeds[("7101-racer-A", 1)] != seeds[("7101-racer-A", 2)]


def test_dry_run_creates_no_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPY_WORKSHOP_RUNS_DIR", str(tmp_path))
    assert runner.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "48 configurations x 2 emulators x 1 replicate(s) = 96 matches; provider calls: 0" in out
    assert "target policy target-v2.1" in out and "DRY RUN" in out
    assert list(tmp_path.iterdir()) == []


def test_resume_does_not_duplicate_a_finished_match(tmp_path, monkeypatch, configs):
    monkeypatch.setenv("SPY_WORKSHOP_RUNS_DIR", str(tmp_path))
    plan = runner.build_plan(configs, ("table-v1",), 1, only=["7101-opportunist-A"])
    entry = plan[0]
    session = "m3-resume-test"
    write_valid_match(tmp_path / session, entry)
    runner.save_manifest(session, {"matches": {entry["match_id"]: {"status": "done", "rel_dir": entry["rel_dir"]}}})

    def no_launch(*a, **k):
        raise AssertionError("a finished match was launched again")
    monkeypatch.setattr(runner, "_launch", no_launch)
    manifest = runner.run_plan(plan, session=session, target="target-v2.1", parallel=2, resume=True)
    assert list(manifest["matches"]) == [entry["match_id"]] and manifest["matches"][entry["match_id"]]["status"] == "done"
    with pytest.raises(WorkshopError):
        runner.run_plan(plan, session=session, target="target-v2.1", parallel=2, resume=False)


def write_valid_match(session_root, entry, **overrides):
    """The files of a finished, valid closed-loop match (what run_one writes), without playing it."""
    run_dir = session_root / entry["rel_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    match = {"match_id": entry["match_id"], "stop_reason": "match_over", "emulator": entry["emulator"],
             "environment_seed": entry["seed"], "policy_rng_seed": entry["policy_rng_seed"], "provider_calls": 0,
             "network_attempts": [], **overrides}
    (run_dir / "match.json").write_text(json.dumps(match))
    for name in ("summary.json", "run.json"):
        (run_dir / name).write_text(json.dumps({"entry": entry}))
    (run_dir / "telemetry_match01.jsonl").write_text('{"event_type": "match_result"}\n')
    return run_dir


# -- policy randomness --------------------------------------------------------------------------------

def test_policy_rng_is_explicit_reproducible_and_emulator_independent(clean):
    b1 = load_policy(DATASET, "gpt61sol:opportunist", emulator="table-v1", target="target-v2.1")
    b2 = load_policy(DATASET, "gpt61sol:opportunist", emulator="logistic-v1", target="target-v2.1")
    c1 = SurrogateController(WHITE, b1, seed=710101, environment_seed=7101)
    c2 = SurrogateController(WHITE, b2, seed=710101, environment_seed=7101)
    assert c1.rng_seed == c2.rng_seed == "surrogate:710101:white_spy:gpt61sol:opportunist"
    assert c1.policy_rng_seed == 710101 and c1.environment_seed == 7101
    assert [c1.decision_rng(k).random() for k in (1, 2, 3)] == [c2.decision_rng(k).random() for k in (1, 2, 3)]
    rows = [r for r in clean if r["policy_key"] == "gpt61sol:opportunist"][:40]
    again = SurrogateController(WHITE, b1, seed=710101, environment_seed=7101)
    first = [c1.decide_payload(r["observation"], c1.decision_rng(i))["sampled"] for i, r in enumerate(rows, 1)]
    second = [again.decide_payload(r["observation"], again.decision_rng(i))["sampled"] for i, r in enumerate(rows, 1)]
    assert first == second
    other = SurrogateController(WHITE, b1, seed=710102, environment_seed=7101)
    assert [other.decision_rng(k).random() for k in (1, 2, 3)] != [c1.decision_rng(k).random() for k in (1, 2, 3)]


def test_fixed_step_clock_never_sleeps():
    clock = runner.FixedStepClock(constants.FPS)
    started = time.monotonic()
    steps = [clock.tick(constants.FPS) for _ in range(5000)]
    assert time.monotonic() - started < 0.5
    assert set(steps) == {1000.0 / constants.FPS} and clock.ticks == 5000


def test_the_network_guard_refuses_connections(tmp_path):
    install_guard()
    GUARD.activate([tmp_path])
    try:
        with pytest.raises(OSError):
            socket.create_connection(("bedrock-runtime.us-east-1.amazonaws.com", 443), timeout=1)
        assert GUARD.network_attempts
    finally:
        GUARD.deactivate()


# -- one real closed-loop match -------------------------------------------------------------------

def test_a_closed_loop_match_is_provider_free_and_complete(tmp_path, monkeypatch, configs):
    monkeypatch.setenv("SPY_WORKSHOP_RUNS_DIR", str(tmp_path))
    session = "m3-match-test"
    plan = runner.build_plan(configs, ("logistic-v1",), 1, only=["7101-opportunist-A"])
    manifest = runner.run_plan(plan, session=session, target="target-v2.1", parallel=1, resume=False)
    rec = manifest["matches"][plan[0]["match_id"]]
    assert rec["status"] == "done" and rec["provider_calls"] == 0 and rec["network_attempts"] == 0
    root = tmp_path / session
    match = json.loads((root / rec["rel_dir"] / "match.json").read_text())
    assert match["stop_reason"] == "match_over" and match["unexpected_writes"] == []
    assert set(match["requests_after_terminal"].values()) == {0}
    assert match["target_policy"] == "target-v2.1" and match["policy_rng_seed"] == 710101
    assert all(p["emulator_artifact_hash"] and p["target_artifact_hash"] and p["coverage_artifact_hash"]
               for p in match["provenance"].values())
    spies, matches = cl.load_closed_loop(root)
    integ = cl.integrity(matches, spies, 1)
    assert integ["provider_calls"] == 0 and integ["llm_or_provider_events"] == 0
    assert integ["decisions_after_terminal"] == 0 and integ["decisions_missing_fields"] == 0
    assert integ["decisions_missing_coverage"] == 0 and integ["decisions_missing_confidence"] == 0
    assert integ["decisions_mismatched_provenance"] == 0
    for s in spies:
        assert s["decisions"] and all(d["coverage"] in cl.COVERAGE_LABELS for d in s["decisions"])
        assert all(0.0 <= d["confidence"] <= 1.0 for d in s["decisions"])
        assert s["lifecycle"] in cl.LIFECYCLES
    # same configuration and seeds again: the same decisions and engine events
    again = runner.run_plan(plan, session="m3-match-test-again", target="target-v2.1", parallel=1, resume=False)
    assert again["matches"][plan[0]["match_id"]]["status"] == "done"
    repro = cl.compare_sessions(root, tmp_path / "m3-match-test-again")
    assert repro["compared"] == 1 and repro["identical"] == 1


# -- analysis ---------------------------------------------------------------------------------------

def _decision(t, action, coverage="HIGH", confidence=0.9, **features):
    f = {"place_trap_legal": False, "inspect_legal": False, "search_legal": False, "opponent_alert_present": False,
         "current_room": "corridor", "health_band": "high", "life_balance": "unknown", "time_pressure": "relaxed",
         "mission_complete": False, **features}
    category = {"move": "MOVE_PROGRESS", "interact": "SEARCH", "inspect": "INSPECT", "avoid_opponent": "AVOID",
                "approach_opponent": "APPROACH", "place_trap": "PLACE_TRAP", "wait": "WAIT"}[action]
    return {"t": t, "features": f, "action": action, "target": None, "category": category, "legal": ["move", "wait"],
            "coverage": coverage, "confidence": confidence, "decision_index": int(t)}


def _spy(source, config_id, actor, decisions, **kw):
    base = {"source": source, "emulator": source, "replicate": 1 if source != cl.PHASE_A else 0, "config_id": config_id,
            "seed": 7101, "orientation": "A", "actor_id": actor, "side": "black" if actor == "black_spy" else "white",
            "policy": "gpt61sol:opportunist", "model": "gpt61sol", "personality": "opportunist",
            "decisions": decisions, "n_decisions": len(decisions), "lifecycle": "ESCAPED", "active_mission_s": 90.0,
            "match_mission_s": 90.0, "fight_times": [], "run_dir": "x"}
    for key in ("traps_placed", "inspections", "searches", "fights", "hearts_lost"):
        base[key] = 0
    return base | kw


def test_low_ood_stratification_splits_by_coverage():
    ds_ = [_decision(1, "move", "HIGH"), _decision(2, "inspect", "MEDIUM", inspect_legal=True),
           _decision(3, "avoid_opponent", "LOW", 0.95, opponent_alert_present=True),
           _decision(4, "approach_opponent", "OOD", 0.4, opponent_alert_present=True)]
    out = cl.low_ood_stratification({"logistic-v1": [_spy("logistic-v1", "c", "black_spy", ds_)]})["logistic-v1"]
    assert out["HIGH/MEDIUM"]["n"] == 2 and out["LOW/OOD"]["n"] == 2
    assert out["LOW/OOD"]["rates"]["P(avoid | alert)"] == {"n": 2, "k": 1, "rate": 0.5}
    assert out["HIGH/MEDIUM"]["rates"]["P(inspect | legal)"]["rate"] == 1.0
    cc = cl.confidence_coverage({cl.PHASE_A: [], "logistic-v1": [_spy("logistic-v1", "c", "black_spy", ds_)]})
    q = cc["logistic-v1"]["quadrants"]
    assert q["HIGH confidence + HIGH coverage"] == 1 and q["HIGH confidence + LOW/OOD coverage"] == 1
    assert q["LOW confidence + LOW/OOD coverage"] == 1


def test_paired_comparison_is_matched_and_signed():
    groups = {cl.PHASE_A: [], cl.B1: [], cl.B2: []}
    for i, (pa, b1, b2) in enumerate([(10, 12, 9), (10, 14, 11), (20, 20, 25)]):
        cfg = f"cfg{i}"
        groups[cl.PHASE_A].append(_spy(cl.PHASE_A, cfg, "black_spy", [], n_decisions=pa))
        groups[cl.B1].append(_spy(cl.B1, cfg, "black_spy", [], n_decisions=b1))
        groups[cl.B2].append(_spy(cl.B2, cfg, "black_spy", [], n_decisions=b2, lifecycle="ELIMINATED" if i == 2 else "ESCAPED"))
    out = cl.paired(groups)
    m = out["metrics"]["n_decisions"]
    assert out["pairs"] == 3 and m["n"] == 3
    assert m["mean_diff_b_minus_a"] == round(((9 - 12) + (11 - 14) + (25 - 20)) / 3, 4)
    assert (m["b_higher"], m["equal"], m["b_lower"]) == (1, 0, 2)
    assert m["closer_to_phase_a"] == {"B1": 1, "B2": 2, "tie": 0}
    assert out["terminal"]["same_lifecycle"] == 2 and out["terminal"]["transitions"] == {"ESCAPED -> ELIMINATED": 1}


def test_coverage_drift_uses_thirds_of_each_spys_active_time():
    ds_ = [_decision(5, "move", "HIGH"), _decision(40, "move", "MEDIUM"), _decision(80, "move", "OOD"),
           _decision(85, "move", "LOW")]
    out = cl.coverage_drift({cl.PHASE_A: [_spy(cl.PHASE_A, "c", "black_spy", ds_)]})[cl.PHASE_A]
    assert out["n_by_third"] == {"early": 1, "middle": 1, "late": 2}
    assert out["low_ood_by_third"] == {"early": 0.0, "middle": 0.0, "late": 1.0}
    assert out["drift_late_minus_early"] == 1.0 and out["low_ood_share"] == 0.5


def _tree_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_phase_a_reference_is_read_only(tmp_path):
    base = PROJECT_ROOT / cl.PHASE_A_DIR
    files = [base / "manifest.json"] + [PROJECT_ROOT / c["phase_a_run_dir"] / name for c in cl.phase_a_configurations()[:6]
                                        for name in ("run.json", "summary.json", "telemetry_match01.jsonl")]
    before = _tree_hash(files)
    install_guard()
    GUARD.activate([tmp_path])
    try:
        spies = cl.load_phase_a()
    finally:
        GUARD.deactivate()
    assert GUARD.unexpected_writes == []
    assert _tree_hash(files) == before
    assert len(spies) == 96 and sum(len(s["decisions"]) for s in spies) == 3951
    assert {s["source"] for s in spies} == {cl.PHASE_A}


# -- target-v2.1 ------------------------------------------------------------------------------------

def test_target_v21_is_frozen_legal_and_keeps_v2_unchanged(clean):
    assert "target-v2.1" in TARGET_VERSIONS
    rows = [r for r in clean if r["policy_key"] == "sonnet55:racer"]
    v21 = TargetPolicyV21().fit(rows, clean)
    assert v21.prior == "pooled" and any(k.endswith("&mission_complete") for k in v21.beta)
    for r in rows[:150]:
        legal = [t for t in schema.legal_targets(r["observation"]).get(r["intended_action"], []) if t]
        choice = v21.choose(r["intended_action"], r["observation"], r["features"])
        assert all(t in legal or t is None for t in choice.distribution)
        assert abs(sum(choice.distribution.values()) - 1.0) < 1e-6
    again = TargetPolicyV21.from_json(json.loads(json.dumps(v21.to_json())))
    r = next(x for x in rows if x["intended_action"] == "move")
    assert again.choose("move", r["observation"], r["features"]).distribution == \
        v21.choose("move", r["observation"], r["features"]).distribution
    # target-v2 is untouched by the refactor: its stored artifact still reproduces from the data
    lp = load_policy(DATASET, "sonnet55:racer", emulator="table-v1", target="target-v2")
    assert TargetPolicyV2().fit(rows, clean).to_json()["beta"] == lp.target_policy.beta
    loaded = load_policy(DATASET, "sonnet55:racer", emulator="logistic-v1", target="target-v2.1")
    assert loaded.target_version == "target-v2.1" and loaded.coverage_hash
