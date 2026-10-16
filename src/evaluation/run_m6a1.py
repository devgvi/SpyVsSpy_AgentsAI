"""M6A.1: live driver - Phase B (optional, recommended: direct-provider
repeatability control, bypassing both agent runtimes entirely) + Phase C
(repeated frozen replay: reuses M6A's exact snapshot set, N samples per
runtime per snapshot) + Phase D (variance decomposition, written directly
from the real data - no numbers here are asserted in advance).

    python -m src.evaluation.run_m6a1 --preflight-only
    python -m src.evaluation.run_m6a1 --samples 5 --region us-east-1
    python -m src.evaluation.run_m6a1 --samples 5 --region us-east-1 --skip-provider-control

Phase A (network-free) is covered by tests/test_m6a1_runtime_variance.py -
not this driver.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

from src import constants
from src.evaluation.multiseed_stability import resolve_output_dir
from src.evaluation.runtime_snapshot_replay import (
    build_native_adapter,
    build_snapshot_set,
    build_strands_adapter,
    run_one_trial,
)
from src.evaluation.runtime_variance_analysis import (
    action_only_distribution,
    assert_snapshot_integrity,
    build_distribution,
    classify_overall,
    classify_snapshot_runtime_effect,
    content_hash,
    distribution_overlap,
    legal_actions_hash,
    memory_hash,
    modal_decision,
    observation_hash,
    payload_shape_hash,
    trial_signature,
    unique_decision_count,
)
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMControllerConfig, compact_observation_for_model, serialize_observation

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m6a1-runtime-vs-intrinsic-variance-v1")
# Spec section 7: recommended 10, minimum 5. Scoped to the documented
# minimum - the same explicit, session-established scope-reduction pattern
# M6A itself used, given cumulative live Bedrock spend - 8 categories x 5
# samples x 2 runtimes = 80 live calls, already ~1.7x M6A's own budget.
DEFAULT_SAMPLES_PER_SNAPSHOT = 5
# Spec section 22: 3 representative snapshots, chosen because M6A's own
# Stage 1 showed within-runtime target-level spread on exactly these three
# (never post-hoc-cherry-picked from this milestone's own data).
PROVIDER_CONTROL_CATEGORIES = ("SIMPLE_NAVIGATION", "MULTIPLE_LEGAL_TARGETS", "TRAP_PLACEMENT_OPPORTUNITY")
PROVIDER_CONTROL_REPEATS = 10
WALL_TIMEOUT = 120.0


def _verify_sleep_assertion() -> bool:
    try:
        output = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10.0).stdout
        return "PreventUserIdleSystemSleep" in output and "caffeinate" in output
    except Exception:  # pragma: no cover
        return False


def _preflight(out_dir: str, region: str) -> bool:
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=60.0)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


class _CapturingModelClient:
    """Records the exact (system_prompt, user_prompt, output_schema)
    triplet `LLMController._request_action` sends, then answers with a
    fixed canned WAIT so the capturing decision completes instantly with
    no network call - Phase B then fires the *captured* triplet directly
    at the real provider N times (spec section 22-23's "exact same
    Bedrock-level request"), never a hand-reconstructed approximation."""

    def __init__(self) -> None:
        self.captured: dict | None = None
        self._inner = MockModelClient(response={"action": "wait"})

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        if self.captured is None:
            self.captured = {"system_prompt": system_prompt, "user_prompt": user_prompt, "output_schema": output_schema}
        return self._inner.generate_action(system_prompt, user_prompt, output_schema, request_context)


def capture_native_request(snapshot) -> dict:
    telemetry = TelemetryLogger()
    client = _CapturingModelClient()
    adapter = build_native_adapter(
        actor_id=constants.WHITE_SPY_ID, model_client=client, telemetry=telemetry, episode_id="m6a1-capture",
        provider_name="bedrock", model_name=MODEL_ID, personality_profile=None,
    )
    run_one_trial(adapter, snapshot.observation, telemetry, "m6a1-capture", wall_timeout=10.0)
    assert client.captured is not None, f"no request captured for {snapshot.snapshot_id}"
    return client.captured


def run_phase_b(out_dir: str, region: str, snapshots_by_category: dict) -> list[dict]:
    """Optional but recommended (spec section 27): estimates variance
    *below* the runtime layer by sending the identical captured payload
    directly through `BedrockClient`, bypassing both agent adapters."""
    rows: list[dict] = []
    for category in PROVIDER_CONTROL_CATEGORIES:
        snapshot = snapshots_by_category[category]
        request = capture_native_request(snapshot)
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        raw_outputs: list[str] = []
        canonical: list[str] = []
        for i in range(PROVIDER_CONTROL_REPEATS):
            response = client.generate_action(request["system_prompt"], request["user_prompt"], request["output_schema"])
            raw_text = response.raw_text or ""
            raw_outputs.append(raw_text)
            payload = response.payload or {}
            sig = trial_signature({"action": payload.get("action"), "target": payload.get("target_id"), "trap_type": payload.get("trap_type")})
            canonical.append(sig)
            print(f"[PROVIDER-CONTROL] {category} #{i}: raw_len={len(raw_text)} canonical={sig} error={response.error_type}")
        row = {
            "category": category,
            "snapshot_id": snapshot.snapshot_id,
            "repeats": PROVIDER_CONTROL_REPEATS,
            "unique_raw_outputs": len(set(raw_outputs)),
            "unique_canonical_decisions": len(set(canonical)),
            "canonical_distribution": dict(Counter(canonical)),
            "prompt_hash": content_hash(request["system_prompt"], request["user_prompt"]),
            "tool_schema_hash": payload_shape_hash(request["output_schema"]),
        }
        rows.append(row)
        print(f"[PROVIDER-CONTROL] {category}: unique_raw={row['unique_raw_outputs']} unique_canonical={row['unique_canonical_decisions']}")
    _write_jsonl(os.path.join(out_dir, "provider_repeatability.jsonl"), rows)
    return rows


def run_phase_c(out_dir: str, region: str, samples_per_snapshot: int, snapshots: list) -> tuple[list[dict], list[dict]]:
    native_trials: list[dict] = []
    strands_trials: list[dict] = []
    for snapshot in snapshots:
        for i in range(samples_per_snapshot):
            episode_id = f"m6a1-native-{snapshot.snapshot_id}-{i}"
            telemetry = TelemetryLogger(path=os.path.join(out_dir, "runtime_steps.jsonl"))
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            adapter = build_native_adapter(
                actor_id=constants.WHITE_SPY_ID, model_client=client, telemetry=telemetry, episode_id=episode_id,
                provider_name="bedrock", model_name=MODEL_ID, personality_profile=None,
            )
            trial = run_one_trial(adapter, snapshot.observation, telemetry, episode_id, wall_timeout=WALL_TIMEOUT)
            trial.update(snapshot_id=snapshot.snapshot_id, category=snapshot.category, sample_index=i)
            native_trials.append(trial)
            print(f"[NATIVE] {snapshot.category} #{i}: {trial_signature(trial)} valid={trial['valid']} error={trial['runtime_error']}")

        for i in range(samples_per_snapshot):
            episode_id = f"m6a1-strands-{snapshot.snapshot_id}-{i}"
            telemetry = TelemetryLogger(path=os.path.join(out_dir, "runtime_steps.jsonl"))
            adapter = build_strands_adapter(
                actor_id=constants.WHITE_SPY_ID, telemetry=telemetry, episode_id=episode_id,
                provider_name="bedrock", model_name=MODEL_ID, region=region, timeout=60.0, personality_profile=None,
            )
            trial = run_one_trial(adapter, snapshot.observation, telemetry, episode_id, wall_timeout=WALL_TIMEOUT)
            trial.update(snapshot_id=snapshot.snapshot_id, category=snapshot.category, sample_index=i)
            strands_trials.append(trial)
            print(f"[STRANDS] {snapshot.category} #{i}: {trial_signature(trial)} valid={trial['valid']} error={trial['runtime_error']}")

    _write_jsonl(os.path.join(out_dir, "native_repeats.jsonl"), native_trials)
    _write_jsonl(os.path.join(out_dir, "strands_repeats.jsonl"), strands_trials)
    return native_trials, strands_trials


def run_phase_d(out_dir: str, snapshots: list, native_trials: list[dict], strands_trials: list[dict]) -> dict:
    by_snapshot_native: dict[str, list[dict]] = {}
    by_snapshot_strands: dict[str, list[dict]] = {}
    for t in native_trials:
        by_snapshot_native.setdefault(t["snapshot_id"], []).append(t)
    for t in strands_trials:
        by_snapshot_strands.setdefault(t["snapshot_id"], []).append(t)

    distributions: dict[str, dict] = {}
    variance_rows: list[dict] = []
    cost_rows: dict[str, dict] = {}
    per_snapshot_classification: list[str] = []

    for snapshot in snapshots:
        sid = snapshot.snapshot_id
        native = by_snapshot_native.get(sid, [])
        strands = by_snapshot_strands.get(sid, [])
        native_dist = build_distribution(native)
        strands_dist = build_distribution(strands)
        native_mode, native_rate = modal_decision(native_dist)
        strands_mode, strands_rate = modal_decision(strands_dist)
        overlap = distribution_overlap(native_dist, strands_dist)
        classification = classify_snapshot_runtime_effect(native_dist, strands_dist)
        per_snapshot_classification.append(classification)

        native_action_dist = action_only_distribution(native)
        strands_action_dist = action_only_distribution(strands)
        action_overlap = distribution_overlap(native_action_dist, strands_action_dist)

        distributions[sid] = {
            "category": snapshot.category,
            "native": {"counts": dict(native_dist), "n": len(native)},
            "strands": {"counts": dict(strands_dist), "n": len(strands)},
        }
        variance_rows.append({
            "snapshot_id": sid, "category": snapshot.category,
            "native_modal_decision": native_mode, "native_modal_rate": native_rate, "native_unique": unique_decision_count(native_dist),
            "strands_modal_decision": strands_mode, "strands_modal_rate": strands_rate, "strands_unique": unique_decision_count(strands_dist),
            "overlap": overlap, "action_level_overlap": action_overlap,
            "same_modal_decision": native_mode == strands_mode,
            "classification": classification,
        })

        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        for label, group in (("native_tool_like", native), ("strands", strands)):
            n = len(group) or 1
            cost_rows.setdefault(sid, {})[label] = {
                "n": len(group),
                "mean_latency_ms": _mean([t.get("latency_ms") for t in group]),
                "mean_input_tokens": _mean([t.get("input_tokens") for t in group]),
                "mean_output_tokens": _mean([t.get("output_tokens") for t in group]),
                "first_pass_validity_rate": sum(1 for t in group if t["valid"] and not t["repair_attempted"]) / n,
                "repair_rate": sum(1 for t in group if t["repair_attempted"]) / n,
                "fallback_rate": sum(1 for t in group if t["fallback_used"]) / n,
                "runtime_error_rate": sum(1 for t in group if t["runtime_error"]) / n,
            }

    all_native_latency = [t.get("latency_ms") for t in native_trials if t.get("latency_ms") is not None]
    all_strands_latency = [t.get("latency_ms") for t in strands_trials if t.get("latency_ms") is not None]
    mean_native_latency = sum(all_native_latency) / len(all_native_latency) if all_native_latency else 0.0
    mean_strands_latency = sum(all_strands_latency) / len(all_strands_latency) if all_strands_latency else 0.0
    # Cost "differs" if the gap is at least 10% of the smaller mean - a
    # documented, round threshold decided before looking at this run's own
    # numbers (M6A already found +52%, far above this bar either way).
    cost_differs = mean_native_latency > 0 and abs(mean_strands_latency - mean_native_latency) / mean_native_latency >= 0.10

    overall = classify_overall(per_snapshot_classification, cost_differs)

    native_modal_rates = [r["native_modal_rate"] for r in variance_rows]
    strands_modal_rates = [r["strands_modal_rate"] for r in variance_rows]

    result = {
        "variance_rows": variance_rows,
        "distributions": distributions,
        "cost_rows": cost_rows,
        "overall_classification": overall,
        "cost_differs": cost_differs,
        "mean_native_modal_rate": sum(native_modal_rates) / len(native_modal_rates) if native_modal_rates else None,
        "mean_strands_modal_rate": sum(strands_modal_rates) / len(strands_modal_rates) if strands_modal_rates else None,
        "mean_native_latency_ms": mean_native_latency,
        "mean_strands_latency_ms": mean_strands_latency,
    }
    with open(os.path.join(out_dir, "snapshot_variance_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"variance_rows": variance_rows}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "decision_distributions.json"), "w", encoding="utf-8") as f:
        json.dump(distributions, f, indent=2, default=str)
    with open(os.path.join(out_dir, "runtime_cost_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(cost_rows, f, indent=2, default=str)
    return result


def write_snapshot_manifest(out_dir: str, snapshots: list) -> None:
    rows = []
    for s in snapshots:
        payload = serialize_observation(s.observation)
        model_payload = compact_observation_for_model(payload, "native_tool_like-v1")
        violations = assert_snapshot_integrity(s)
        rows.append({
            "snapshot_id": s.snapshot_id, "category": s.category,
            "observation_hash": observation_hash(payload),
            "legal_actions_hash": legal_actions_hash(s.legal_actions),
            "memory_hash": memory_hash({}),  # fresh/empty memory for every snapshot - M6A's own documented scope decision
            "runtime_payload_shape_hash": payload_shape_hash(model_payload),
            "integrity_violations": violations,
        })
    with open(os.path.join(out_dir, "snapshot_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_SNAPSHOT)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--skip-provider-control", action="store_true")
    args = parser.parse_args()

    from src.llm.clients.bedrock_client import resolve_bedrock_region
    region = resolve_bedrock_region(args.region)

    out_dir = resolve_output_dir(BASE_OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    if not _verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        sys.exit(1)

    if not _preflight(out_dir, region):
        print("[FATAL] Bedrock preflight failed - see provider_events.jsonl", file=sys.stderr)
        sys.exit(1)

    if args.preflight_only:
        print("[OK] Preflight passed.")
        return

    snapshots = build_snapshot_set()
    snapshots_by_category = {s.category: s for s in snapshots}
    for s in snapshots:
        violations = assert_snapshot_integrity(s)
        if violations:
            print(f"[FATAL] snapshot {s.snapshot_id} failed integrity check: {violations}", file=sys.stderr)
            sys.exit(1)
    write_snapshot_manifest(out_dir, snapshots)

    start = datetime.now(timezone.utc)
    provider_control_rows = [] if args.skip_provider_control else run_phase_b(out_dir, region, snapshots_by_category)
    native_trials, strands_trials = run_phase_c(out_dir, region, args.samples, snapshots)
    phase_d = run_phase_d(out_dir, snapshots, native_trials, strands_trials)
    end = datetime.now(timezone.utc)

    experiment = {
        "experiment_id": "m6a1-runtime-vs-intrinsic-variance-v1",
        "started_utc": start.isoformat(), "ended_utc": end.isoformat(),
        "model_id": MODEL_ID, "region": region, "provider": "bedrock",
        "samples_per_snapshot": args.samples, "snapshot_count": len(snapshots),
        "native_trial_count": len(native_trials), "strands_trial_count": len(strands_trials),
        "provider_control_ran": bool(provider_control_rows),
    }
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "experiment": experiment,
            "overall_classification": phase_d["overall_classification"],
            "cost_differs": phase_d["cost_differs"],
            "mean_native_modal_rate": phase_d["mean_native_modal_rate"],
            "mean_strands_modal_rate": phase_d["mean_strands_modal_rate"],
            "mean_native_latency_ms": phase_d["mean_native_latency_ms"],
            "mean_strands_latency_ms": phase_d["mean_strands_latency_ms"],
            "variance_rows": phase_d["variance_rows"],
            "provider_repeatability": provider_control_rows,
        }, f, indent=2, default=str)

    print(f"\n[DONE] M6A.1 complete. Output: {out_dir}")
    print(f"Overall classification: {phase_d['overall_classification']}")


if __name__ == "__main__":
    main()
