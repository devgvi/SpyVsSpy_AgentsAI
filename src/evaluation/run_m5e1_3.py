"""M5E.1.3: automated live-diagnostic entry point.

    python -m src.evaluation.run_m5e1_3

1. Measures the real production inter-request gap from M5E.1.2's own
   wrapper telemetry (never guessed).
2. Builds the required 9-cell pacing/connection-reuse matrix (spec
   section 10's required cells; the three optional intermediate-gap
   fresh-client cells are skipped, per that section's own allowance,
   given this session's cumulative live-diagnostic time today).
3. Runs each condition in a fixed, deterministically balanced order (not
   increasing gap order, not client-mode-sorted) against the exact same
   30-item production-like corpus, with a fixed cooldown between
   conditions and a resource snapshot before/mid/after each one.

Headless (SDL_VIDEODRIVER=dummy). Writes everything under
evaluation_runs/m5e1_3-request-pacing-v1/ and never touches any prior
M5D/M5E/M5E.1/M5E.1.1/M5E.2/M5E.1.2 output directory.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import json
import sys
import time
from datetime import datetime, timezone

from src.evaluation.pacing_experiment import (
    balanced_condition_order,
    build_condition_matrix,
    build_corpus,
    load_provider_calls_by_episode,
    measure_production_gap_stats,
    run_pacing_condition,
    summarize_condition,
)
from src.evaluation.resource_telemetry import find_ollama_pid, sample_full_resource_state

MODEL = "gemma4:latest"
BASE_URL = "http://localhost:11434"
NUM_CTX = 16384
LLM_TIMEOUT = 60.0
N_REQUESTS_PER_CONDITION = 30
COOLDOWN_SECONDS = 45.0  # spec section 22: fixed, documented, chosen before the run, not tuned on results
OUT_DIR = os.path.join("evaluation_runs", "m5e1_3-request-pacing-v1")
M5E1_2_WRAPPER_DECISIONS_PATH = os.path.join("evaluation_runs", "m5e1_2-resource-saturation-v1", "wrapper_decisions.jsonl")


def _append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    requests_path = os.path.join(OUT_DIR, "requests.jsonl")
    resource_samples_path = os.path.join(OUT_DIR, "resource_samples.jsonl")

    print(f"=== M5E.1.3 diagnostics starting {datetime.now(timezone.utc).isoformat()} ===")

    gap_stats = measure_production_gap_stats(load_provider_calls_by_episode(M5E1_2_WRAPPER_DECISIONS_PATH))
    print("[PRODUCTION GAP] measured from real M5E.1.2 telemetry:", json.dumps(gap_stats))
    production_gap_ms = gap_stats["median_ms"]
    if production_gap_ms is None:
        raise RuntimeError("could not measure a real production gap from M5E.1.2 telemetry - refusing to guess one")

    corpus = build_corpus(N_REQUESTS_PER_CONDITION, seed=971)
    print(f"[CORPUS] {len(corpus)} items, sizes {corpus[0].prompt_chars}..{corpus[-1].prompt_chars} chars")

    matrix = build_condition_matrix(production_gap_ms)
    ordered_conditions = balanced_condition_order(matrix)
    print("[CONDITION ORDER]", [c.condition_id for c in ordered_conditions])

    with open(os.path.join(OUT_DIR, "conditions.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "production_gap_stats": gap_stats,
                "cooldown_seconds": COOLDOWN_SECONDS,
                "requests_per_condition": N_REQUESTS_PER_CONDITION,
                "ordered_conditions": [c.to_dict() for c in ordered_conditions],
            },
            f, indent=2, default=str,
        )

    ollama_pid = find_ollama_pid()
    condition_summaries = {}
    all_records_by_condition: dict[str, list[dict]] = {}

    for idx, condition in enumerate(ordered_conditions, start=1):
        print(f"[CONDITION {idx}/{len(ordered_conditions)}] {condition.condition_id} (gap={condition.gap_target_ms:.0f}ms, client={condition.client_mode}) starting {datetime.now(timezone.utc).isoformat()}")
        pre_snapshot = sample_full_resource_state(f"{condition.condition_id}_before", ollama_pid, BASE_URL, MODEL)
        _append_jsonl(resource_samples_path, pre_snapshot)

        records: list[dict] = []

        def _on_request(i, record):
            d = record.to_dict()
            records.append(d)
            _append_jsonl(requests_path, d)
            if i == N_REQUESTS_PER_CONDITION // 2:
                mid_snapshot = sample_full_resource_state(f"{condition.condition_id}_mid", ollama_pid, BASE_URL, MODEL)
                _append_jsonl(resource_samples_path, mid_snapshot)
            if not d["success"] or i % 10 == 0:
                print(f"  #{i} success={d['success']} latency={d['wall_latency_ms']:.0f}ms gap={d['actual_gap_ms']}")

        run_pacing_condition(
            experiment_id="m5e1_3-request-pacing-v1", condition=condition, corpus=corpus,
            num_ctx=NUM_CTX, base_url=BASE_URL, model_name=MODEL, timeout=LLM_TIMEOUT, on_request=_on_request,
        )

        post_snapshot = sample_full_resource_state(f"{condition.condition_id}_after", ollama_pid, BASE_URL, MODEL)
        _append_jsonl(resource_samples_path, post_snapshot)

        summary = summarize_condition(records)
        condition_summaries[condition.condition_id] = summary
        all_records_by_condition[condition.condition_id] = records
        print(f"[CONDITION {idx}/{len(ordered_conditions)}] {condition.condition_id} finished: {summary}")

        if idx < len(ordered_conditions):
            print(f"[COOLDOWN] {COOLDOWN_SECONDS}s before next condition")
            time.sleep(COOLDOWN_SECONDS)

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"condition_summaries": condition_summaries, "production_gap_stats": gap_stats}, f, indent=2, default=str)

    print(json.dumps(condition_summaries, indent=2, default=str))
    print(f"\nOutputs written under: {OUT_DIR}/")
    print(f"=== M5E.1.3 diagnostics finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
