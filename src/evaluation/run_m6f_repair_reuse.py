"""M6F spec sections 26, 37, 48, 62: repair interoperability WITHOUT new
live spend. M6D.2 already collected the exact comparison M6F needs -
INVALID_TRAP_TYPE + MISSING_REQUIRED_FIELD, CURRENT_PRODUCTION (N0) +
NEUTRAL_STRUCTURED (N2) contracts, canonical (P0) rendering, on both
Native (`evaluation_runs/m6d2-neutral-repair-contract-v1/native_trials.jsonl`)
and Strands (`.../strands_trials.jsonl`). Spec section 48 explicitly
forbids reopening the M6D.2 search, and section 26 asks to reuse proven
data, not just proven game states - re-running these 231 calls a second
time would be exactly the "totally new world" duplication spec section 26
warns against, at real additional Bedrock cost for zero new information.

This module filters that existing data to the M6F-relevant cells, tags it
with `reused_from`, and appends it to M6F's own `native_trials.jsonl`/
`strands_trials.jsonl` so those two files remain the single source of
truth across all four scenario families (S0-S3).

    python -m src.evaluation.run_m6f_repair_reuse
"""

from __future__ import annotations

import json
import os

SOURCE_DIR = os.path.join("evaluation_runs", "m6d2-neutral-repair-contract-v1")
OUT_DIR = os.path.join("evaluation_runs", "m6f-runtime-interoperability-v1")
SELECTED_FAILURE_FAMILIES = ("INVALID_TRAP_TYPE", "MISSING_REQUIRED_FIELD")
SELECTED_CONTRACTS = ("N0_CURRENT_PRODUCTION", "N2_NEUTRAL_STRUCTURED")
SELECTED_PERTURBATION = "P0_CANONICAL"


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _relevant(row: dict) -> bool:
    return (
        row.get("failure_family") in SELECTED_FAILURE_FAMILIES
        and row.get("contract") in SELECTED_CONTRACTS
        and row.get("perturbation") == SELECTED_PERTURBATION
    )


def _tag(row: dict, runtime: str) -> dict:
    tagged = dict(row)
    tagged["scenario_family"] = "S2_REPAIR_DECISION"
    tagged["runtime"] = runtime
    tagged["reused_from"] = "m6d2-neutral-repair-contract-v1"
    return tagged


def run() -> dict:
    native_rows = [_tag(r, "native") for r in _read_jsonl(os.path.join(SOURCE_DIR, "native_trials.jsonl")) if _relevant(r)]
    strands_rows = [_tag(r, "strands") for r in _read_jsonl(os.path.join(SOURCE_DIR, "strands_trials.jsonl")) if _relevant(r)]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "native_trials.jsonl"), "a", encoding="utf-8") as f:
        for r in native_rows:
            f.write(json.dumps(r, default=str) + "\n")
    with open(os.path.join(OUT_DIR, "strands_trials.jsonl"), "a", encoding="utf-8") as f:
        for r in strands_rows:
            f.write(json.dumps(r, default=str) + "\n")

    return {"native_reused": len(native_rows), "strands_reused": len(strands_rows)}


if __name__ == "__main__":
    result = run()
    print(f"[DONE] Reused M6D.2 repair data into M6F: {result}")
