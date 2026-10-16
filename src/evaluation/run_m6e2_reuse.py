"""M6E.2 spec section 19's own recommended plan (3 snapshots x 8 conditions
x 5 samples = 120 calls) is satisfied by REUSING the 45 calls M6E.1
already collected for 5 of S2's and 4 of S1's 8 cells - the exact same
FactSpec, exact same conditions, exact same production shapes - rather
than re-spending live budget for identical information (this session's
established discipline, most recently applied by M6F's own S2 repair
reuse). Only the genuinely new cells (S2: 3, S1: 4, S0: all 8 - S0 never
received fact-injection treatment in M6E.1) need new live calls, run by
`run_m6e2_native_replay.py`.

    python -m src.evaluation.run_m6e2_reuse
"""

from __future__ import annotations

import json
import os

from src.evaluation.m6e2_factorial import CONDITION_TO_CELL

SOURCE_PATH = os.path.join("evaluation_runs", "m6e1-lossless-compaction-counterbalancing-v1", "native_duplicate_count_trials.jsonl")
OUT_DIR = os.path.join("evaluation_runs", "m6e2-representation-interaction-matrix-v1")

REUSABLE = {
    ("S2_KNOWN_LOCKED_OBJECT", "D1_SINGLE_COPY"), ("S2_KNOWN_LOCKED_OBJECT", "D2_DOUBLE_COPY"),
    ("S2_KNOWN_LOCKED_OBJECT", "L_MEM_AUX"), ("S2_KNOWN_LOCKED_OBJECT", "R1_COMPACT_FORMAT"),
    ("S2_KNOWN_LOCKED_OBJECT", "I1_SINGLE_COPY_COMPACT"),
    ("S1_RECENT_FAILED_ACTION", "D1_SINGLE_COPY"), ("S1_RECENT_FAILED_ACTION", "D2_DOUBLE_COPY"),
    ("S1_RECENT_FAILED_ACTION", "L_MEM_AUX"), ("S1_RECENT_FAILED_ACTION", "R1_COMPACT_FORMAT"),
}


def run() -> dict:
    rows = []
    with open(SOURCE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    reused = []
    for r in rows:
        key = (r["scenario_family"], r["condition"])
        if key in REUSABLE:
            tagged = dict(r)
            tagged["cell"] = list(CONDITION_TO_CELL[r["condition"]])
            tagged["reused_from"] = "m6e1-lossless-compaction-counterbalancing-v1"
            reused.append(tagged)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "native_trials.jsonl"), "a", encoding="utf-8") as f:
        for r in reused:
            f.write(json.dumps(r, default=str) + "\n")

    return {"reused_rows": len(reused)}


if __name__ == "__main__":
    result = run()
    print(f"[DONE] Reused M6E.1 factorial cells into M6E.2: {result}")
