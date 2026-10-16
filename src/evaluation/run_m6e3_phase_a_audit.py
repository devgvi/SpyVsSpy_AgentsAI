"""M6E.3 Phase A/16 static audit (spec sections 13, 16): computed entirely
network-free. Produces `history_manifest.json` and
`history_window_audit.md`.

    python -m src.evaluation.run_m6e3_phase_a_audit
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src.evaluation.history_window import (
    HISTORY_FIELDS,
    WINDOW_ORDER,
    event_manifest,
    history_fact_hash,
    history_sequence_hash,
    history_window_hash,
)
from src.evaluation.m6e3_scenarios import KNOWN_FAILURE_EVENT, PRIMARY_FAMILIES, build_primary_scenarios, build_window

OUT_DIR = os.path.join("evaluation_runs", "m6e3-memory-horizon-v1")


def token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def run() -> dict:
    scenarios = build_primary_scenarios()
    history_manifest = {}
    audit_rows = []

    for family in PRIMARY_FAMILIES:
        s = scenarios[family]
        history_manifest[family] = {"scenario_id": s.scenario_id, "events": event_manifest(s.memory_payload_full)}

        seen_hashes = {}
        for window in WINDOW_ORDER:
            r = build_window(s, window)
            payload = {**s.observation_payload, "strategy_memory": r.memory_payload}
            text = json_dumps_sorted(payload)
            entries_retained = sum(len(r.memory_payload.get(f, [])) for f in HISTORY_FIELDS)
            entries_total = sum(len(s.memory_payload_full.get(f, [])) for f in HISTORY_FIELDS)
            wh = history_window_hash(r.memory_payload)
            redundant_with = seen_hashes.get(wh)
            seen_hashes.setdefault(wh, window)
            audit_rows.append({
                "scenario": family, "window": window, "entries_retained": entries_retained,
                "entries_removed": entries_total - entries_retained, "characters": len(text),
                "estimated_tokens": token_estimate(text), "redundant_with_window": redundant_with,
                "history_window_hash": wh, "history_sequence_hash": history_sequence_hash(r.memory_payload),
                "history_fact_hash": history_fact_hash(r.memory_payload),
            })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "history_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(history_manifest, f, indent=2, default=str)

    _write_audit_md(audit_rows)
    return {"rows": len(audit_rows)}


def json_dumps_sorted(payload: dict) -> str:
    import json as _json

    return _json.dumps(payload, sort_keys=True)


def _write_audit_md(rows: list[dict]) -> None:
    lines = [
        "# M6E.3 Phase A — Static History Window Audit\n",
        "Spec sections 16-18. Computed network-free before any live call.\n",
        f"**Known failure event (LONG_HISTORY)**: `{KNOWN_FAILURE_EVENT}` — present in",
        "H_FULL/H4/H3, absent starting at H2 (verified in `test_known_failure_event_present_h_full_h4_h3_absent_h2_h1_h0`).\n",
        "| Scenario | Window | Entries retained | Entries removed | Characters | Est. tokens | Redundant with |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['window']} | {row['entries_retained']} | {row['entries_removed']} | "
            f"{row['characters']} | {row['estimated_tokens']} | {row['redundant_with_window'] or '-'} |"
        )
    lines.append("")
    lines.append(
        "S1's history lists top out at 1 entry (H4/H3/H2/H1 all byte-identical to H_FULL); "
        "S0's M6E.3-seeded history tops out at 3 entries (H4/H3 byte-identical to H_FULL). "
        "Only genuinely distinct windows are live-sampled - see `call_plan.json`."
    )
    with open(os.path.join(OUT_DIR, "history_window_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    result = run()
    print(f"[DONE] Phase A static audit: {result}")
