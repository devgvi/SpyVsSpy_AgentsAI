"""M6B Phase A (spec sections 10-12, 27): retrospective, network-free
classification of every PLACE_TRAP attempt already recorded in M6A.2's
own telemetry - never mutates M6A.2's raw outputs (spec section 47),
only reads them.

    python -m src.evaluation.run_m6b_phase_a
"""

from __future__ import annotations

import json
import os

from src.evaluation.place_trap_reliability_analysis import classify_place_trap_event, summarize_place_trap_attempts

M6A2_DIRS = [
    "evaluation_runs/m6a2-runtime-trajectory-effects-v1/pilot",
    "evaluation_runs/m6a2-runtime-trajectory-effects-v1/primary",
]
OUT_DIR = "evaluation_runs/m6b-tool-schema-argument-reliability-v1"


def _load_decision_events(path: str) -> list[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("event_type") == "llm_decision":
                events.append(e)
    return events


def run() -> dict:
    native_events: list[dict] = []
    strands_events: list[dict] = []
    for d in M6A2_DIRS:
        path = os.path.join(d, "decisions.jsonl")
        if not os.path.exists(path):
            continue
        events = _load_decision_events(path)
        native_events.extend(e for e in events if "A_native_tool_like_bedrock" in (e.get("episode_id") or ""))
        strands_events.extend(e for e in events if "B_strands_bedrock" in (e.get("episode_id") or ""))

    native_summary = summarize_place_trap_attempts(native_events)
    strands_summary = summarize_place_trap_attempts(strands_events)

    hallucinated_trap_types: dict[str, int] = {}
    for e in strands_events:
        for src_key in ("raw_response", "repair_raw_response"):
            proposal = e.get(src_key)
            if proposal and proposal.get("action") == "place_trap":
                tt = proposal.get("trap_type")
                if tt is not None:
                    key = str(tt)
                    hallucinated_trap_types[key] = hallucinated_trap_types.get(key, 0) + 1

    result = {
        "denominator_definition": "all decisions where PLACE_TRAP was the model's initial proposal, or (if repair was attempted) the repair's own proposal - counted once per decision, whichever attempt the final classification reflects",
        "sources": M6A2_DIRS,
        "total_decisions": {"native_tool_like": len(native_events), "strands": len(strands_events)},
        "native_tool_like": native_summary,
        "strands": strands_summary,
        "place_trap_selection_rate": {
            "native_tool_like": (native_summary["attempts"] / len(native_events)) if native_events else None,
            "strands": (strands_summary["attempts"] / len(strands_events)) if strands_events else None,
        },
        "strands_proposed_trap_type_values": dict(sorted(hallucinated_trap_types.items(), key=lambda kv: -kv[1])),
        "note_on_m6a2s_own_57.5pct_figure": (
            "M6A.2's own summary.md reported '69/120 decisions (57.5%) ended in a WAIT fallback' for "
            "Strands in the primary run alone - a different denominator (all decisions, one run) from "
            "this report's 'all PLACE_TRAP attempts, both pilot and primary' (0/99). M6A.2 also described "
            "'9 of those 78 repairs actually succeeded' - retrospective analysis here found those 9 were "
            "cases where the model ABANDONED place_trap for a different action (e.g. move) on repair, "
            "which validated fine; none of them were an actual successful place_trap. M6A.2's own number "
            "was not wrong on its own terms, but this report's more precise denominator (conditional on "
            "PLACE_TRAP actually being attempted) is the correct one for schema-reliability purposes "
            "(spec section 33)."
        ),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "m6a2_place_trap_failure_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    md = _render_markdown(result)
    with open(os.path.join(OUT_DIR, "m6a2_place_trap_failure_analysis.md"), "w", encoding="utf-8") as f:
        f.write(md)

    return result


def _render_markdown(result: dict) -> str:
    n = result["native_tool_like"]
    s = result["strands"]
    lines = [
        "# M6A.2 PLACE_TRAP Failure Analysis (M6B Phase A, retrospective)",
        "",
        f"Denominator: {result['denominator_definition']}.",
        "",
        "## Required M6A.2 failure table (spec section 56)",
        "",
        "| runtime | PLACE_TRAP attempts | successful | failed | failure rate | selection rate (of all decisions) |",
        "|---|---|---|---|---|---|",
        f"| native_tool_like | {n['attempts']} | {n['successful']} | {n['failed']} | {n['failure_rate']:.1%} | {result['place_trap_selection_rate']['native_tool_like']:.1%} |",
        f"| strands | {s['attempts']} | {s['successful']} | {s['failed']} | {s['failure_rate']:.1%} | {result['place_trap_selection_rate']['strands']:.1%} |",
        "",
        "## Failure category breakdown",
        "",
        "| category | native_tool_like | strands |",
        "|---|---|---|",
    ]
    from src.evaluation.place_trap_reliability_analysis import ALL_FAILURE_CATEGORIES, SUCCESS

    for cat in (SUCCESS,) + ALL_FAILURE_CATEGORIES:
        nc = n["counts_by_category"].get(cat, 0)
        sc = s["counts_by_category"].get(cat, 0)
        if nc or sc:
            lines.append(f"| {cat} | {nc} | {sc} |")
    lines += [
        "",
        "## Strands' proposed `trap_type` values (initial or repair, whenever the action was place_trap)",
        "",
        "| value | count |",
        "|---|---|",
    ]
    for value, count in result["strands_proposed_trap_type_values"].items():
        lines.append(f"| {value!r} | {count} |")
    lines += ["", "## Reconciling M6A.2's own informal figures", "", result["note_on_m6a2s_own_57.5pct_figure"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    r = run()
    print(json.dumps({"native_tool_like": r["native_tool_like"], "strands": r["strands"]}, indent=2))
