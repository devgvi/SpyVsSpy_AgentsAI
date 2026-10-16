"""M5D.7: controlled runtime-representation experiment.

Determines whether the Native-vs-Strands gap M5D.3/M5D.5/M5D.6 observed
comes primarily from (A) action/legal-action *representation* or (B) true
runtime/orchestration differences, by holding everything else in the
stack constant across three variants and replaying the same deterministic
scenario fixtures through each:

    native_current-v1    (existing NativeAgentAdapter / LLMController path)
    native_tool_like-v1  (same Native path, tool-like presentation only -
                           see src/llm_controller.py's
                           format_legal_actions_tool_like/
                           NATIVE_TOOL_LIKE_SYSTEM_PROMPT)
    strands_tool-v1      (existing StrandsAgentAdapter / real tool calling)

This module operates on synthetic single-decision `ScenarioFixture`s, not
full Game episodes - the M5D harness (EpisodeRunner/ExperimentRunner) runs
whole missions; M5D.7's question is about one decision's action-selection
behavior under a controlled, replayable Observation, so a lighter, direct
runner is more honest than forcing this into episode shape.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src.llm.telemetry import TelemetryLogger
from src.llm_controller import (
    DEFAULT_SYSTEM_PROMPT,
    NATIVE_CURRENT_REPRESENTATION,
    NATIVE_TOOL_LIKE_REPRESENTATION,
    NATIVE_TOOL_LIKE_SYSTEM_PROMPT,
    STRANDS_TOOL_REPRESENTATION,
    LLMController,
    LLMControllerConfig,
)
from src.observation import Observation, ObservedObject

VARIANTS: tuple[str, ...] = (NATIVE_CURRENT_REPRESENTATION, NATIVE_TOOL_LIKE_REPRESENTATION, STRANDS_TOOL_REPRESENTATION)


class _FakeNavGrid:
    """Minimal NavGrid stand-in (see tests/test_interaction_recovery.py) -
    scenario fixtures care about the *decision* a variant makes, not
    real pathfinding, so a trivial always-resolvable path is enough for a
    proposed MOVE to be accepted by ActionExecutor without erroring."""

    def nearest_approach_point(self, rect, max_distance, from_xy):
        return (from_xy[0] + 80.0, from_xy[1])

    def find_path(self, start_xy, goal_xy):
        return [goal_xy]


def _obj(id: str, **overrides: Any) -> ObservedObject:
    base = dict(
        id=id,
        type="desk",
        label=id,
        position=(10.0, 10.0),
        collision_size=(20.0, 20.0),
        interactable=True,
        trappable=False,
        searched=False,
        inspected=False,
        trap_status="unknown",
        trap_type=None,
        trap_is_mine=False,
        door_open=None,
        lock_type=None,
        locked=False,
        can_unlock=False,
    )
    base.update(overrides)
    return ObservedObject(**base)


def make_observation(**overrides: Any) -> Observation:
    base = dict(
        actor_id="white",
        seed=1,
        position=(660.0, 1124.0),
        facing="east",
        stunned=False,
        stunned_seconds=0.0,
        busy=False,
        busy_seconds=0.0,
        mission_time_remaining=180.0,
        mission_complete=False,
        inventory=frozenset(),
        knowledge=frozenset(),
        required_items=("passport", "secret_documents"),
        trap_inventory={},
        nearby_object_id=None,
        items_found_here=frozenset(),
        objects=(),
        nav_grid=_FakeNavGrid(),
        log_event=lambda *a, **k: None,
        current_room="main_office",
        known_rooms=frozenset({"main_office"}),
        known_doors=frozenset(),
        known_locked_objects={},
        known_item_locations={},
        visible_other_actor=False,
        opponent=None,
        last_seen_opponent=None,
        revealed_opponent=None,
        revealed_opponent_seconds=0.0,
    )
    base.update(overrides)
    return Observation(**base)


@dataclass
class ScenarioFixture:
    scenario_id: str
    description: str
    observation: Observation
    # (action, target_id) pairs expected legal / illegal under this
    # fixture - a cheap, deterministic self-check run before ever calling
    # a live model (spec section 9's "expected legality facts").
    expected_legal: frozenset[tuple[str, str | None]] = field(default_factory=frozenset)
    expected_illegal: frozenset[tuple[str, str | None]] = field(default_factory=frozenset)


def build_scenarios() -> list[ScenarioFixture]:
    """Spec section 8's five scenarios, with 1 and 3 merged into one fixture
    (`far_object`) - they describe the identical game state ("desk visible,
    far, near_object_id=null"), differing only in *which* live-observed
    outcome is measured from it (first-pass legality vs. repair/recovery
    behavior), so replaying it twice would waste live model calls without
    adding a new controlled condition. Documented explicitly, not a silent
    scope reduction.
    """
    desk_far = _obj("desk_01", position=(9000.0, 9000.0))
    scenario_far = ScenarioFixture(
        scenario_id="far_object",
        description="desk_01 visible but far; near_object_id=null. Covers spec section 8 scenarios 1 and 3.",
        observation=make_observation(objects=(desk_far,), nearby_object_id=None),
        expected_legal=frozenset({("move", "desk_01")}),
        expected_illegal=frozenset({("inspect", "desk_01"), ("interact", "desk_01")}),
    )

    statue_near = _obj("statue_01", type="statue_bust", label="statue_01")
    scenario_near = ScenarioFixture(
        scenario_id="near_object",
        description="statue_01 is the actor's current nearby_object_id; INSPECT(statue_01) is legal.",
        observation=make_observation(objects=(statue_near,), nearby_object_id="statue_01"),
        expected_legal=frozenset({("inspect", "statue_01"), ("interact", "statue_01")}),
    )

    desk_inspected = _obj("desk_01", inspected=True, searched=False)
    scenario_post_inspect = ScenarioFixture(
        scenario_id="post_inspect_search",
        description="desk_01 nearby and already inspected; SEARCH (interact) is the legal next step.",
        observation=make_observation(objects=(desk_inspected,), nearby_object_id="desk_01"),
        expected_legal=frozenset({("interact", "desk_01"), ("inspect", "desk_01")}),
    )

    door_far = _obj("archives_door", type="door", position=(9000.0, 1.0))
    statue_near2 = _obj("statue_01", type="statue_bust", label="statue_01")
    desk_far2 = _obj("desk_01", position=(9000.0, 9000.0))
    scenario_multi = ScenarioFixture(
        scenario_id="multiple_legal_actions",
        description="Several simultaneously-legal options: MOVE to two far objects/door, INSPECT the one nearby object.",
        observation=make_observation(objects=(door_far, statue_near2, desk_far2), nearby_object_id="statue_01"),
        expected_legal=frozenset(
            {("move", "archives_door"), ("move", "statue_01"), ("move", "desk_01"), ("inspect", "statue_01"), ("interact", "statue_01")}
        ),
        expected_illegal=frozenset({("inspect", "desk_01"), ("inspect", "archives_door")}),
    )

    return [scenario_far, scenario_near, scenario_post_inspect, scenario_multi]


def verify_scenario_legality(scenario: ScenarioFixture) -> None:
    """Spec section 9's deterministic self-check - run before ever
    dispatching a live model call for this fixture."""
    from src.llm_controller import enumerate_legal_actions

    legal = {(e["action"], e.get("target_id")) for e in enumerate_legal_actions(scenario.observation)}
    missing = scenario.expected_legal - legal
    unexpected = scenario.expected_illegal & legal
    if missing or unexpected:
        raise AssertionError(
            f"scenario {scenario.scenario_id!r} legality mismatch: missing={missing} unexpected_legal={unexpected}"
        )


def run_decision(
    variant: str,
    scenario: ScenarioFixture,
    telemetry: TelemetryLogger,
    model_name: str = "gemma4:latest",
    llm_base_url: str = "http://localhost:11434",
    llm_timeout: float = 60.0,
    repetition: int = 0,
) -> dict[str, Any] | None:
    """Runs exactly one decision for one (variant, scenario) pair against
    the real configured model/provider and returns the emitted
    `llm_decision` telemetry event (None if nothing was emitted - should
    not happen for a real client, but defensive)."""
    episode_id = f"m5d7__{variant}__{scenario.scenario_id}__rep{repetition}"

    if variant in (NATIVE_CURRENT_REPRESENTATION, NATIVE_TOOL_LIKE_REPRESENTATION):
        from src.llm.clients.ollama_client import OllamaClient

        client = OllamaClient(model_name=model_name, base_url=llm_base_url, timeout=llm_timeout)
        system_prompt = NATIVE_TOOL_LIKE_SYSTEM_PROMPT if variant == NATIVE_TOOL_LIKE_REPRESENTATION else DEFAULT_SYSTEM_PROMPT
        config = LLMControllerConfig(system_prompt=system_prompt, representation_variant=variant)
        controller = LLMController(
            "white", model_client=client, config=config, telemetry=telemetry,
            episode_id=episode_id, provider_name="ollama", model_name=model_name,
        )
        controller.get_action(scenario.observation)
        if controller._thinking_thread:
            controller._thinking_thread.join(timeout=llm_timeout * 3 + 5)
    elif variant == STRANDS_TOOL_REPRESENTATION:
        from src.agents.strands_agent import StrandsAgentAdapter

        controller = StrandsAgentAdapter(
            "white", provider_name="ollama", model_name=model_name,
            llm_base_url=llm_base_url, llm_timeout=llm_timeout,
            telemetry=telemetry, episode_id=episode_id,
        )
        controller.get_action(scenario.observation)
        if controller._thinking_thread:
            controller._thinking_thread.join(timeout=llm_timeout * 3 + 5)
    else:
        raise ValueError(f"unknown representation variant: {variant!r}")

    matching = [e for e in telemetry.events if e.get("episode_id") == episode_id and e.get("event_type") == "llm_decision"]
    return matching[-1] if matching else None


def run_experiment(
    output_dir: str,
    repetitions: int = 3,
    model_name: str = "gemma4:latest",
    llm_base_url: str = "http://localhost:11434",
    llm_timeout: float = 60.0,
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, Any]:
    """Runs every (variant x scenario x repetition) combination live,
    writes decisions.jsonl/scenario_results.jsonl/experiment.json/
    summary.json under `output_dir`, and returns the summary dict."""
    os.makedirs(output_dir, exist_ok=True)
    scenarios = build_scenarios()
    for s in scenarios:
        verify_scenario_legality(s)

    telemetry = TelemetryLogger(os.path.join(output_dir, "decisions.jsonl"))
    scenario_results: list[dict[str, Any]] = []
    started_at = time.time()

    for scenario in scenarios:
        for variant in variants:
            for rep in range(repetitions):
                start = time.monotonic()
                event = run_decision(
                    variant, scenario, telemetry,
                    model_name=model_name, llm_base_url=llm_base_url, llm_timeout=llm_timeout, repetition=rep,
                )
                scenario_results.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "representation_variant": variant,
                        "repetition": rep,
                        "wall_time_s": time.monotonic() - start,
                        "decision_id": (event or {}).get("decision_id"),
                        "initial_model_output_valid": (event or {}).get("initial_model_output_valid"),
                        "initial_error_type": (event or {}).get("initial_error_type"),
                        "repair_attempted": (event or {}).get("repair_attempted"),
                        "repair_valid": (event or {}).get("repair_valid"),
                        "recovery_outcome": (event or {}).get("recovery_outcome"),
                        "final_model_action_valid": (event or {}).get("final_model_action_valid"),
                        "validated_action": (event or {}).get("validated_action"),
                        "latency_ms": (event or {}).get("latency_ms"),
                        "input_tokens": (event or {}).get("input_tokens"),
                        "output_tokens": (event or {}).get("output_tokens"),
                    }
                )

    finished_at = time.time()
    telemetry.close()

    with open(os.path.join(output_dir, "scenario_results.jsonl"), "w", encoding="utf-8") as f:
        for row in scenario_results:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize(scenario_results)
    experiment_record = {
        "experiment_id": "m5d7-runtime-representation-v1",
        "started_at": started_at,
        "finished_at": finished_at,
        "repetitions": repetitions,
        "model_name": model_name,
        "llm_base_url": llm_base_url,
        "llm_timeout": llm_timeout,
        "variants": list(variants),
        "scenarios": [s.scenario_id for s in scenarios],
    }
    with open(os.path.join(output_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(render_summary_markdown(summary))

    return summary


def summarize(scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates raw per-decision rows grouped by representation_variant
    (spec section 31's 'aggregation grouped by representation variant'),
    then further broken down per scenario - a pure function over the raw
    rows, safe to recompute later from scenario_results.jsonl alone."""
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in scenario_results:
        by_variant.setdefault(row["representation_variant"], []).append(row)

    def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        first_pass_valid = sum(1 for r in rows if r.get("initial_model_output_valid"))
        repair_attempts = sum(1 for r in rows if r.get("repair_attempted"))
        repair_successes = sum(1 for r in rows if r.get("repair_attempted") and r.get("repair_valid"))
        final_valid = sum(1 for r in rows if r.get("final_model_action_valid"))
        latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        in_tok = [r["input_tokens"] for r in rows if r.get("input_tokens") is not None]
        out_tok = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
        error_types: dict[str, int] = {}
        for r in rows:
            et = r.get("initial_error_type")
            if et:
                error_types[et] = error_types.get(et, 0) + 1
        return {
            "n": n,
            "first_pass_valid": first_pass_valid,
            "first_pass_valid_rate": (first_pass_valid / n) if n else None,
            "repair_attempts": repair_attempts,
            "repair_successes": repair_successes,
            "repair_success_rate": (repair_successes / repair_attempts) if repair_attempts else None,
            "final_valid": final_valid,
            "final_valid_rate": (final_valid / n) if n else None,
            "error_type_breakdown": error_types,
            "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
            "total_input_tokens": sum(in_tok) if in_tok else None,
            "total_output_tokens": sum(out_tok) if out_tok else None,
        }

    result: dict[str, Any] = {"by_variant": {}, "by_variant_by_scenario": {}}
    for variant, rows in by_variant.items():
        result["by_variant"][variant] = _stats(rows)
        by_scenario: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_scenario.setdefault(r["scenario_id"], []).append(r)
        result["by_variant_by_scenario"][variant] = {sid: _stats(srows) for sid, srows in by_scenario.items()}
    return result


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Experiment: m5d7-runtime-representation-v1", ""]
    lines.append("| variant | n | first_pass_valid_rate | repair_success_rate | mean_latency_ms |")
    lines.append("|---|---|---|---|---|")
    for variant, stats in summary.get("by_variant", {}).items():
        lines.append(
            f"| {variant} | {stats['n']} | {stats['first_pass_valid_rate']} | {stats['repair_success_rate']} | {stats['mean_latency_ms']} |"
        )
    return "\n".join(lines) + "\n"
