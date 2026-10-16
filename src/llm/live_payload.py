"""Live play only: a smaller model payload, the same facts.

Measured 2026-10-02 (60-match tournament): a model call carried about 8.7k
input tokens - the system prompt (~2k, now cached) plus a ~14.8k-character
JSON payload whose strategy memory alone grew to ~6k characters. Much of it
repeated what the payload or the system prompt already says. LiveGame
installs `slim_live_payload` as the payload transform of both runtimes
(Native `LLMControllerConfig.model_payload_transform`, Strands
`_model_payload_transform`) and turns on compact JSON separators, so recorded
runs and the zero-call labs keep their exact payloads.

What is removed, and why nothing the agent needs is lost:
- static texts the live system prompt already states: `encounter.rule`,
  `encounter.options`, `encounter.alert.decide`, `puzzle.how_to_answer`,
  `puzzle.wrong_code_costs`, `puzzle.rule`, `time_budget.note`,
  `action_costs.note` (LIVE_PAYLOAD_LEGEND and the rules carry them);
- bookkeeping a model cannot use: `schema_version`, `legal_actions_count`,
  `strategy_memory.recent_decisions[*].decision_id` / `observation_hash`
  (a uuid and a sha256 on every one of the 10 entries);
- repeats of another section of the same payload: `strategy_memory.mission`
  (= `mission` + `time_budget.required_items_missing`),
  `strategy_memory.opponent_memory` (= `opponent`),
  `strategy_memory.exploration.visited_rooms` (= `memory.visited_rooms`),
  a disabled `communication` block with no messages, zero message counts;
- empty values: null fields of room objects and memory records, a lock's
  `requires: "unknown"`; doors carry no search / trap fields (never trappable);
- precision: game times to 0.1 s, positions to whole pixels.
Lists the memory builds from sets are sorted, so the payload no longer
depends on the hash seed.
"""
from __future__ import annotations

from typing import Any, Callable

JSON_SEPARATORS = (",", ":")
_DOOR_ONLY_DROPS = ("trappable", "searched", "inspected", "trap_status")

LIVE_PAYLOAD_LEGEND = (
    "PAYLOAD: a field with no value is left out (room objects, memory records); doors have no search or trap "
    "fields. time_budget.minimum_known_required_time_s is a lower bound - finding a missing item may need more "
    "searching and walking. action_costs are seconds of your mission time; a null move_eta_s means the route runs "
    "through rooms you have not explored."
)


def _round_time(value: Any) -> Any:
    return round(value, 1) if isinstance(value, float) else value


def _no_nulls(record: dict[str, Any]) -> dict[str, Any]:
    return {k: _round_time(v) if k == "game_time" else v for k, v in record.items() if v is not None}


def _slim_object(obj: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in obj.items() if v is not None}
    if out.get("type") == "door":
        for key in _DOOR_ONLY_DROPS:
            out.pop(key, None)
    return out


def _slim_memory(memory: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in memory.items():
        if key in ("schema_version", "mission", "opponent_memory"):
            continue
        if key == "exploration":
            out[key] = {k: v for k, v in value.items() if k != "visited_rooms"}
        elif key == "known_locks":
            out[key] = [{k: v for k, v in lock.items() if not (k == "requires" and v == "unknown")} for lock in value]
        elif key in ("known_empty_objects", "known_searched_objects"):
            out[key] = sorted(value)
        elif key == "recent_decisions":
            out[key] = [_no_nulls({k: v for k, v in d.items() if k not in ("decision_id", "observation_hash")})
                        for d in value]
        elif key in ("recent_interactions", "recent_failures", "recent_events"):
            out[key] = [_no_nulls(r) for r in value]
        elif key in ("messages_sent_count", "messages_received_count") and not value:
            continue
        else:
            out[key] = value
    return out


def slim_live_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The compact model payload (after `strategy_memory` is added), smaller."""
    out = {k: v for k, v in payload.items() if k not in ("schema_version", "legal_actions_count")}
    room = out.get("room")
    if isinstance(room, dict) and isinstance(room.get("objects"), list):
        out["room"] = {**room, "objects": [_slim_object(o) for o in room["objects"]]}
    opponent = out.get("opponent")
    if isinstance(opponent, dict) and isinstance(opponent.get("last_seen"), dict):
        seen = {k: v for k, v in opponent["last_seen"].items() if k != "actor_id"}
        if isinstance(seen.get("position"), list | tuple):
            seen["position"] = [int(round(float(c))) for c in seen["position"]]
        out["opponent"] = {**opponent, "last_seen": _no_nulls(seen)}
        opponent = out["opponent"]
    if isinstance(opponent, dict) and isinstance(opponent.get("position"), list | tuple):
        out["opponent"] = {**opponent, "position": [int(round(float(c))) for c in opponent["position"]]}
    encounter = out.get("encounter")
    if isinstance(encounter, dict):
        slim = {k: v for k, v in encounter.items() if k not in ("rule", "options")}
        if isinstance(slim.get("alert"), dict):
            slim["alert"] = {k: v for k, v in slim["alert"].items() if k != "decide"}
        out["encounter"] = slim
    puzzle = out.get("puzzle")
    if isinstance(puzzle, dict):
        out["puzzle"] = {k: v for k, v in puzzle.items() if k not in ("how_to_answer", "wrong_code_costs", "rule")}
    for key in ("time_budget", "action_costs"):
        if isinstance(out.get(key), dict):
            out[key] = {k: v for k, v in out[key].items() if k != "note"}
    communication = out.get("communication")
    if isinstance(communication, dict) and not communication.get("enabled") \
            and not communication.get("messages_received_recently"):
        out.pop("communication")
    if isinstance(out.get("strategy_memory"), dict):
        out["strategy_memory"] = _slim_memory(out["strategy_memory"])
    return out


def compose(first: Callable | None, then: Callable) -> Callable:
    """`then` applied after an existing transform (if any)."""
    if first is None:
        return then
    return lambda payload: then(first(payload))
