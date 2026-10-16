"""`--debug`: a per-match debug file for the live game.

    workshop_runs/<session>/live/<run-id>/debug_matchNN.jsonl

One JSON object per line, in time order. `kind` says what it is:

* `debug_start` - configuration (no secrets), seed, episode id, timing,
  personalities, encounters, recorder version;
* `frame` - every frame: wall time, frame number, dt, speed, paused,
  mission clock, and per spy: position, room, facing, moving, stun / busy
  timers, near object, lifecycle phase (thinking / ready / held / acting /
  idle / done), the action the executor is running (target, path progress),
  the action handed to the engine this frame, request counters, live-async
  counters; plus the encounter state (distance, cooldown, armed);
* `engine_event` - every engine event (`GameState.event_log`: searches,
  inspections, traps, doors, fights, stuns ...) as it happens, with the
  frame it happened on;
* `agent` - agent lifecycle changes: request launched, answer ready,
  answer committed / discarded, action started / ended (with its terminal
  reason);
* `key` - a control key pressed (window runs);
* `crash` - the exception and traceback if the run fails;
* `debug_end` - frame count and why the match ended.

Read only: nothing here can change a decision. Every line passes through
the same secret redaction as the telemetry. Without `--debug` nothing is
written.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from src import constants

RECORDER_VERSION = "live-debug-v1"


def _r(value: Any, nd: int = 2) -> Any:
    return round(float(value), nd) if isinstance(value, int | float) else value


def _agents(game) -> dict[str, Any]:
    return {constants.BLACK_SPY_ID: game.black_llm_controller, constants.WHITE_SPY_ID: game.bot_controller}


def _phase(game, actor_id: str, agent) -> str:
    state = game.state
    if state.mission_complete.get(actor_id) or (hasattr(state, "out_of_play") and state.out_of_play(actor_id)):
        return "done"
    if getattr(agent, "_thinking_in_progress", False):
        return "thinking"
    if actor_id in getattr(game, "_held_decisions", {}):
        return "held"
    if getattr(agent, "response_ready_uncommitted", False):
        return "ready"
    executor = getattr(agent, "_executor", None)
    if executor is not None and executor.active is not None:
        return "acting"
    return "idle"


def _action_label(action) -> dict | None:
    if action is None:
        return None
    kind = getattr(getattr(action, "type", None), "value", None) or str(getattr(action, "type", None))
    out = {"type": kind}
    for key in ("target_id", "trap_type"):
        if getattr(action, key, None):
            out[key] = getattr(action, key)
    direction = getattr(action, "direction", None)
    if direction is not None:
        out["direction"] = [_r(direction[0], 3), _r(direction[1], 3)]
    return out


class DebugRecorder:
    """One per match. `redact` is the telemetry's secret scrubber."""

    def __init__(self, path: Path, redact) -> None:
        self.path = path
        self._redact = redact
        self._handle = path.open("a", encoding="utf-8")
        self._frame = 0
        self._events_seen = 0
        self._started = time.monotonic()
        self._last: dict[str, dict] = {}
        self.lines = 0

    # -- writing --------------------------------------------------------------
    def write(self, kind: str, **fields: Any) -> None:
        record = {"kind": kind, "t": round(time.monotonic() - self._started, 3), "frame": self._frame, **fields}
        self._handle.write(self._redact(json.dumps(record, sort_keys=True, default=str)) + "\n")
        self.lines += 1

    def start(self, game, config) -> None:
        state = game.state
        self.write(
            "debug_start", recorder=RECORDER_VERSION, episode_id=state.episode_id, seed=state.seed,
            configuration=config.to_public_dict(), decision_timing=getattr(game, "llm_execution_mode", None),
            encounters_enabled=getattr(state, "encounters_enabled", False),
            mission_time_total=state.mission_time_total,
        )

    # -- per frame ------------------------------------------------------------
    def frame(self, game, *, dt: float, speed: float, paused: bool, actions: dict | None = None) -> None:
        self._frame += 1
        state = game.state
        actors = {}
        for actor_id, agent in _agents(game).items():
            actor = state.actors[actor_id]
            executor = getattr(agent, "_executor", None)
            active = executor.active if executor is not None else None
            live = getattr(agent, "live_async_stats", None)
            entry = {
                "pos": [_r(actor.x, 1), _r(actor.y, 1)],
                "room": state.level.room_id_at(actor.x, actor.y),
                "facing": actor.facing, "moving": actor.moving,
                "stunned": _r(actor.stunned_timer), "busy": _r(actor.action_cooldown),
                "near": state.current_interaction_target.get(actor_id),
                "phase": _phase(game, actor_id, agent),
                "lifecycle": game.state.lifecycle(actor_id) if hasattr(game.state, "lifecycle") else None,
                "status": getattr(agent, "status", None),
                "executing": None if active is None else {
                    "action": active.action, "target_id": active.target_id,
                    "path_step": active.path_index, "path_len": len(active.path or []),
                },
                "applied": _action_label((actions or {}).get(actor_id)),
                "requests": getattr(agent, "decisions_requested", None),
                "completed": getattr(agent, "decisions_completed", None),
            }
            if live is not None and getattr(agent, "execution_mode", None) == "live_async":
                entry["live_async"] = live.as_dict()
            actors[actor_id] = entry
            self._agent_transitions(actor_id, agent, entry)
        encounter = None
        if getattr(state, "encounters_enabled", False):
            b, w = state.actors[constants.BLACK_SPY_ID], state.actors[constants.WHITE_SPY_ID]
            encounter = {"distance": _r(((b.x - w.x) ** 2 + (b.y - w.y) ** 2) ** 0.5, 1),
                         "cooldown": _r(state.encounter_cooldown), "armed": state.encounter_armed}
        self.write("frame", dt=_r(dt, 4), speed=speed, paused=paused,
                   mission_time_remaining=_r(state.mission_time_remaining, 3), actors=actors, encounter=encounter)
        self._engine_events(state)

    def _engine_events(self, state) -> None:
        log = state.event_log
        for event in log[self._events_seen:]:
            self.write("engine_event", event=event)
        self._events_seen = len(log)

    def _agent_transitions(self, actor_id: str, agent, entry: dict) -> None:
        # a new match starts idle with nothing requested, so frame 1's launch is recorded too
        previous = self._last.get(actor_id) or {"phase": "idle", "requests": 0, "completed": 0, "executing": None}
        current = {"phase": entry["phase"], "requests": entry["requests"], "completed": entry["completed"],
                   "executing": entry["executing"] and (entry["executing"]["action"], entry["executing"]["target_id"])}
        if current["requests"] != previous["requests"]:
            self.write("agent", actor_id=actor_id, change="request_launched", requests=current["requests"])
        if current["completed"] != previous["completed"]:
            self.write("agent", actor_id=actor_id, change="answer_ready", completed=current["completed"],
                       decision_id=getattr(agent, "_last_decision_id", None))
        if previous["phase"] in ("ready", "held", "thinking") and current["phase"] in ("acting", "idle"):
            live = entry.get("live_async") or {}
            self.write("agent", actor_id=actor_id, change="answer_consumed",
                       classification=live.get("last_classification"), stale_reason=live.get("last_stale_reason"))
        if current["executing"] != previous["executing"]:
            if previous["executing"] is not None:
                executor = getattr(agent, "_executor", None)
                finished = getattr(executor, "last_finished", None)
                self.write("agent", actor_id=actor_id, change="action_ended", action=previous["executing"][0],
                           target_id=previous["executing"][1],
                           failure_detail=getattr(finished, "failure_detail", None))
            if current["executing"] is not None:
                self.write("agent", actor_id=actor_id, change="action_started", action=current["executing"][0],
                           target_id=current["executing"][1])
        self._last[actor_id] = current

    # -- end ------------------------------------------------------------------
    def key(self, name: str, state: dict) -> None:
        self.write("key", key=name, state=state)

    def crash(self, exc: BaseException) -> None:
        self.write("crash", error_type=type(exc).__name__, error=str(exc),
                   traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    def close(self, reason: str, game=None) -> None:
        if self._handle.closed:
            return
        if game is not None:
            self._engine_events(game.state)
        self.write("debug_end", reason=reason, frames=self._frame)
        self._handle.close()
