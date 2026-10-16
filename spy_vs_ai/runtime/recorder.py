"""The replay bundle (FINAL-2 spec 26, 27): an engine-input log plus the match's decision telemetry.

The engine is deterministic given its seed (every random draw comes from seeded generators in GameState) and the
exact sequence of calls it receives. The recorder wraps, on the live GameState instance only, the three calls a
frame makes into the engine - advance_mission_clock (the wall-clock timer), apply_action (per actor, in commit
order) and update(dt) - plus any entry a controller appends to the engine's event log itself (the scripted bot logs
its target choices there), and stores them per frame (run-length encoded). An offline replay builds the same game
(same level, seed and rules, no agents) and re-issues exactly those calls, so it reproduces the committed action and
engine-event order without any controller, provider or network. The engine's own event log digest is stored and
checked at the end of the replay.

Never stored: API keys, AWS credentials, authorization headers or any secret value (the bundle is scrubbed with the
match's Scrubber and checked by `secrets.leaks` before it is written).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from spy_vs_ai import RUNTIME_SCHEMA

REPLAY_SCHEMA = "spy-vs-ai-replay-v1"


def encode_action(action) -> list[Any]:
    direction = list(action.direction) if action.direction is not None else None
    out = [action.type.value, direction, action.target_id, action.trap_type, action.message, action.accompanying_message,
           action.code]
    while out and out[-1] is None:
        out.pop()
    return out


def decode_action(data: list[Any]):
    from src.actions import ActionType, GameAction
    data = list(data) + [None] * (7 - len(data))
    return GameAction(ActionType(data[0]), direction=tuple(data[1]) if data[1] is not None else None, target_id=data[2],
                      trap_type=data[3], message=data[4], accompanying_message=data[5], code=data[6])


def event_digest(event_log: list[dict[str, Any]]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(event_log, sort_keys=True, default=str).encode()).hexdigest()


def final_state(state) -> dict[str, Any]:
    return {a: {"x": round(actor.x, 6), "y": round(actor.y, 6), "lifecycle": state.lifecycle(a),
                "escaped": bool(state.mission_complete.get(a)), "lives": state.lives.get(a) if state.lives_enabled else None}
            for a, actor in sorted(state.actors.items())}


class EngineRecorder:
    """Frames of engine calls; installed on one GameState instance (the class is never patched)."""

    def __init__(self) -> None:
        self.frames: list[list[Any]] = []  # [repeat, ops]
        self._ops: list[list[Any]] | None = None
        self._last_key: str | None = None
        self.frame_count = 0
        self.events_at: list[list[int]] = []  # [frame index, telemetry events count] when it changed
        self._events = 0
        self._seen = 0
        self._sync = None

    def install(self, state) -> None:
        apply, update, advance = state.apply_action, state.update, state.advance_mission_clock
        self._seen = len(state.event_log)

        def sync() -> None:
            # entries a controller appended to the engine's event log (the scripted bot logs its targets / paths)
            log = state.event_log
            if self._ops is not None and len(log) > self._seen:
                self._ops.append(["e", [dict(e) for e in log[self._seen:]]])
            self._seen = len(log)

        def apply_action(actor_id, action, dt):
            sync()
            if self._ops is not None:
                self._ops.append(["a", actor_id, encode_action(action), dt])
            try:
                return apply(actor_id, action, dt)
            finally:
                self._seen = len(state.event_log)

        def update_(dt):
            sync()
            if self._ops is not None:
                self._ops.append(["u", dt, sorted(state.budget_hold) if state.budget_hold else []])
            try:
                return update(dt)
            finally:
                self._seen = len(state.event_log)

        def advance_(seconds):
            sync()
            if self._ops is not None:
                self._ops.append(["c", seconds])
            try:
                return advance(seconds)
            finally:
                self._seen = len(state.event_log)

        self._sync = sync
        state.apply_action, state.update, state.advance_mission_clock = apply_action, update_, advance_

    def begin(self) -> None:
        self._ops = []
        if self._sync is not None:
            self._sync()  # appended between frames: recorded at the start of this one

    def end(self, events: int) -> None:
        if self._sync is not None:
            self._sync()
        ops, self._ops = self._ops or [], None
        key = json.dumps(ops)
        if self.frames and key == self._last_key:
            self.frames[-1][0] += 1
        else:
            self.frames.append([1, ops])
            self._last_key = key
        if events != self._events:
            self.events_at.append([self.frame_count, events])
            self._events = events
        self.frame_count += 1


def iter_frames(frames: list[list[Any]]):
    for repeat, ops in frames:
        for _ in range(int(repeat)):
            yield ops


def apply_frame(game, ops: list[list[Any]]) -> None:
    """Re-issues one recorded frame into the engine, in the order Game._step made the calls."""
    state = game.state
    actions = {op[1]: decode_action(op[2]) for op in ops if op[0] == "a"}
    hold = next((op[2] for op in ops if op[0] == "u"), None)
    prepared = False
    for op in ops:
        if op[0] == "c":  # LiveGame._step advances the wall-clock timer before Game._step
            state.advance_mission_clock(op[1])
            continue
        if op[0] == "e":  # what a controller logged into the engine's event log at this point
            state.event_log.extend(json.loads(json.dumps(op[1])))
            continue
        if not prepared:  # Game._step: facing, then the action-budget hold, then the actions, then update
            if actions and getattr(game, "face_interaction_targets", False):
                game._face_interaction_targets(actions)
            if hold is not None:
                state.budget_hold = set(hold)
            prepared = True
        if op[0] == "a":
            state.apply_action(op[1], actions[op[1]], op[3])
        elif op[0] == "u":
            state.update(op[1])


def bundle(*, runtime_config: dict[str, Any], live_config: dict[str, Any], manifest: dict[str, Any], recorder: EngineRecorder,
           events: list[dict[str, Any]], decisions: list[dict[str, Any]], summary: dict[str, Any], game, complete: bool,
           started_utc: str, finished_utc: str) -> dict[str, Any]:
    keep = ("surrogate_decision", "hybrid_decision", "llm_decision", "opponent_decision", "provider_call", "barrier_commit",
            "live_async_commit", "engine_event", "spy_fight", "match_result", "action_lifecycle", "enhanced_trap")
    traps = getattr(game, "traps", None)
    return {
        "schema": REPLAY_SCHEMA, "runtime_schema": RUNTIME_SCHEMA,
        "kind": "runtime match replay (not a scientific experiment record)",
        "engine": manifest.get("engine"), "level": runtime_config["level"], "level_file": manifest.get("level_file"),
        "seed": game.state.seed, "config": runtime_config, "live_config": live_config,
        "spies": manifest.get("spies"), "execution": manifest.get("execution"),
        "started_utc": started_utc, "finished_utc": finished_utc, "complete": complete,
        "frames": recorder.frames, "frame_count": recorder.frame_count, "events_at": recorder.events_at,
        # the kept events and their position in the match telemetry (events_at counts every event)
        "telemetry": [e for e in events if e.get("event_type") in keep],
        "telemetry_index": [i for i, e in enumerate(events) if e.get("event_type") in keep],
        "decisions": decisions, "outcome": {"match_result": getattr(game, "match_result", None), "summary": summary},
        "verification": {"engine_event_digest": event_digest(game.state.event_log),
                         "engine_events": len(game.state.event_log), "final_state": final_state(game.state)},
        # enhanced gameplay (a playable-runtime rule): the door traps, their variants and what they did. The
        # resolver's events are in the engine event log too, so the engine digest above covers them.
        "enhanced_gameplay": bool(runtime_config.get("enhanced_gameplay", False)),
        "enhanced_traps": traps.summary() if traps is not None else None,
        "secrets_recorded": False,
    }


__all__ = ["EngineRecorder", "bundle", "apply_frame", "iter_frames", "encode_action", "decode_action", "event_digest",
           "final_state", "REPLAY_SCHEMA"]
