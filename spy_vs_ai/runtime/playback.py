"""Offline replay of a saved match (FINAL-2 spec 27): 0 provider calls, the committed action / event order reproduced.

A `Playback` builds the same game as the recorded match (same level file, seed and rules) with no agent at all -
every spy is a `ReplayAgent`, which never decides and never builds a client - and re-issues the recorded engine calls
frame by frame (spy_vs_ai.runtime.recorder). The recorded decision telemetry is revealed in step with the frames, so
the play / research views show what the agents decided, when they decided it. At the end the engine's own event log
digest, the final spy states and the ranking are compared with the recording: REPRODUCED or DIVERGED.

Nothing here imports a provider adapter or reads a credential; the replay CLI additionally runs under the offline
guard (no network, no provider client construction, credential-like environment names removed).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from spy_vs_ai.runtime import config as rc
from spy_vs_ai.runtime.decisions import DecisionLog
from spy_vs_ai.runtime.recorder import REPLAY_SCHEMA, apply_frame, event_digest, final_state, iter_frames

REPRODUCED = "REPRODUCED"
DIVERGED = "DIVERGED"
# engine rules a recording carries; the replay game uses the recorded values
ENGINE_FIELDS = {"seed": "seed", "time_limit_s": "time_limit", "mission_clock": "mission_clock",
                 "mission_timer": "mission_timer", "decision_timing": "decision_timing", "commit_policy": "commit_policy",
                 "encounters": "encounters", "lives": "lives", "items": "items", "puzzle": "puzzle", "spawn": "spawn"}
MAX_FROZEN_FRAMES_SHOWN = 30  # interactive playback: a long frozen stretch (a model thinking) is shortened to 0.5 s


class ReplayError(ValueError):
    pass


class ReplayTelemetry:
    """The replay game's telemetry: `events` is the recorded telemetry revealed so far; whatever the replayed engine
    would emit goes to `emitted` (never written anywhere)."""
    path = None

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.emitted: list[dict[str, Any]] = []

    def emit_event(self, event: dict[str, Any]) -> dict[str, Any]:
        self.emitted.append(event)
        return event

    def __getattr__(self, name: str):
        if name.startswith("emit_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


class ReplayDecisionLog(DecisionLog):
    """The decision log over the recorded telemetry; the observation summaries the live run captured (surrogate /
    hybrid spies: the agent's own observation, not in the telemetry) come from the recorded decision records."""

    def __init__(self, agents: dict[str, dict[str, Any]], recorded: list[dict[str, Any]]) -> None:
        super().__init__(agents)
        self.recorded = {r.get("seq"): r for r in recorded or [] if isinstance(r, dict)}

    def _consume(self, e: dict[str, Any]) -> dict[str, Any] | None:
        rec = super()._consume(e)
        if rec is not None and not rec.get("observation"):
            old = self.recorded.get(rec["seq"])
            if old and old.get("spy") == rec["spy"] and old.get("observation"):
                rec["observation"] = old["observation"]
        return rec


def load(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise ReplayError(f"replay file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReplayError(f"{p} is not a JSON replay bundle ({exc})") from exc
    if not isinstance(data, dict) or data.get("schema") != REPLAY_SCHEMA:
        raise ReplayError(f"{p} is not a {REPLAY_SCHEMA} bundle (schema {data.get('schema') if isinstance(data, dict) else '?'})")
    for key in ("config", "frames", "events_at", "telemetry", "verification", "seed"):
        if key not in data:
            raise ReplayError(f"{p}: the bundle has no '{key}'")
    data["_path"] = str(p)
    return data


def _normal(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


class Playback:
    """One recorded match, re-issued into a fresh engine with no agents."""

    def __init__(self, data: dict[str, Any], *, view: str | None = None, speed: float | None = None,
                 label: str | None = None) -> None:
        from workshop.live.overlay import TelemetryView
        from workshop.policy_emulator import stage4_level_select as ls
        from workshop.scripts.tournament import code_fingerprint
        from spy_vs_ai.runtime.session import ACTOR_OF, RuntimeGame, to_live_config
        self.data = data
        cfg = rc.from_dict(data["config"])
        # a recording made before the enhanced gameplay rules has no flag: it ran without them (legacy traps only)
        changes: dict[str, Any] = {"enhanced_gameplay": bool(data["config"].get("enhanced_gameplay", False))}
        changes.update({k: v for k, v in (("view", view), ("speed", speed)) if v is not None})
        self.config = rc.validate(dataclasses.replace(cfg, **changes))
        ls.select(self.config.level)
        recorded_level = (data.get("level_file") or {}).get("sha256")
        self.checks = {"level_file_identical": recorded_level is None or recorded_level == ls.file_sha256(self.config.level),
                       "game_code_identical": (data.get("engine") or {}).get("game_code_fingerprint") in (None, code_fingerprint())}
        live = to_live_config(self.config)
        recorded = data.get("live_config") or {}
        overrides = {field: recorded[key] for key, field in ENGINE_FIELDS.items() if key in recorded}
        overrides["seed"] = data["seed"]
        self.live_config = dataclasses.replace(live, **overrides)
        self.game = RuntimeGame(self.live_config, None, runtime_config=self.config, replay=True, record=False)
        self.telemetry = ReplayTelemetry()
        self.game.telemetry = self.telemetry
        if self.game.state.seed != data["seed"]:
            raise ReplayError(f"the replay engine seed {self.game.state.seed} is not the recorded seed {data['seed']}")
        self.frames = list(iter_frames(data["frames"]))
        if data.get("frame_count") not in (None, len(self.frames)):
            raise ReplayError("the bundle's frames are incomplete")
        self.events_at = [tuple(x) for x in data["events_at"]]
        self.recorded = list(data["telemetry"])
        self.recorded_index = list(data.get("telemetry_index") or range(len(self.recorded)))
        self.i = 0
        self._ea = 0
        self._ri = 0
        self._limit = 0
        self.result: dict[str, Any] | None = None
        agents = {}
        for side in rc.ACTORS:
            a = self.config.agent(side)
            agents[ACTOR_OF[side]] = {"policy": a.policy, "label": a.descriptor.label, "kind": a.descriptor.kind,
                                      "base": a.descriptor.base, "personality": a.personality, "provider": a.provider,
                                      "model": a.model}
        renderer = self.game.renderer
        renderer.game = self.game
        renderer.view = TelemetryView(self.live_config)
        renderer.decision_log = ReplayDecisionLog(agents, data.get("decisions") or [])
        renderer.summary = (data.get("outcome") or {}).get("summary") or None
        renderer.replay = {"label": label or data.get("replay_label") or Path(data.get("_path", "match")).name,
                           "frame": 0, "frames": len(self.frames), "verdict": None}

    # -- stepping ----------------------------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.i >= len(self.frames)

    def next_frame_frozen(self) -> bool:
        return not self.done and not self.frames[self.i]

    def _reveal(self, limit: int) -> None:
        events = self.telemetry.events
        while self._ri < len(self.recorded) and self.recorded_index[self._ri] < limit:
            events.append(self.recorded[self._ri])
            self._ri += 1

    def step(self) -> None:
        apply_frame(self.game, self.frames[self.i])
        while self._ea < len(self.events_at) and self.events_at[self._ea][0] <= self.i:
            self._limit = self.events_at[self._ea][1]
            self._ea += 1
        self._reveal(self._limit)
        self.i += 1
        self.game.renderer.replay["frame"] = self.i

    def run_to_end(self) -> dict[str, Any]:
        while not self.done:
            self.step()
        return self.finish()

    def finish(self) -> dict[str, Any]:
        """Every frame applied: the ranking (if the recording was a finished match) and the verification."""
        if self.result is not None:
            return self.result
        self._reveal(float("inf"))
        game, data = self.game, self.data
        game.renderer.feed()
        recorded_result = (data.get("outcome") or {}).get("match_result")
        match_result = game.finish_match() if (recorded_result is not None or game.match_over()) else None
        v = data["verification"]
        digest = event_digest(game.state.event_log)
        checks = dict(self.checks)
        checks["engine_event_digest_identical"] = digest == v.get("engine_event_digest")
        checks["engine_event_count_identical"] = len(game.state.event_log) == v.get("engine_events")
        checks["final_state_identical"] = _normal(final_state(game.state)) == _normal(v.get("final_state"))
        if data.get("enhanced_traps") is not None:
            # door traps: target, variant, trigger time / order, victim and effect
            replayed = game.traps.summary() if game.traps is not None else None
            checks["enhanced_traps_identical"] = _normal(replayed) == _normal(data["enhanced_traps"])
        if recorded_result is not None:
            checks["ranking_identical"] = _normal((match_result or {}).get("ranking")) == _normal(recorded_result.get("ranking"))
            checks["escape_order_identical"] = _normal((match_result or {}).get("escape_order")) == _normal(
                recorded_result.get("escape_order"))
        engine_ok = all(v for k, v in checks.items() if k not in ("level_file_identical", "game_code_identical"))
        verdict = REPRODUCED if engine_ok else DIVERGED
        renderer = game.renderer
        renderer.match_result = match_result
        renderer.match_over = True
        renderer.replay["verdict"] = verdict
        self.result = {"verdict": verdict, "checks": checks, "frames": len(self.frames),
                       "engine_events": len(game.state.event_log), "engine_event_digest": digest,
                       "recorded_engine_event_digest": v.get("engine_event_digest"),
                       "decisions_shown": renderer.decision_log.count if renderer.decision_log else 0,
                       "provider_calls": 0, "complete_match": bool(data.get("complete")),
                       "game_time_s": round(game.state.elapsed_time, 3)}
        return self.result


__all__ = ["Playback", "ReplayTelemetry", "ReplayDecisionLog", "load", "ReplayError", "REPRODUCED", "DIVERGED",
           "MAX_FROZEN_FRAMES_SHOWN"]
