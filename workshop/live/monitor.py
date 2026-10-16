"""Read-only stability checks around each live `Game._step` (M7.1).

Nothing here can change a decision: it samples the agents' existing
lifecycle attributes before and after each step, and the telemetry events
the controllers already emit. Definitions follow M6G.3's barrier probe:

* duplicate request (HP09 violation): a new model request launched by a
  spy that, at the start of that step, already had a response in flight,
  ready-but-uncommitted, or held by the barrier;
* unused response: a completed decision that was never handed to the
  loop, never cancelled by the existing busy/stunned discard, and is not
  still pending at the end;
* game time advanced while busy: a step after which some spy is still
  thinking, holding a ready answer or held by the barrier, yet the shared
  mission clock moved. In evaluation timing (frozen clock + barrier) this
  must be 0. In live_async timing it is the point: the clock keeps running
  while a model thinks, so the check becomes the reverse - "frozen while
  thinking" (a running step whose clock did not move while a spy thinks)
  must be 0;
* stall: the longest wall-clock gap with no new decision and no commit
  while the match is running (not paused, not over);
* overlay consistency: whenever the overlay shows the same decision id as
  the newest telemetry event for that spy, every field it derives from
  that decision must equal the event (provider, model, runtime, source,
  repair, fallback, action); lag frames (overlay one event behind) are
  counted separately and must stay short.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src import constants
from workshop.live.config import ACTORS, RUNTIMES
from workshop.live.overlay import TelemetryView, _action_dict_label


def _agents(game) -> dict[str, Any]:
    return {constants.BLACK_SPY_ID: game.black_llm_controller, constants.WHITE_SPY_ID: game.bot_controller}


def _busy(game, actor_id: str, agent) -> dict[str, bool]:
    return {
        "thinking": bool(getattr(agent, "_thinking_in_progress", False)),
        "ready": bool(getattr(agent, "response_ready_uncommitted", False)),
        "held": actor_id in getattr(game, "_held_decisions", {}),
    }


@dataclass
class ActorStats:
    requests: int = 0
    hand_overs: int = 0
    duplicate_requests: int = 0
    memory_chars: list = field(default_factory=list)


class LiveMonitor:
    LAG_TOLERANCE_STEPS = 3  # the overlay redraws every frame; > 3 frames behind is a desync

    def __init__(self) -> None:
        self.stats = {a: ActorStats() for a in ACTORS}
        self.steps = 0
        self.busy_steps = 0
        self.time_advanced_while_busy = 0
        self.frozen_while_thinking = 0
        self.timing: str | None = None
        self.max_clock_advance_while_busy = 0.0
        self.game_time_advanced = 0.0
        self.last_progress_wall = time.monotonic()
        self.max_stall_seconds = 0.0
        self.max_model_wait_seconds = 0.0
        self._wait_started: float | None = None
        self._last_progress_marker: tuple = ()
        self.overlay_checks = 0
        self.overlay_mismatches: list[dict] = []
        self.overlay_lag_run = {a: 0 for a in ACTORS}
        self.overlay_max_lag_steps = 0

    # -- one step --------------------------------------------------------
    def step(self, game, dt: float) -> None:
        agents = _agents(game)
        before = {a: _busy(game, a, agents[a]) for a in ACTORS}
        requested_before = {a: getattr(agents[a], "decisions_requested", 0) for a in ACTORS}
        clock_before = game.state.elapsed_time  # world time (both mission clock modes)

        game._step(dt)

        self.timing = getattr(game, "llm_execution_mode", None)
        self.steps += 1
        clock_after = game.state.elapsed_time
        advanced = clock_after - clock_before
        self.game_time_advanced += max(0.0, advanced)
        after = {a: _busy(game, a, agents[a]) for a in ACTORS}
        for a in ACTORS:
            agent, st = agents[a], self.stats[a]
            new_requests = getattr(agent, "decisions_requested", 0) - requested_before[a]
            st.requests += new_requests
            if new_requests and any(before[a].values()):
                st.duplicate_requests += new_requests
            # A held spy is not polled, so its flag still reads True from the
            # step that handed over; count a hand-over only on a polled step.
            if not before[a]["held"] and getattr(agent, "last_call_committed_ready_response", False):
                st.hand_overs += 1
        if any(any(flags.values()) for flags in after.values()):
            self.busy_steps += 1
            if advanced > 1e-9:
                self.time_advanced_while_busy += 1
                self.max_clock_advance_while_busy = max(self.max_clock_advance_while_busy, advanced)
            elif dt > 0 and any(flags["thinking"] for flags in after.values()):
                self.frozen_while_thinking += 1

    # -- progress / stall ------------------------------------------------
    def note_progress(self, view: TelemetryView, running: bool, game=None) -> None:
        """A stall is wall time with no new decision or commit while nothing is
        outstanding (no model call in flight, nothing ready or held) - i.e.
        the loop is not waiting on a model, it is simply not deciding.
        Waiting on a slow model call is tracked separately."""
        marker = (tuple(view.decision_counts.get(a, 0) for a in ACTORS), view.commits)
        now = time.monotonic()
        outstanding = game is not None and any(any(_busy(game, a, ag).values()) for a, ag in _agents(game).items())
        if marker != self._last_progress_marker or not running:
            self._last_progress_marker = marker
            self.last_progress_wall = now
            self._wait_started = None
            return
        if outstanding:
            self._wait_started = self._wait_started or now
            self.max_model_wait_seconds = max(self.max_model_wait_seconds, now - self._wait_started)
            self.last_progress_wall = now  # waiting on a model is not a stall
        else:
            self._wait_started = None
            self.max_stall_seconds = max(self.max_stall_seconds, now - self.last_progress_wall)

    # -- overlay vs telemetry -------------------------------------------
    def check_overlay(self, game, view: TelemetryView) -> None:
        events = game.telemetry.events if game.telemetry is not None else []
        latest: dict[str, dict] = {}
        for e in events:
            if e.get("event_type") == "llm_decision":
                latest[e.get("actor_id")] = e
        for a in ACTORS:
            ev = latest.get(a)
            if ev is None:
                continue
            snap = view.actor(game, a)
            if snap.decision_id != ev.get("decision_id"):
                self.overlay_lag_run[a] += 1
                self.overlay_max_lag_steps = max(self.overlay_max_lag_steps, self.overlay_lag_run[a])
                continue
            self.overlay_lag_run[a] = 0
            self.overlay_checks += 1
            shown_action = snap.action
            telemetry_action = _action_dict_label(ev.get("executed_action") or ev.get("validated_action"))
            expected = {
                "provider": ev.get("provider"), "model": ev.get("model"), "runtime": ev.get("agent_runtime"),
                "source": ev.get("action_source"), "repair": bool(ev.get("repair_attempted")),
                "fallback": bool(ev.get("fallback_used")),
            }
            shown = {
                "provider": snap.provider, "model": snap.model, "runtime": RUNTIMES.get(snap.runtime, snap.runtime),
                "source": snap.last_source, "repair": snap.last_repair, "fallback": snap.last_fallback,
            }
            diffs = {k: (shown[k], expected[k]) for k in expected if shown[k] != expected[k]}
            if shown_action != telemetry_action:
                diffs["action"] = (shown_action, telemetry_action)
            if diffs:
                self.overlay_mismatches.append({"actor": a, "decision_id": ev.get("decision_id"), "diffs": diffs})
        # The commit line: a release happens inside _step before render, and the
        # view updates during render, so the frame just drawn must show it.
        if game.last_barrier_release is not None:
            self.overlay_checks += 1
            if view.last_commit is not game.last_barrier_release:
                self.overlay_mismatches.append({"actor": None, "decision_id": None,
                                                "diffs": {"commit": (view.last_commit, game.last_barrier_release)}})

    def record_memory(self, view: TelemetryView) -> None:
        for a in ACTORS:
            ev = view.last_decision.get(a)
            if ev and isinstance(ev.get("memory_size_chars"), int | float):
                chars = self.stats[a].memory_chars
                if not chars or chars[-1] != ev["memory_size_chars"]:
                    chars.append(ev["memory_size_chars"])

    # -- report ------------------------------------------------------------
    def report(self, game) -> dict[str, Any]:
        agents = _agents(game)
        events = game.telemetry.events if game.telemetry is not None else []
        per_actor = {}
        for a in ACTORS:
            agent, st = agents[a], self.stats[a]
            completed = sum(1 for e in events if e.get("event_type") == "llm_decision" and e.get("actor_id") == a)
            cancelled = int(getattr(agent, "ready_responses_cancelled", 0) or 0)
            live = getattr(agent, "live_async_stats", None) if getattr(agent, "execution_mode", None) == "live_async" else None
            stale_discards = int(live.discarded_due_to_stale) if live is not None else 0
            flags = _busy(game, a, agent)
            pending = int(flags["ready"] or flags["held"])
            in_flight = int(flags["thinking"])
            unused = max(0, completed - st.hand_overs - cancelled - stale_discards - pending)
            mem = st.memory_chars
            per_actor[a] = {
                "requests_launched": st.requests,
                "decisions_completed": completed,
                "hand_overs": st.hand_overs,
                "responses_cancelled_busy_or_stunned": cancelled,
                "responses_discarded_stale": stale_discards,
                "live_async": live.as_dict() if live is not None else None,
                "pending_at_end": pending,
                "in_flight_at_end": in_flight,
                "unused_responses": unused,
                "duplicate_requests": st.duplicate_requests,
                "memory_chars_first_max_last": [mem[0], max(mem), mem[-1]] if mem else None,
            }
        return {
            "steps": self.steps,
            "busy_steps": self.busy_steps,
            "game_time_advanced_s": round(self.game_time_advanced, 2),
            "decision_timing": self.timing,
            "steps_game_time_advanced_while_busy": self.time_advanced_while_busy,
            "steps_frozen_while_thinking": self.frozen_while_thinking,
            "max_clock_advance_while_busy_s": round(self.max_clock_advance_while_busy, 4),
            "max_stall_seconds": round(self.max_stall_seconds, 1),
            "max_model_wait_seconds": round(self.max_model_wait_seconds, 1),
            "overlay_checks": self.overlay_checks,
            "overlay_mismatches": self.overlay_mismatches[:20],
            "overlay_mismatch_count": len(self.overlay_mismatches),
            "overlay_max_lag_steps": self.overlay_max_lag_steps,
            "actors": per_actor,
        }
