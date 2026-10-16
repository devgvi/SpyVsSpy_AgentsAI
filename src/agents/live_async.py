"""Live-play decision timing: LIVE_ASYNC (bugfix live flow, Part A).

Evaluation mode (the default, used by every M5/M6 experiment and stored
replay) freezes the shared game clock while any LLM request is in flight and,
between two LLM agents, holds each answer at the M6G.3 response barrier. That
is what makes those runs reproducible, and it is also why a live match looks
frozen most of the time: in the M7.1 ui-check run a model call was
outstanding in 84.7% of frames, so 480 s of wall time produced 72 s of game
time.

``execution_mode="live_async"`` is a separate, opt-in mode used only by the
workshop live launcher:

* the game clock keeps running while requests are in flight
  (``Game._step`` only freezes for ``execution_mode == "evaluation"``);
* there is no pair barrier (``Game._barrier_active`` requires evaluation mode
  for both agents), so each spy commits on its own;
* a thinking spy with no active action stands still (WAIT); an active MOVE
  keeps executing because requests are only launched at action boundaries;
* HP09 is unchanged: no new request while one is in flight or ready;
* the answer is not started when it arrives on the background thread. It is
  classified on the game thread against the world as it is *now*:

  - FRESH: the observation digest equals the one the request was built from;
  - STALE_BUT_STILL_LEGAL: the world changed but the action is still in the
    current legal action set (and, for PLACE_TRAP, the trap is still held);
  - STALE_AND_ILLEGAL: it is not - the action is discarded (never applied,
    never silently turned into a WAIT decision), counted, recorded, and the
    actor asks for a new decision on the next frame.

Nothing here changes action semantics (M6G.1): only when the world ticks and
which actor may act while the other thinks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.actions import ActionType, GameAction
from src.observation import Observation

EVALUATION = "evaluation"
LIVE_ASYNC = "live_async"
DECISION_TIMING_MODES = (EVALUATION, LIVE_ASYNC)

FRESH = "FRESH"
STALE_BUT_STILL_LEGAL = "STALE_BUT_STILL_LEGAL"
STALE_AND_ILLEGAL = "STALE_AND_ILLEGAL"
CLASSIFICATIONS = (FRESH, STALE_BUT_STILL_LEGAL, STALE_AND_ILLEGAL)
# V3: the actor became terminal (escaped, timed out, eliminated) while its
# answer was in flight or ready - never committed.
DISCARDED_TERMINAL = "DISCARDED_TERMINAL"

# Actions that need no target to be legal.
_UNTARGETED = {"wait"}


@dataclass
class PendingLiveDecision:
    """A completed decision waiting for the game thread to commit it."""

    decision_id: str | None
    decision_boundary_id: str
    snapshot_id: str  # digest of the observation the request was built from
    action_dict: dict[str, Any]
    valid: bool  # False: the model output was rejected and the fallback is WAIT
    decision_game_time: float
    request_start_game_time: float | None
    # mission time budget V2: the time_budget/action_costs the model saw
    facts: dict | None = None
    # live fight rule: the actor's KO count when the request was made - an
    # answer to a request from before a KO never applies after the respawn
    ko_count: int = 0
    # live encounter alerts: whether the model saw the opponent close
    alert_seen: bool = False


@dataclass
class LiveAsyncStats:
    """Per-agent live-async bookkeeping, read by the monitor and the overlay."""

    requests: int = 0
    commits: int = 0
    fresh: int = 0
    stale_but_still_legal: int = 0
    stale_and_illegal: int = 0
    discarded_due_to_stale: int = 0
    discarded_terminal: int = 0
    last_classification: str | None = None
    last_stale_reason: str | None = None
    # current request (reset at each launch)
    request_start_game_time: float | None = None
    world_ticks_during_request: int = 0
    opponent_move_ticks_during_request: int = 0
    # this actor moved while the opponent had a request in flight, since this
    # actor's previous commit
    moved_during_opponent_request_ticks: int = 0
    boundary_seq: int = 0
    history: list = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests, "commits": self.commits, "fresh": self.fresh,
            "stale_but_still_legal": self.stale_but_still_legal, "stale_and_illegal": self.stale_and_illegal,
            "discarded_due_to_stale": self.discarded_due_to_stale,
            "discarded_terminal": self.discarded_terminal,
            "last_classification": self.last_classification, "last_stale_reason": self.last_stale_reason,
        }


def is_live_async(agent: Any) -> bool:
    return str(getattr(agent, "execution_mode", EVALUATION)).lower() == LIVE_ASYNC


def game_clock(observation: Observation) -> float:
    """The shared mission clock reading (seconds remaining) - the one game
    clock both actors run on. Durations are start minus commit."""
    return round(float(observation.mission_time_remaining), 3)


def action_still_legal(observation: Observation, action_dict: dict[str, Any],
                       legal_actions: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """Is this previously validated action still in the current legal set?

    Uses the same legal surface the model was offered (`legal_actions` of
    the current serialized observation), so "legal" means exactly what it
    meant at request time - only evaluated against the world as it is now."""
    action = str(action_dict.get("action") or "wait").lower()
    if action in _UNTARGETED:
        return True, None
    target_id = action_dict.get("target_id")
    same_action = [entry for entry in legal_actions if entry.get("action") == action]
    if not same_action:
        return False, f"{action}_no_longer_legal"
    if not any(entry.get("target_id") == target_id for entry in same_action):
        return False, f"{action}_target_{target_id}_no_longer_legal"
    if action == "place_trap":
        trap_type = action_dict.get("trap_type")
        if observation.trap_inventory.get(trap_type, 0) <= 0:
            return False, f"trap_type_{trap_type}_no_longer_in_inventory"
    return True, None


def classify_ready_decision(pending: PendingLiveDecision, current_digest: str, observation: Observation,
                            legal_actions: list[dict[str, Any]]) -> tuple[str, str | None]:
    if getattr(observation, "ko_count", 0) != pending.ko_count:
        return STALE_AND_ILLEGAL, "knocked_out_during_request"
    from src.encounter_intents import encounter_alert
    if not pending.alert_seen and encounter_alert(observation) is not None:
        # decided before the opponent came close: the agent decides again, with the alert
        return STALE_AND_ILLEGAL, "opponent_came_close_during_request"
    if current_digest == pending.snapshot_id:
        return FRESH, None
    legal, reason = action_still_legal(observation, pending.action_dict, legal_actions)
    if legal:
        return STALE_BUT_STILL_LEGAL, "observation_changed_during_request"
    return STALE_AND_ILLEGAL, reason


# -- agent-level helpers shared by LLMController and StrandsAgentAdapter ------

def on_request_launched(agent: Any, observation: Observation, observation_digest: str) -> str:
    """Records a new per-actor decision boundary; returns its id."""
    stats: LiveAsyncStats = agent.live_async_stats
    agent._live_request_digest = observation_digest
    agent._live_request_ko_count = getattr(observation, "ko_count", 0)
    from src.encounter_intents import encounter_alert
    agent._live_request_alert = encounter_alert(observation) is not None
    stats.requests += 1
    stats.boundary_seq += 1
    stats.request_start_game_time = game_clock(observation)
    stats.world_ticks_during_request = 0
    stats.opponent_move_ticks_during_request = 0
    boundary_id = f"{agent.actor_id}:b{stats.boundary_seq}"
    agent._live_boundary_id = boundary_id
    return boundary_id


def note_world_tick(agent: Any, *, self_moved: bool, opponent_moved: bool, opponent_thinking: bool) -> None:
    """Called by Game._step once per frame in live_async mode only."""
    stats: LiveAsyncStats = agent.live_async_stats
    if getattr(agent, "_thinking_in_progress", False):
        stats.world_ticks_during_request += 1
        if opponent_moved:
            stats.opponent_move_ticks_during_request += 1
    if opponent_thinking and self_moved:
        stats.moved_during_opponent_request_ticks += 1


def make_pending(agent: Any, *, decision_id: str | None, action_dict: dict[str, Any], valid: bool,
                 decision_game_time: float) -> PendingLiveDecision:
    stats: LiveAsyncStats = agent.live_async_stats
    return PendingLiveDecision(
        decision_id=decision_id,
        decision_boundary_id=getattr(agent, "_live_boundary_id", None) or f"{agent.actor_id}:b{stats.boundary_seq}",
        snapshot_id=getattr(agent, "_live_request_digest", None) or "",
        action_dict=dict(action_dict),
        valid=valid,
        decision_game_time=decision_game_time,
        request_start_game_time=stats.request_start_game_time,
        facts=getattr(agent, "_last_request_facts", None),
        ko_count=getattr(agent, "_live_request_ko_count", 0),
        alert_seen=getattr(agent, "_live_request_alert", False),
    )


def _emit(agent: Any, pending: PendingLiveDecision, observation: Observation, classification: str,
          reason: str | None, discarded: bool) -> None:
    stats: LiveAsyncStats = agent.live_async_stats
    commit_time = game_clock(observation)
    record = {
        "event_type": "live_async_commit",
        "episode_id": getattr(agent, "episode_id", None),
        "actor_id": agent.actor_id,
        "agent_runtime": getattr(agent, "agent_runtime", None) or getattr(agent, "runtime_name", None),
        "decision_id": pending.decision_id,
        "decision_boundary_id": pending.decision_boundary_id,
        "snapshot_id": pending.snapshot_id[:16],
        "request_id": pending.decision_id,
        "action": pending.action_dict.get("action"),
        "target_id": pending.action_dict.get("target_id"),
        "trap_type": pending.action_dict.get("trap_type"),
        "model_output_valid": pending.valid,
        # mission clock readings (seconds remaining); the response is
        # observed on the first frame after it arrives, which is the frame
        # it is committed on, so ready == commit.
        "game_time_convention": "mission_time_remaining",
        "request_start_game_time": pending.request_start_game_time,
        "request_ready_game_time": commit_time,
        "commit_game_time": commit_time,
        "game_time_during_request": (round(pending.request_start_game_time - commit_time, 3)
                                     if pending.request_start_game_time is not None else None),
        # V3 §7: the deadline the model decided on vs. the one it is committed against
        "time_remaining_at_request": pending.request_start_game_time,
        "time_remaining_at_commit": commit_time,
        "world_ticks_during_request": stats.world_ticks_during_request,
        "opponent_moved_during_request": stats.opponent_move_ticks_during_request > 0,
        "actor_moved_during_opponent_request": stats.moved_during_opponent_request_ticks > 0,
        "stale_classification": classification,
        "stale_reason": reason,
        "discarded_due_to_stale": discarded,
    }
    record.update(_decision_facts(agent, pending, discarded))
    stats.moved_during_opponent_request_ticks = 0
    stats.history.append(record)
    del stats.history[:-50]
    telemetry = getattr(agent, "telemetry", None)
    if telemetry is not None:
        telemetry.emit_event(record)


def _decision_facts(agent: Any, pending: PendingLiveDecision, discarded: bool) -> dict[str, Any]:
    """Mission time budget V2 (§28, §31B-H): the time facts the model decided
    on, what the chosen action cost, and the model's own short decision
    summary (`reason`) - display/telemetry only, never read by the engine.
    A fallback WAIT has no model summary: a neutral facts caption is
    recorded instead, labelled as such."""
    from src.mission_time import chosen_action_cost, facts_caption, rationale_fact_check

    facts = pending.facts or {}
    budget = facts.get("time_budget")
    out: dict[str, Any] = {
        "personality": getattr(getattr(agent, "config", None), "personality_profile", None)
        or getattr(agent, "personality_profile", None) or "default",
    }
    if budget is None:
        return out
    action = pending.action_dict
    cost = chosen_action_cost(action, facts) if pending.valid else None
    summary = (action.get("reason") or None) if pending.valid else None
    mismatches = rationale_fact_check(summary, budget)
    out.update({
        "mission_time_limit": budget.get("time_limit_s"),
        "mission_time_elapsed": budget.get("time_used_s"),
        "mission_time_remaining": budget.get("time_remaining_s"),
        "known_exit_eta": (budget.get("exit") or {}).get("eta_s"),
        "known_required_time": budget.get("minimum_known_required_time_s"),
        "slack_s": budget.get("slack_s"),
        "urgency": budget.get("urgency"),
        "chosen_action": action.get("action"),
        "chosen_action_time_cost": cost,
        "chosen_move_eta": cost if action.get("action") == "move" else None,
        "decision_summary": summary,
        "decision_summary_source": "model" if summary else "system_caption",
        "facts_caption": facts_caption(budget, cost),
        "rationale_fact_mismatch": bool(mismatches),
        "rationale_fact_mismatch_detail": mismatches or None,
        "summary_superseded": discarded,
    })
    return out


def _count(stats: LiveAsyncStats, classification: str, reason: str | None) -> None:
    stats.last_classification, stats.last_stale_reason = classification, reason
    if classification == FRESH:
        stats.fresh += 1
    elif classification == STALE_BUT_STILL_LEGAL:
        stats.stale_but_still_legal += 1
    elif classification == DISCARDED_TERMINAL:
        stats.discarded_terminal += 1
    else:
        stats.stale_and_illegal += 1


def _discard(agent: Any, reason_text: str) -> GameAction:
    """Drop a ready decision without applying it and make the actor decide again."""
    stats: LiveAsyncStats = agent.live_async_stats
    stats.discarded_due_to_stale += 1
    agent._memory_tracker.clear_pending_decision()
    # forget the request digest and the throttle: the next frame asks again
    agent._last_observation_digest = None
    agent._next_decision_allowed_at = 0.0
    agent.last_action = GameAction(ActionType.WAIT)
    agent.status = "IDLE"
    agent.last_reason = reason_text
    return agent.last_action


def commit_ready_decision(agent: Any, observation: Observation, legal_actions: list[dict[str, Any]],
                          current_digest: str) -> GameAction:
    """Game thread: classify the ready decision and apply it, or discard it.

    Only called when the agent's response is ready and not in flight (HP09),
    so the executor is idle: requests are launched only at action boundaries
    and a live decision is never started on the background thread."""
    pending: PendingLiveDecision = agent._live_pending
    agent._live_pending = None
    stats: LiveAsyncStats = agent.live_async_stats
    classification, reason = classify_ready_decision(pending, current_digest, observation, legal_actions)
    _count(stats, classification, reason)
    if classification == STALE_AND_ILLEGAL:
        # consumed (HP09: one logical decision -> one consumed response), not handed over
        agent._response_ready_uncommitted = False
        _emit(agent, pending, observation, classification, reason, discarded=True)
        return _discard(agent, f"discarded stale decision ({reason}); deciding again")

    stats.commits += 1
    if pending.valid:
        a = pending.action_dict
        executed, start_boundary, _action_id = agent._executor.start(
            observation, pending.decision_id, a.get("action", "wait"), a.get("target_id"), a.get("trap_type"),
            pending.decision_game_time, **_message_kwargs(agent, a),
        )
        if start_boundary is not None:
            agent._finish_action_lifecycle(observation, start_boundary)
            # The action ended the moment it started (an instant action, or a
            # MOVE to where the actor already is): it may have changed nothing
            # the actor sees, so the digest would never change and the actor
            # would idle - live: a spy with every item stood at the exit for
            # 92 s. Decide again after it, behind the normal throttle.
            agent._last_observation_digest = None
    else:
        executed = GameAction(ActionType.WAIT)
    agent.last_action = executed
    _emit(agent, pending, observation, classification, reason, discarded=False)
    return agent._hand_over(executed)


def _message_kwargs(agent: Any, action_dict: dict[str, Any]) -> dict[str, Any]:
    # Native's executor call passes the message fields; Strands' never did
    # (it has no message schema) - keep each runtime's own call shape.
    extra = {"code": action_dict["code"]} if action_dict.get("code") is not None else {}  # live riddle, both runtimes
    if getattr(agent, "_live_start_with_messages", False):
        return {"message": action_dict.get("message"), "accompanying_message": action_dict.get("accompanying_message"), **extra}
    return extra


IDLE_REDECIDE_SECONDS = 3.0


def idle_redecide(agent: Any, now: float | None = None) -> None:
    """Live async safety net: an actor that has been idle - nothing in
    flight, nothing ready, nothing executing - for IDLE_REDECIDE_SECONDS
    decides again even if its observation did not change.

    `now` defaults to wall time (live async). The simulated-clock lab
    (evaluation timing) passes game time instead, so the opponent's model
    latency - during which the world is frozen - never counts as idling."""
    import time as _time

    busy = agent._thinking_in_progress or agent._response_ready_uncommitted or agent._executor.in_progress
    now = _time.monotonic() if now is None else now
    if busy or agent._last_observation_digest is None:
        agent._live_idle_since = None
        return
    since = getattr(agent, "_live_idle_since", None)
    if since is None:
        agent._live_idle_since = now
    elif now - since >= IDLE_REDECIDE_SECONDS:
        agent._last_observation_digest = None
        agent._live_idle_since = None


def is_terminal(observation: Observation) -> bool:
    """Escaped, timed out (live lifecycle / action budget) or eliminated."""
    return bool(observation.mission_complete or observation.time_expired)


def cancel_reason(observation: Observation) -> str:
    if is_terminal(observation):
        return DISCARDED_TERMINAL
    return "actor_busy_stunned_or_mission_complete"


def cancel_ready_decision(agent: Any, observation: Observation, reason: str) -> None:
    """The actor became busy/stunned/finished while its answer was ready:
    the existing cancel path runs (recorded in ready_responses_cancelled);
    this records the live classification and forgets the digest."""
    pending: PendingLiveDecision | None = agent._live_pending
    if pending is None:
        return
    agent._live_pending = None
    classification = DISCARDED_TERMINAL if reason == DISCARDED_TERMINAL else STALE_AND_ILLEGAL
    _count(agent.live_async_stats, classification, reason)
    _emit(agent, pending, observation, classification, reason, discarded=True)
    # counted in ready_responses_cancelled by the existing path, not in
    # discarded_due_to_stale (the monitor subtracts both from "unused")
    agent._memory_tracker.clear_pending_decision()
    agent._last_observation_digest = None
    agent._next_decision_allowed_at = 0.0
