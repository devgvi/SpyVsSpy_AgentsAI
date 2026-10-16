"""Agent decision memory and strategic context for LLM-driven actors.

M5B: Compact, persistent context derived only from legitimate actor knowledge.
Never reconstructs hidden world truth. Memory updates only from authoritative
engine results after action execution completes, not from model intent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any
from collections import deque


@dataclass
class MissionProgress:
    """What have I accomplished toward the mission?"""
    have: list[str] = field(default_factory=list)  # items acquired
    missing: list[str] = field(default_factory=list)  # items still needed
    ready_to_exit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExplorationMemory:
    """Where have I explored?"""
    visited_rooms: list[str] = field(default_factory=list)  # all visited
    recently_visited: list[str] = field(default_factory=list)  # last N rooms

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnownLock:
    """A lock I've discovered, and whether I can open it now."""
    target_id: str
    lock_type: str  # "key", "keycard", "code"
    requires: str  # item or knowledge id
    satisfied: bool  # do I have it right now?

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LockMemory:
    """What locks block progress, and what satisfies them?"""
    locks: dict[str, KnownLock] = field(default_factory=dict)  # target_id -> KnownLock

    def to_dict(self) -> dict[str, Any]:
        return {
            "locks": {tid: lock.to_dict() for tid, lock in self.locks.items()}
        }


@dataclass
class InteractionRecord:
    """A meaningful interaction outcome."""
    game_time: float
    action: str  # "inspect", "search", "interact", etc.
    target_id: str
    result: str  # semantic result: "empty", "found_item", "locked", "no_change"
    detail: str | None = None  # e.g., "found passport" or "requires archive_key"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureRecord:
    """A recent failure or blocked attempt."""
    game_time: float
    action: str
    target_id: str | None
    reason: str  # category of failure: "missing_item", "locked", "unreachable", etc.
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionRecord:
    """A previous LLM decision and its outcome."""
    game_time: float
    decision_id: str
    selected_action: str
    target_id: str | None
    valid: bool
    action_source: str  # "llm", "fallback", "provisional"
    engine_result: str | None = None  # "success", "blocked", "failed", etc.
    observation_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GameEvent:
    """A meaningful game event the actor is aware of."""
    game_time: float
    event_type: str  # "ITEM_FOUND", "DOOR_UNLOCKED", "DOOR_OPENED", "TRAP_TRIGGERED", etc.
    target_id: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MessageRecord:
    """M5F.2 spec section 7: "the sender may remember what it sent, the
    recipient may remember what it received" - one record either way,
    distinguished by which deque it lives in (`AgentMemory.messages_sent`
    vs `messages_received`), never a shared/merged list. `message_id` for
    a sent message is that decision's own `decision_id` (one message per
    SEND_MESSAGE decision); for a received message it's the engine's real
    `Message.message_id` from `GameState.messages`, used for de-dup."""
    game_time: float
    message_id: str
    sender_actor_id: str
    recipient_actor_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpponentMemory:
    """What do I know about the opponent?"""
    currently_visible: bool = False
    last_seen_room: str | None = None
    last_seen_time: float | None = None
    # Never store: live position, inventory, hidden objectives, etc.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentMemory:
    """
    Bounded, persistent strategic context for a single actor.
    
    Updated only from authoritative engine results after action execution.
    Memory never reconstructs hidden GameState truth.
    All contents are derived from legitimate actor observations.
    """

    MEMORY_SCHEMA_VERSION = "memory-v1"

    def __init__(
        self,
        actor_id: str,
        max_recent_events: int = 10,
        max_recent_failures: int = 15,
        max_recent_decisions: int = 20,
    ):
        self.actor_id = actor_id
        self.max_recent_events = max_recent_events
        self.max_recent_failures = max_recent_failures
        self.max_recent_decisions = max_recent_decisions

        # Persistent strategic facts
        self.mission = MissionProgress()
        self.exploration = ExplorationMemory()
        self.locks = LockMemory()
        self.opponent = OpponentMemory()

        # Bounded recent history (FIFO queue)
        self.recent_interactions: deque[InteractionRecord] = deque()
        self.recent_failures: deque[FailureRecord] = deque()
        self.recent_decisions: deque[DecisionRecord] = deque()
        self.recent_events: deque[GameEvent] = deque()

        # Persistent objects that remain known after interaction
        self.known_empty_objects: set[str] = set()  # objects already searched and empty
        self.known_searched_objects: set[str] = set()  # all searched objects

        # M5F.2: independent per-side communication history - never shared
        # between two actors' AgentMemory instances (each controller owns
        # its own AgentMemory entirely; see game.py's `_build_llm_agent`,
        # which always constructs two fully separate instances). Bounded
        # the same way recent_failures is, not unbounded chat history
        # (spec section 21).
        self.messages_sent: deque[MessageRecord] = deque()
        self.messages_received: deque[MessageRecord] = deque()
        self._received_message_ids: set[str] = set()  # de-dup guard for record_message_received

    def record_mission_progress(self, have: list[str], missing: list[str], ready_to_exit: bool) -> None:
        """Update mission status from authoritative engine knowledge."""
        self.mission.have = have
        self.mission.missing = missing
        self.mission.ready_to_exit = ready_to_exit

    def record_visited_room(self, room_id: str) -> None:
        """Record that the actor entered a room."""
        if room_id not in self.exploration.visited_rooms:
            self.exploration.visited_rooms.append(room_id)
        # Keep recently visited as a rolling window
        self.exploration.recently_visited = list(self.exploration.visited_rooms[-5:])

    def record_known_lock(self, target_id: str, lock_type: str, requires: str, satisfied: bool) -> None:
        """Record that a lock was discovered."""
        self.locks.locks[target_id] = KnownLock(
            target_id=target_id,
            lock_type=lock_type,
            requires=requires,
            satisfied=satisfied,
        )

    def update_lock_satisfied(self, target_id: str, satisfied: bool) -> None:
        """Update whether we now have what a lock requires."""
        if target_id in self.locks.locks:
            self.locks.locks[target_id].satisfied = satisfied

    def record_interaction(
        self,
        game_time: float,
        action: str,
        target_id: str,
        result: str,
        detail: str | None = None,
    ) -> None:
        """Record a meaningful interaction outcome."""
        record = InteractionRecord(
            game_time=game_time,
            action=action,
            target_id=target_id,
            result=result,
            detail=detail,
        )
        self.recent_interactions.append(record)
        if len(self.recent_interactions) > 20:  # Keep some for context
            self.recent_interactions.popleft()

        # Mark persistent facts
        if action == "search":
            self.known_searched_objects.add(target_id)
            if result == "empty":
                self.known_empty_objects.add(target_id)

    def record_failure(
        self,
        game_time: float,
        action: str,
        target_id: str | None,
        reason: str,
        detail: str | None = None,
    ) -> None:
        """Record that an action failed."""
        record = FailureRecord(
            game_time=game_time,
            action=action,
            target_id=target_id,
            reason=reason,
            detail=detail,
        )
        self.recent_failures.append(record)
        if len(self.recent_failures) > self.max_recent_failures:
            self.recent_failures.popleft()

    def record_decision(
        self,
        game_time: float,
        decision_id: str,
        selected_action: str,
        target_id: str | None,
        valid: bool,
        action_source: str,
        observation_hash: str | None = None,
    ) -> None:
        """Record an LLM decision (before engine execution)."""
        record = DecisionRecord(
            game_time=game_time,
            decision_id=decision_id,
            selected_action=selected_action,
            target_id=target_id,
            valid=valid,
            action_source=action_source,
            observation_hash=observation_hash,
        )
        self.recent_decisions.append(record)
        if len(self.recent_decisions) > self.max_recent_decisions:
            self.recent_decisions.popleft()

    def update_decision_result(self, decision_id: str, engine_result: str) -> None:
        """After engine execution, update the decision with its outcome."""
        for record in self.recent_decisions:
            if record.decision_id == decision_id:
                record.engine_result = engine_result
                break

    def record_event(
        self,
        game_time: float,
        event_type: str,
        target_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Record a meaningful game event."""
        record = GameEvent(
            game_time=game_time,
            event_type=event_type,
            target_id=target_id,
            detail=detail,
        )
        self.recent_events.append(record)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events.popleft()

    def record_message_sent(self, game_time: float, message_id: str, recipient_actor_id: str, text: str) -> None:
        """Called once, from `ActionExecutor.finish` (spec section 7: the
        sender's own memory, authoritative-outcome side, same discipline
        as every other memory update - never from raw model intent)."""
        record = MessageRecord(
            game_time=game_time, message_id=message_id, sender_actor_id=self.actor_id,
            recipient_actor_id=recipient_actor_id, text=text or "",
        )
        self.messages_sent.append(record)
        if len(self.messages_sent) > 20:
            self.messages_sent.popleft()

    def record_message_received(self, game_time: float, message_id: str, sender_actor_id: str, text: str) -> None:
        """Called from `AgentMemoryTracker.update_from_observation` for
        every message in this tick's `Observation.messages_received_recently`
        - de-duped by `message_id` since the same message can appear in
        that bounded window across several consecutive ticks (spec section
        20: exactly one memory record per real message, never one per tick
        it happens to still be in view)."""
        if message_id in self._received_message_ids:
            return
        self._received_message_ids.add(message_id)
        record = MessageRecord(
            game_time=game_time, message_id=message_id, sender_actor_id=sender_actor_id,
            recipient_actor_id=self.actor_id, text=text or "",
        )
        self.messages_received.append(record)
        if len(self.messages_received) > 20:
            self.messages_received.popleft()

    def record_opponent_sighting(self, game_time: float, room_id: str) -> None:
        """Opponent was just seen in a room."""
        self.opponent.currently_visible = True
        self.opponent.last_seen_room = room_id
        self.opponent.last_seen_time = game_time

    def record_opponent_left_visibility(self) -> None:
        """Opponent is no longer visible."""
        self.opponent.currently_visible = False
        # Keep last_seen_room and last_seen_time for context

    def get_recent_repetitions(self, window_size: int = 5) -> list[dict[str, Any]]:
        """
        Identify actions that have been repeated recently without state change.
        Helps the model recognize loops.
        """
        repetitions: dict[tuple[str, str | None], int] = {}
        for record in list(self.recent_decisions)[-window_size:]:
            key = (record.selected_action, record.target_id)
            repetitions[key] = repetitions.get(key, 0) + 1

        result = []
        for (action, target_id), count in repetitions.items():
            if count >= 2:
                # Find the most recent occurrence
                last_result = None
                for record in reversed(self.recent_decisions):
                    if record.selected_action == action and record.target_id == target_id:
                        last_result = record.engine_result
                        break
                result.append({
                    "action": action,
                    "target_id": target_id,
                    "count": count,
                    "last_result": last_result,
                })
        return result

    def to_compact_dict(self) -> dict[str, Any]:
        """Serialize to the compact format sent to the LLM."""
        return {
            "schema_version": self.MEMORY_SCHEMA_VERSION,
            "mission": self.mission.to_dict(),
            "exploration": {
                "visited_rooms": self.exploration.visited_rooms,
                "recently_visited": self.exploration.recently_visited,
            },
            "known_locks": [lock.to_dict() for lock in self.locks.locks.values()],
            "known_empty_objects": list(self.known_empty_objects),
            "known_searched_objects": list(self.known_searched_objects),
            "recent_interactions": [r.to_dict() for r in list(self.recent_interactions)[-10:]],
            "recent_failures": [r.to_dict() for r in list(self.recent_failures)[-10:]],
            "recent_events": [e.to_dict() for e in self.recent_events],
            "recent_decisions": [d.to_dict() for d in list(self.recent_decisions)[-10:]],
            "recent_repetitions": self.get_recent_repetitions(),
            "opponent_memory": self.opponent.to_dict(),
            # M5F.2: counts only, not full text - the model already sees
            # its actual recent received messages via Observation's own
            # bounded `communication.messages_received_recently` (spec
            # section 21); duplicating that text here would just double
            # the token cost for no new information. Sent-message counts
            # have no other prompt-visible source at all, so both counts
            # are surfaced here as the "may be summarized through existing
            # AgentMemory" spec section 21 allows for older history.
            "messages_sent_count": len(self.messages_sent),
            "messages_received_count": len(self.messages_received),
        }

    def to_json(self) -> str:
        """Full JSON serialization (may be large; use to_compact_dict for prompt)."""
        data = self.to_compact_dict()
        # Include full history for telemetry/debugging, not for prompt
        data["_full_decisions"] = [d.to_dict() for d in self.recent_decisions]
        data["_full_failures"] = [f.to_dict() for f in self.recent_failures]
        data["_full_events"] = [e.to_dict() for e in self.recent_events]
        return json.dumps(data, separators=(",", ":"))

    def get_memory_size_chars(self) -> int:
        """Compact serialization size for telemetry."""
        return len(json.dumps(self.to_compact_dict(), separators=(",", ":")))

    def to_telemetry_dict(self) -> dict[str, Any]:
        """Metadata for telemetry reporting."""
        return {
            "memory_schema_version": self.MEMORY_SCHEMA_VERSION,
            "memory_size_chars": self.get_memory_size_chars(),
            "visited_rooms_count": len(self.exploration.visited_rooms),
            "known_requirements_count": len(self.locks.locks),
            "known_empty_objects_count": len(self.known_empty_objects),
            "recent_events_count": len(self.recent_events),
            "recent_failures_count": len(self.recent_failures),
            "recent_decisions_count": len(self.recent_decisions),
            "recent_repetitions_count": len(self.get_recent_repetitions()),
            # M5E.1 spec section 7: total record count across every bounded
            # deque (a single number to correlate against instability,
            # independent of memory_size_chars which weights by content
            # length) and whether the opponent is currently a live or
            # ever-seen presence in this actor's own memory - relevant only
            # to the active-opponent condition, always False under a
            # passive/no-opponent run.
            "memory_entries": (
                len(self.recent_interactions) + len(self.recent_failures) + len(self.recent_decisions) + len(self.recent_events)
            ),
            "last_seen_opponent_present": self.opponent.last_seen_room is not None,
        }


class AgentMemoryTracker:
    """M5C.1: wraps an AgentMemory with the M5B/M5B.1 authoritative-outcome
    diffing and pending-decision correlation logic that used to live inside
    LLMController.

    Extracted so every agent runtime adapter (Native, Strands, future ones)
    gets identical memory-update semantics from one implementation instead
    of each reimplementing the Observation diff - M5C.1 spec section 12:
    "M5B memory remains common game-side context." Only ever reads
    Observation, never GameState, same boundary as before.
    """

    def __init__(
        self,
        actor_id: str,
        max_recent_events: int = 10,
        max_recent_failures: int = 15,
        max_recent_decisions: int = 20,
    ) -> None:
        self.memory = AgentMemory(actor_id, max_recent_events, max_recent_failures, max_recent_decisions)
        # None until the first observation is snapshotted, so the first tick
        # never fabricates a "transition" out of thin air.
        self._prev_object_snapshot: dict[str, dict[str, Any]] | None = None
        self._prev_items_found: set[str] = set()
        self._prev_mission_complete: bool = False
        self._prev_visible_other_actor: bool = False
        # The most recent target-directed decision whose engine outcome
        # hasn't been resolved yet. Cleared (or overwritten) as soon as we
        # observe its result, so it never grows unbounded.
        self._pending_decision: dict[str, Any] | None = None

    def update_from_observation(self, observation: Any) -> None:
        """Update persistent strategic context from the authoritative
        observation. Call once per tick, before requesting a new decision."""
        self._record_authoritative_outcomes(observation)

        # Mission progress: what do I have, what am I missing? `ready_to_exit`
        # is "do I currently hold everything required" (has_all_required_items),
        # not `mission_complete` - that only flips after actually exiting, by
        # which point the caller's early-return guard for a completed mission
        # stops calling this at all, so it could never be observed as True here.
        self.memory.record_mission_progress(
            have=list(observation.inventory),
            missing=[item for item in observation.required_items if item not in observation.inventory],
            ready_to_exit=observation.has_all_required_items(),
        )

        self.memory.record_visited_room(observation.current_room)

        # Known locked objects and whether this actor currently satisfies them.
        # `obj.can_unlock` is already the authoritative, engine-computed answer
        # (see observation.py's _lock_knowledge) - never recomputed here.
        # ObservedObject doesn't expose the specific requirement id (only its
        # lock_type category), so `requires` stays "unknown" rather than
        # guessing at a value the actor was never legitimately shown.
        for obj in observation.objects:
            if obj.lock_type:
                self.memory.record_known_lock(
                    target_id=obj.id,
                    lock_type=obj.lock_type,
                    requires="unknown",
                    satisfied=obj.can_unlock,
                )

        if observation.visible_other_actor:
            self.memory.record_opponent_sighting(
                game_time=observation.mission_time_remaining,
                room_id=observation.current_room,
            )
        else:
            self.memory.record_opponent_left_visibility()

        # M5F.2 spec section 7: fold newly-observed received messages into
        # this actor's own memory - `record_message_received`'s own
        # de-dup guard makes this safe to call every tick even though the
        # same message stays in `messages_received_recently`'s bounded
        # window across several consecutive ticks.
        for m in getattr(observation, "messages_received_recently", ()):
            self.memory.record_message_received(
                game_time=m["game_time"], message_id=m["message_id"],
                sender_actor_id=m["sender_actor_id"], text=m["text"],
            )

    def _record_authoritative_outcomes(self, observation: Any) -> None:
        """Diff this tick's Observation against the last one to detect
        engine-validated outcomes (search/inspect resolutions, lock changes,
        doors, mission progress, opponent sightings) and fold them into
        memory. This is the only place memory is updated from action
        *results* - never from what the model asked for.
        """
        game_time = observation.mission_time_remaining

        if self._pending_decision is not None and not self._pending_decision.get("checked"):
            self._pending_decision["checked"] = True
            self._check_pending_decision_failure(observation, game_time)

        if self._prev_object_snapshot is not None:
            for obj in observation.objects:
                prev = self._prev_object_snapshot.get(obj.id)
                if prev is None:
                    continue

                if not prev["searched"] and obj.searched:
                    found = obj.id in observation.items_found_here and obj.id not in self._prev_items_found
                    self.memory.record_interaction(game_time, "search", obj.id, "found_item" if found else "empty")
                    self.memory.record_event(game_time, "ITEM_FOUND" if found else "OBJECT_SEARCHED", obj.id)
                    self._resolve_pending_decision_success(obj.id)

                if not prev["inspected"] and obj.inspected:
                    if obj.trap_status == "detected":
                        self.memory.record_interaction(game_time, "inspect", obj.id, "trap_detected", obj.trap_type)
                        self.memory.record_event(game_time, "TRAP_DETECTED", obj.id, obj.trap_type)
                    else:
                        self.memory.record_interaction(game_time, "inspect", obj.id, "clear")
                    self._resolve_pending_decision_success(obj.id)

                if prev["trap_status"] != "detected" and obj.trap_status == "detected" and obj.trap_is_mine:
                    # Only ever true when this actor's own place_trap call
                    # created the trap engine-side - never the opponent's.
                    self.memory.record_interaction(game_time, "place_trap", obj.id, "trap_placed", obj.trap_type)
                    self.memory.record_event(game_time, "TRAP_PLACED", obj.id, obj.trap_type)
                    self._resolve_pending_decision_success(obj.id)

                if prev["locked"] and not obj.locked:
                    self.memory.record_event(game_time, "DOOR_UNLOCKED" if obj.type == "door" else "LOCK_SATISFIED", obj.id)
                    self.memory.update_lock_satisfied(obj.id, True)

                if prev["door_open"] is False and obj.door_open:
                    self.memory.record_event(game_time, "DOOR_OPENED", obj.id)
                    self._resolve_pending_decision_success(obj.id)

            if not self._prev_mission_complete and observation.mission_complete:
                self.memory.record_event(game_time, "MISSION_OBJECTIVE_COMPLETED")

            if observation.visible_other_actor and not self._prev_visible_other_actor:
                self.memory.record_event(game_time, "OPPONENT_SEEN", detail=observation.current_room)

        self._prev_object_snapshot = {
            obj.id: {
                "searched": obj.searched,
                "inspected": obj.inspected,
                "trap_status": obj.trap_status,
                "locked": obj.locked,
                "door_open": obj.door_open,
            }
            for obj in observation.objects
        }
        self._prev_items_found = set(observation.items_found_here)
        self._prev_mission_complete = observation.mission_complete
        self._prev_visible_other_actor = observation.visible_other_actor

    def _check_pending_decision_failure(self, observation: Any, game_time: float) -> None:
        """One tick after a target-directed decision, check whether the engine
        actually rejected it. Interact/inspect/place_trap/disarm all resolve
        against `current_interaction_target` (proximity-based, exposed as
        `nearby_object_id`) rather than the caller's requested target_id - so
        a request for an out-of-reach target silently no-ops at the engine,
        and a request for a locked target stays locked. Both are legitimate,
        self-knowledge failure reasons; neither reconstructs hidden state.
        """
        pending = self._pending_decision
        requested_target_id = pending.get("requested_target_id")
        resolved_target_id = pending.get("target_id")
        action = pending["action"]

        if requested_target_id is not None and requested_target_id != resolved_target_id:
            self.memory.record_failure(
                game_time=game_time,
                action=action,
                target_id=requested_target_id,
                reason="unreachable",
                detail=f"was near {resolved_target_id}" if resolved_target_id else "was near nothing",
            )
            self.memory.update_decision_result(pending["decision_id"], "blocked_unreachable")
            self.memory.record_event(game_time, "ACTION_FAILED", requested_target_id, "unreachable")
            return

        if resolved_target_id is None:
            return

        obj = observation.get(resolved_target_id)
        if obj is not None and obj.locked and not obj.can_unlock:
            self.memory.record_failure(
                game_time=game_time,
                action=action,
                target_id=resolved_target_id,
                reason="locked",
                detail=f"requires {obj.lock_type or 'unknown'}",
            )
            self.memory.update_decision_result(pending["decision_id"], "blocked_locked")
            self.memory.record_event(game_time, "ACTION_FAILED", resolved_target_id, "locked")

    def _resolve_pending_decision_success(self, target_id: str) -> None:
        pending = self._pending_decision
        if pending is not None and pending.get("target_id") == target_id:
            self.memory.update_decision_result(pending["decision_id"], "success")
            self._pending_decision = None

    def set_pending_decision(
        self,
        decision_id: str,
        normalized_action: str,
        requested_target_id: str | None,
        nearby_object_id: str | None,
    ) -> None:
        """Track a just-issued decision for one-tick-later outcome checking.
        `normalized_action` must already be a canonical action name (e.g.
        from ModelOutputValidation.validated_action["action"]) - this does
        not re-normalize it. Only target-directed actions resolve against
        `nearby_object_id` at the engine (see _check_pending_decision_failure);
        move/wait have no such boolean resolution to diff, so they don't get
        a pending entry.
        """
        if normalized_action in {"interact", "inspect", "place_trap", "disarm"}:
            self._pending_decision = {
                "decision_id": decision_id,
                "action": normalized_action,
                "target_id": nearby_object_id,
                "requested_target_id": requested_target_id,
                "checked": False,
            }
        else:
            self._pending_decision = None

    def clear_pending_decision(self) -> None:
        self._pending_decision = None
