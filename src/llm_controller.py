"""Minimal LLM-backed controller for the White Spy.

This milestone keeps the integration narrow: a controller sees only a serialized
Observation, picks from a legal action surface, and emits a validated GameAction.
It never reads GameState directly and never crashes on malformed output.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, replace as dataclasses_replace
from datetime import datetime, timezone
from math import hypot
from typing import Any
import threading

from src import constants
from src.actions import ActionType, GameAction
from src.agents import live_async
from src.mission_time import annotate_live_payload
from src.encounter_intents import (
    ENCOUNTER_ACTIONS,
    encounter_facts,
    encounter_legal_actions,
    normalize_encounter_action,
)
from src.agents.action_lifecycle import ActionExecutor
from src.llm.agent_memory import AgentMemory, AgentMemoryTracker
from src.llm.clients.base import ModelClient, ModelResponse
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.observation import Observation

PROMPT_VERSION = "m5a-agent-v1"
OBSERVATION_SCHEMA_VERSION = "obs-v1"
ACTION_SCHEMA_VERSION = "action-v1"
MODEL_OBSERVATION_SCHEMA_VERSION = "obs-v2"

DEFAULT_SYSTEM_PROMPT = """You control one spy in a competitive embassy escape game.
Your goal is to complete your mission and escape before the deadline.
Use only information in the supplied observation and memory.
Memory contains your accumulated knowledge: mission progress, visited rooms,
known locked objects, failed attempts, and recent events.
Use this context to avoid repeating actions that already failed and to prioritize
progress toward missing objectives.
Choose exactly one action from legal_actions.
Never invent objects, rooms, items, knowledge, or actions.
Return only structured output matching the required schema.
For a door: INTERACT changes the door's own state (open<->close) and does not
move you; door_open/traversable show whether you can currently walk through it;
interact_effect shows exactly what INTERACT would do right now. To go through
an already-open door, MOVE to a target on the other side - re-interacting with
an open door only closes it again.
""".strip()

# M5D.7: Native representation variants (spec sections 2-5, 12) - the game-
# side legal action *set*, Observation, memory, validator, and repair-v2
# are always identical across variants; only how that legal-action set is
# *presented* in the prompt differs. Both are still ordinary Native
# completion-API JSON output - never real tool calling (spec section 4).
NATIVE_CURRENT_REPRESENTATION = "native_current-v1"
NATIVE_TOOL_LIKE_REPRESENTATION = "native_tool_like-v1"
STRANDS_TOOL_REPRESENTATION = "strands_tool-v1"

NATIVE_TOOL_LIKE_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + """

You are calling one function this turn:

Function: choose_action
Arguments:
  action     - one of the available action names below
  target_id  - the object id to act on (omit for wait)
  trap_type  - which trap to place (place_trap only)
  reason     - a short one-sentence rationale (optional)

The AVAILABLE ACTIONS section lists, for each action name, exactly which
target_id values are currently allowed - the same legal_actions set as
always, just grouped like a function's allowed argument values instead of
a flat list. Choose exactly one available action/target combination.
Return your call as JSON matching the required schema - this is not a
tool-calling API, just ordinary structured output."""
)


def format_legal_actions_tool_like(legal_actions: list[dict[str, Any]], door_info: dict[str, dict[str, Any]] | None = None) -> str:
    """M5D.7 spec sections 4, 17: presents the *same* legal_actions data
    Variant A already has as an API-like block grouped by action name,
    instead of Variant A's compact `{action: [target_id, ...]}` mapping.
    Purely a presentation transform - never adds, removes, or reorders
    which actions/targets are legal; do not call this on anything other
    than the real legal_actions list already computed by
    enumerate_legal_actions.

    `door_info` (M5E.4.2, optional): `{target_id: {"door_open":..,
    "traversable":.., "interact_effect":.., "connects":..}}` for door
    targets - when given, annotates each door target inline (e.g.
    `- archives_door [open, interact_effect=close]` under INTERACT,
    `- archives_door [traversable, connects=hallway<->archives]` under
    MOVE) so the state/effect distinction is visible right where the
    target is offered, not only buried in the room object list above.
    """
    by_action: dict[str, list[str | None]] = {}
    for entry in legal_actions:
        name = _normalize_action_name(entry.get("action"))
        by_action.setdefault(name, []).append(entry.get("target_id"))

    def annotate(action_name: str, target: str) -> str:
        info = (door_info or {}).get(target)
        if not info:
            return f"- {target}"
        if action_name == "interact":
            state = "open" if info.get("door_open") else "closed"
            effect = info.get("interact_effect")
            return f"- {target} [{state}, interact_effect={effect}]"
        if action_name == "move":
            connects = info.get("connects")
            connects_str = f"{connects[0]}<->{connects[1]}" if connects else "unknown"
            traversable = "traversable" if info.get("traversable") else "not traversable yet"
            return f"- {target} [{traversable}, connects={connects_str}]"
        return f"- {target}"

    lines = ["AVAILABLE ACTIONS", ""]
    for name in sorted(by_action):
        targets = [t for t in by_action[name] if t is not None]
        lines.append(f"{name.upper()}(target_id)")
        if targets:
            lines.append("Allowed targets:")
            for t in targets:
                lines.append(annotate(name, t))
        else:
            lines.append("No target required.")
        lines.append("")
    lines.append("Choose exactly one available action.")
    return "\n".join(lines)

ACTION_ALIASES = {
    "wait": "wait",
    "pause": "wait",
    "noop": "wait",
    "move": "move",
    "move_to": "move",
    "go_to": "move",
    "travel": "move",
    "interact": "interact",
    "search": "interact",
    "open": "interact",
    "unlock": "interact",
    "enter_code": "interact",
    "exit": "interact",
    "inspect": "inspect",
    "place_trap": "place_trap",
    "trap": "place_trap",
    "disarm": "disarm",
    "disable": "disarm",
    # M5F.2: target_id carries the recipient actor id, same convention
    # every other targeted action already uses (spec section 3's own
    # sample payload names it `recipient_id`, aliased below).
    "send_message": "send_message",
    "message": "send_message",
    "communicate": "send_message",
}

VALID_ACTION_NAMES = tuple(sorted({v for v in ACTION_ALIASES.values()}))
TARGET_REQUIRED_ACTIONS = {
    "move", "inspect", "interact", "place_trap", "disarm", "open", "unlock", "enter_code", "exit", "send_message",
}
# M5D.4/M5D.5: the action types GameState resolves against
# `current_interaction_target` (proximity), as opposed to MOVE (reachability
# - see enumerate_legal_actions). Used to classify an illegal target as
# specifically "out_of_range" rather than the more generic "illegal_target".
_PROXIMITY_GATED_ACTIONS = {"inspect", "interact", "place_trap", "disarm"}


@dataclass
class LLMControllerConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_retries: int = 1
    fallback_action: str = "wait"
    status: str = "IDLE"
    execution_mode: str = "evaluation"
    prompt_version: str = PROMPT_VERSION
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION
    # M5D.7: which Native representation variant this controller presents
    # legal actions with (spec sections 4-5, 12) - never changes the legal
    # action set itself, only how compact_observation_for_model formats it.
    # A caller selecting native_tool_like-v1 should also pass
    # system_prompt=NATIVE_TOOL_LIKE_SYSTEM_PROMPT explicitly - this stays
    # decoupled/no implicit magic, so both are always visible at the call site.
    representation_variant: str = NATIVE_CURRENT_REPRESENTATION
    # M5E.1: a purely diagnostic tag ("passive"/"active") distinguishing
    # which opponent condition produced this controller's decisions (spec
    # section 5) - never read by any decision/validation logic, only
    # threaded through to telemetry so the long-horizon stability analysis
    # can group decisions by condition without re-deriving it from the
    # episode's variant config after the fact.
    condition: str | None = None
    # M5F.1: which strategic-priority profile (spec section 2's four,
    # plus "default") this controller's prompt carries - validated and
    # rendered once at construction time (see `LLMController.__init__`'s
    # `self._personality_block`), never re-read from here at call time,
    # so a caller mutating `config.personality_profile` after construction
    # has no effect (mirrors `representation_variant`'s own frozen-at-
    # construction discipline). `None` (the default) is `DEFAULT` -
    # byte-identical pre-M5F.1 prompt behavior for every existing caller.
    personality_profile: str | None = None
    # M5F.2 spec section 13: a fixed, per-episode OFF/ON condition - `False`
    # (every pre-M5F.2 caller's implicit value) means SEND_MESSAGE is not
    # offered at all (see `enumerate_legal_actions`) and the communication
    # policy block is never appended to the system prompt (see
    # `LLMController.__init__`'s `self._communication_block`), so OFF-
    # condition behavior is byte-identical to every pre-M5F.2 caller's.
    communication_enabled: bool = False
    # M5F.2.2 spec section 1: which communication cost representation this
    # controller uses - `False` (FULL_COST, every pre-M5F.2.2 caller's
    # implicit value) is byte-identical to M5F.2/M5F.2.1 (standalone
    # SEND_MESSAGE action). `True` (ZERO_DECISION_COST) replaces that with
    # an optional envelope message accompanying any physical action - see
    # `enumerate_legal_actions`, `validate_model_output`'s `zero_decision_cost`
    # param, and `src/llm/communication.py`'s two block variants.
    # Meaningless unless `communication_enabled` is also True.
    zero_decision_cost_messaging: bool = False
    # M5F.2.3 spec section 6: flat top-level `message_to`/`message_text`
    # fields instead of a nested `message` object - model-output
    # reliability only, never a policy change. Meaningless unless
    # `zero_decision_cost_messaging` is also True (spec section 2: this
    # builds on the envelope, never on FULL_COST).
    flat_message_schema: bool = False
    # M5F.2.3 spec section 3: the bounded free-message rule - minimum
    # decisions between sent messages. `0` (default, every pre-M5F.2.3
    # caller's implicit value) means no cooldown - unbounded availability,
    # M5F.2.2's own ZERO_DECISION_COST behavior. `3` is the spec's own
    # preferred value (available, then unavailable for the next 2
    # decisions after a send, then available again). Meaningless unless
    # `zero_decision_cost_messaging` is also True.
    message_cooldown_decisions: int = 0
    # M5F.3A.4: an optional evaluation-only hook, `(dict) -> dict`, applied
    # to `model_payload` (the DISPLAY sent to the model) immediately after
    # `compact_observation_for_model` builds it - never to
    # `observation_payload` (the separate, untouched structure
    # `legal_actions` for actual validation is drawn from - see
    # `_request_action`'s own "keeping the authoritative observation_payload
    # intact for in-process validation" comment). This is what lets
    # `src/evaluation/representation_control.py` reorder or restructure
    # what the model SEES (DECOUPLED's independent legal-actions/room-
    # objects ordering, SINGLE_CANDIDATE_VIEW's compact candidate block)
    # without ever weakening what the engine actually validates against -
    # no new GameAction type, no validation bypass, spec sections 10, 29.
    # `None` (every pre-M5F.3A.4 caller) means no transform - byte-
    # identical prior behavior.
    model_payload_transform: Any = None
    # live play (src/llm/live_payload.py): the payload JSON without spaces
    compact_payload_json: bool = False
    # simulated-clock lab (evaluation timing in LiveGame): decision pacing on the
    # game clock. The live_async idle safety net runs (an idle spy re-decides
    # after 3 s instead of standing still; evaluation never re-asks on an
    # unchanged observation otherwise), and both it and the 0.5 s minimum between
    # two requests count game time, not wall time - with the world frozen while
    # models think, a wall-time throttle let a spy re-decide every frame.
    game_clock_pacing: bool = False
    # M5F.3A.6: an optional evaluation-only hook, `(dict) -> dict`, applied
    # to the model's own raw response dict (after `_normalize_target_id`,
    # before validation) - both on the initial call and, identically, on
    # any repair call. This is the mapping layer
    # `src/evaluation/goal_identity.py` needs: M5F.3A.4/M5F.3A.5 could get
    # away with the model-facing goal id literally BEING the real object
    # id (no remapping needed), but this milestone's cross-mapping design
    # requires genuinely opaque ids (e.g. "k7f3") that carry no inherent
    # connection to any real object - the model's own `target_id="k7f3"`
    # is not a real, legal engine id at all, so it must be rewritten to
    # the real object id THIS TRIAL's mapping currently assigns before
    # `validate_model_output` (which checks against the true, unmodified
    # Observation/legal_actions - never weakened by this hook) ever sees
    # it. `None` (every pre-M5F.3A.6 caller) means no transform - byte-
    # identical prior behavior.
    model_response_transform: Any = None
    # M6G.3 (HP09 - one logical decision, one consumed response): while a
    # completed decision has not been handed to the game loop yet, never
    # launch a replacement request - hand the ready decision over instead
    # and re-evaluate the changed observation on the next call. `False` is
    # the pre-M6G.3 lifecycle exactly (Game(commit_policy="answer_order")).
    ready_response_guard: bool = True
    # Bugfix (navigation/exploration): the live-state representation -
    # reachable-only MOVE targets and the structured `exploration` section
    # (see serialize_observation). `False` reproduces the payload every
    # M5/M6 experiment was recorded with.
    live_state_representation: bool = True
    # Bugfix mission time budget V2: add `time_budget` + `action_costs` to the
    # payload (src/mission_time.py) and the live mission directive to the
    # system prompt. Off by default; the workshop LiveGame turns both on.
    live_time_facts: bool = False
    live_prompt_addendum: str | None = None


def _direction_to_target(observation: Observation, target_id: str) -> tuple[float, float]:
    target = observation.get(target_id)
    if target is None:
        return (0.0, 0.0)
    dx = target.position[0] - observation.position[0]
    dy = target.position[1] - observation.position[1]
    magnitude = hypot(dx, dy)
    if magnitude == 0:
        return (0.0, 0.0)
    return (dx / magnitude, dy / magnitude)


def _normalize_action_name(value: Any) -> str:
    if value is None:
        return "wait"
    text = str(value).strip().lower()
    return ACTION_ALIASES.get(text, text.replace("-", "_").replace(" ", "_"))


def _normalize_target_id(response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        return response
    normalized = dict(response)
    target_id = normalized.get("target_id")
    object_id = normalized.get("object_id")
    # M5F.2: spec section 3's own canonical SEND_MESSAGE example names the
    # field `recipient_id` - accepted as an alias for target_id, same
    # pattern object_id already used, so the rest of the pipeline (legal-
    # action checking, ActionExecutor, telemetry) needs no send_message-
    # specific target field at all. Kept exactly as conditional as the
    # pre-existing object_id fallback (only ever *sets* target_id, never
    # unconditionally re-writes it) - a response with none of these three
    # keys at all must still come out with no "target_id" key present, not
    # a newly-fabricated `None` one (several tests assert exact raw-
    # response dict shape, e.g. `== {}`).
    recipient_id = normalized.get("recipient_id")
    if target_id is None and object_id is not None:
        normalized["target_id"] = object_id
    elif target_id is None and recipient_id is not None:
        normalized["target_id"] = recipient_id
    normalized.pop("object_id", None)
    normalized.pop("recipient_id", None)
    return normalized


FUNCTION_ENVELOPE_NAMES = ("choose_action",)


def unwrap_function_call_envelope(response: Any) -> Any:
    """Live feedback: the tool-like prompt ("You are calling one function
    this turn: choose_action ...") made the model answer
    {"action": "choose_action", "arguments": {...}} (or {"name": ...,
    "arguments": ...}) instead of the arguments themselves - a repair or a
    fallback each time. This returns the arguments. Set as the live
    workshop's model_response_transform only (recorded runs keep their
    exact validation)."""
    if not isinstance(response, dict):
        return response
    name = response.get("action") or response.get("name") or response.get("function")
    inner = response.get("arguments") or response.get("parameters") or response.get("args")
    if name in FUNCTION_ENVELOPE_NAMES and isinstance(inner, dict):
        return _normalize_target_id(dict(inner))
    if len(response) == 1 and isinstance(response.get("choose_action"), dict):
        return _normalize_target_id(dict(response["choose_action"]))
    return response


def _action_requires_target(action_name: str) -> bool:
    return _normalize_action_name(action_name) in TARGET_REQUIRED_ACTIONS


def convert_response_to_action(
    observation: Observation,
    response: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> GameAction | None:
    """Convert an already-plausible canonical response into a typed
    GameAction, checking it against the current legal-action surface.

    Module-level (not a method) since it needs no controller state - M5C.1
    reuses this directly so every runtime adapter (Native, Strands, future
    ones) validates against the exact same rules rather than each
    reimplementing action/target legality (see src/agents/strands_agent.py).
    """
    if not isinstance(response, dict):
        return None

    response = _normalize_target_id(response)
    legal_by_action: dict[str, set[str]] = {}
    for entry in legal_actions:
        action_name = _normalize_action_name(entry.get("action", "wait"))
        target_id = entry.get("target_id")
        if target_id is not None:
            legal_by_action.setdefault(action_name, set()).add(target_id)

    normalized = _normalize_action_name(response.get("action", "wait"))
    target_id = response.get("target_id")
    trap_type = response.get("trap_type")

    if normalized == "wait":
        return GameAction(ActionType.WAIT)

    if getattr(observation, "encounters_enabled", False) and normalize_encounter_action(normalized) in ENCOUNTER_ACTIONS:
        # the executor turns the intent into movement (encounter_intents)
        normalized = normalize_encounter_action(normalized)
        if target_id is None or target_id not in legal_by_action.get(normalized, set()):
            return None
        return GameAction(ActionType.MOVE, target_id=target_id)

    if normalized == "enter_safe_code":
        code = response.get("code")
        if target_id is None or target_id not in legal_by_action.get("enter_safe_code", set()) or not _valid_code(code):
            return None
        return GameAction(ActionType.INTERACT, target_id=target_id, code=str(code).strip())
    if normalized in {"interact", "search", "open", "unlock", "enter_code", "exit"}:
        normalized = "interact"

    if _action_requires_target(normalized):
        if target_id is None:
            return None
        if normalized == "move":
            if target_id not in legal_by_action.get("move", set()):
                return None
            return GameAction(ActionType.MOVE, direction=_direction_to_target(observation, target_id))
        if normalized == "inspect":
            if target_id not in legal_by_action.get("inspect", set()):
                return None
            return GameAction(ActionType.INSPECT, target_id=target_id)
        if normalized == "interact":
            if target_id not in legal_by_action.get("interact", set()):
                return None
            return GameAction(ActionType.INTERACT, target_id=target_id)
        if normalized == "place_trap":
            legal_targets = legal_by_action.get("place_trap", set())
            if target_id not in legal_targets:
                return None
            trap_name = trap_type or "bomb"
            return GameAction(ActionType.PLACE_TRAP, target_id=target_id, trap_type=trap_name)
        if normalized == "disarm":
            legal_targets = legal_by_action.get("disarm", set())
            if target_id not in legal_targets:
                return None
            return GameAction(ActionType.DISARM, target_id=target_id)
        if normalized == "send_message":
            # Recipient legality (real actor / not self / communication
            # actually enabled) is entirely enforced by legal_by_action's
            # own membership check here - `enumerate_legal_actions` only
            # ever offers the one legal recipient (spec section 39's
            # "invalid recipient"/"self recipient" cases both fail here,
            # naturally, with no send_message-specific logic needed).
            # Message-content validity (non-empty, length) is checked
            # earlier by `validate_model_output` before this function is
            # even called - see its dedicated missing_message/
            # oversized_message categories.
            if target_id not in legal_by_action.get("send_message", set()):
                return None
            message_text = response.get("message")
            if not isinstance(message_text, str) or not message_text.strip():
                return None
            return GameAction(
                ActionType.SEND_MESSAGE, target_id=target_id, message=message_text.strip()[: constants.MESSAGE_MAX_LENGTH]
            )

    if normalized == "move":
        direction = response.get("direction")
        if isinstance(direction, (list, tuple)) and len(direction) == 2:
            return GameAction(ActionType.MOVE, direction=tuple(float(v) for v in direction))
        return None

    return None


_ENTER_CODE_ALIASES = {"enter_safe_code", "enter_code", "crack_safe", "enter_combination", "safe_code"}


def _valid_code(code: Any) -> bool:
    """A safe code is 1-8 digits (given as a string or a number)."""
    text = str(code).strip() if isinstance(code, (str, int)) and not isinstance(code, bool) else ""
    return text.isdigit() and 1 <= len(text) <= 8


def enumerate_legal_actions(observation: Observation, reachable_only: bool = False,
                            known_trap_blocks_place_trap: bool = False,
                            encounter_actions: bool = False,
                            inspect_trappable_only: bool = False) -> list[dict[str, Any]]:
    """Build a compact legal surface from the observation-only state.

    `reachable_only` (bugfix navigation/exploration §10): offer MOVE only to
    targets the actor can actually walk to right now. The live controllers
    turn it on (see `serialize_observation(live_state=True)`); the default
    keeps the legal surface every stored M5/M6 snapshot was recorded with,
    so offline replays of that evidence stay byte-identical."""

    actions: list[dict[str, Any]] = [{"action": "wait"}]
    seen: set[tuple[str, str | None]] = set()

    def add(action: str, target_id: str | None = None) -> None:
        key = (action, target_id)
        if key in seen:
            return
        seen.add(key)
        entry: dict[str, Any] = {"action": action}
        if target_id is not None:
            entry["target_id"] = target_id
        actions.append(entry)

    has_any_trap = sum(observation.trap_inventory.values()) > 0
    nearby_obj = observation.get(observation.nearby_object_id) if observation.nearby_object_id else None

    for obj in observation.objects:
        # Bugfix (navigation/exploration §10): "legal" MOVE means the target
        # can actually be walked to right now. Before, every object in the
        # level was offered, including furniture behind a locked or closed
        # door, and the MOVE then failed with PATH_UNREACHABLE (21% of the
        # MOVEs in the M7/M7.1 live runs) - known_open_questions.md #2. The
        # check uses only the walkable grid (walls, furniture, door states:
        # geometry the actor can see), never hidden state.
        if reachable_only and (obj.interactable or obj.type == "door") and not move_target_reachable(observation, obj):
            continue
        if obj.interactable:
            # M5D.2: a genuine second root cause behind M5D.1's "no actor
            # ever left the starting room or successfully inspected/
            # searched anything" - MOVE was legal only toward doors, so
            # the model had no legal way to ever request walking up to a
            # desk/safe/cabinet, even once decision-thrashing was fixed.
            # _direction_to_target already worked for any target_id; only
            # this legality gate was door-only (spec sections 6-7, 16).
            add("move", obj.id)
        if obj.type == "door":
            add("move", obj.id)

    # M5D.4: recognized != legal. INSPECT/INTERACT/PLACE_TRAP/DISARM all
    # resolve engine-side against `current_interaction_target` - the one
    # object (if any) currently within INTERACTION_DISTANCE, exposed here
    # as `nearby_object_id` - never against whatever target_id the model
    # names (see GameState._target_object). Before this fix, every
    # interactable object in the room was offered as "legal" regardless of
    # range, so the model had no way to tell which of them it could
    # actually act on right now: a live-telemetry audit of M5D.3's 28
    # attempted interactions found 27/28 (96%) were exactly this - a
    # legal-but-out-of-range target that silently no-opped (see
    # CHANGELOG.md's M5D.4 entry). Offering only the real nearby target (at
    # most one, matching the engine's own single-target design) makes
    # `legal_actions` mean "can meaningfully attempt now," not merely
    # "a recognized action naming a real object somewhere."
    if nearby_obj is not None:
        # Mission time budget V2 (found live): the engine only inspects
        # trappable objects (GameState._do_inspect); INSPECT on a door was
        # offered, did nothing, left the observation unchanged, and the spy
        # stood idle waiting for a change - 22 s of mission time in one run.
        # The live surface now matches the engine; the recorded one is kept.
        if nearby_obj.trappable or not inspect_trappable_only:
            add("inspect", nearby_obj.id)

        # INTERACT covers search/open/unlock/exit depending on object type
        # (see GameState._interact) - a lock the actor can't currently
        # satisfy is a legitimate, actor-known reason to withhold it
        # (obj.locked/obj.can_unlock are the actor's own knowledge, never
        # hidden state - spec section 9/29), not a hidden engine rejection.
        if nearby_obj.lock_type is None or not nearby_obj.locked or nearby_obj.can_unlock:
            add("interact", nearby_obj.id)

        # PLACE_TRAP requires a prior INSPECT of this exact object
        # (GameState._place_trap: "Inspect the {label} before trapping
        # it.") and at least one trap in inventory - both are the actor's
        # own, legitimately-known state, not hidden truth. Whether a
        # *hidden* trap already occupies the object is not checked here on
        # purpose - the actor cannot legitimately know that in advance, so
        # the engine remains the final authority on that specific case
        # (spec section 7: never leak an undetected blocker).
        # Bugfix live flow (Part C, found in the SABOTEUR-vs-RACER live run):
        # a trap the actor already KNOWS is there - its own, or a detected
        # enemy one (trap_status "detected") - makes PLACE_TRAP certain to
        # fail engine-side ("already has a trap on it"), yet it was offered,
        # and the saboteur tried to "stack" a second trap on its own. That is
        # the actor's own knowledge, so the live surface withholds it; a
        # hidden enemy trap is still never revealed here (see above). Off by
        # default: the recorded M5/M6 surface stays byte-identical.
        known_trap = known_trap_blocks_place_trap and nearby_obj.trap_status == "detected"
        # V3 (live, own traps dangerous): search first, trap after. Trapping
        # an object you still have to search means disarming your own trap
        # (the reported PLACE_TRAP -> DISARM -> SEARCH) or setting it off.
        own_rules = getattr(observation, "own_traps_dangerous", False)
        unsearched_blocks = own_rules and not nearby_obj.searched
        # live trap rules (GameState._place_trap): a search counts as knowing
        # the object - no inspection needed before trapping it
        knows_object = nearby_obj.inspected or (getattr(observation, "trap_after_search", False)
                                                and nearby_obj.searched)
        if nearby_obj.trappable and knows_object and has_any_trap and not known_trap and not unsearched_blocks:
            add("place_trap", nearby_obj.id)

        if nearby_obj.trap_status == "detected" and not (own_rules and own_trap_just_placed(observation, nearby_obj.id)):
            add("disarm", nearby_obj.id)

        # live riddle: the safe takes a code the spy works out itself
        puzzle = getattr(observation, "puzzle", None)
        if puzzle and puzzle.get("type") == "riddle" and nearby_obj.id == "ambassador_safe" \
                and not puzzle.get("safe_code_known"):
            add("enter_safe_code", nearby_obj.id)

    exit_obj = observation.get("exit_door")
    if exit_obj is not None and (not reachable_only or move_target_reachable(observation, exit_obj)):
        add("move", "exit_door")

    # M5F.2 spec section 13: SEND_MESSAGE exists in legal_actions at all
    # only when this episode's communication condition is ON - the OFF
    # control genuinely omits the action (never merely "legal but
    # discouraged"). The only legal recipient is the actor's one opponent
    # (`other_actor_id` - always populated, legitimate self-knowledge, spec
    # section 3's "recipient exists"/"sender != recipient" are both
    # trivially satisfied by construction here, never by a runtime check).
    # M5F.2.2 spec section 2: under ZERO_DECISION_COST, SEND_MESSAGE stops
    # being offered as a standalone action at all - the envelope's own
    # optional `message` field (handled entirely in `validate_model_output`/
    # the model-facing schema, never in this legal-action set) replaces it
    # for that condition. FULL_COST (the default) is completely unaffected.
    if observation.communication_enabled and not observation.zero_decision_cost_messaging:
        add("send_message", observation.other_actor_id)
    # Bugfix live flow V2 (Part D): approach/avoid the opponent - offered only
    # on the live surface, only when encounters are a rule of the match and
    # the opponent is legitimately visible (src/encounter_intents.py).
    if encounter_actions:
        for entry in encounter_legal_actions(observation):
            add(entry["action"], entry["target_id"])
    return actions


def own_trap_just_placed(observation: Observation, object_id: str) -> bool:
    """V3 SELF_REVERSAL_NO_STATE_CHANGE guard: the actor placed its own trap
    on this object and has not left it since - disarming it now would only
    undo what it just did. Once it walks away, DISARM is offered again."""
    for record in getattr(observation, "self_placed_traps", ()) or ():
        if record.get("target_id") == object_id and record.get("armed") and record.get("fresh"):
            return True
    return False


def own_trap_facts(observation: Observation) -> list[dict[str, Any]]:
    """V3: the model-facing own-trap memory (live payloads only)."""
    return [
        {"target_id": r["target_id"], "trap_type": r["trap_type"], "room": r.get("room"), "status": r["status"]}
        for r in getattr(observation, "self_placed_traps", ()) or ()
    ]


def _door_affordance(obj: Any) -> dict[str, Any]:
    """M5E.4.2: derived, door-only fields making INTERACT's current effect
    and MOVE_TO's traversability explicit to the model - computed purely
    from `door_open`/`locked`/`can_unlock`/`connects` (all already actor-
    legitimate, spec section 14), never a new stored field or new engine
    rule. Root cause this fixes (M5E.4.1's traced finding): nothing
    previously told the model that INTERACT on an open door doesn't move
    it anywhere, or what room lies beyond a door it just opened - `{}` for
    every non-door object, so callers can `**`-merge this unconditionally."""
    if obj.type != "door":
        return {}
    if obj.locked and not obj.can_unlock:
        # INTERACT isn't offered as legal here at all (see
        # enumerate_legal_actions's lock gate) - no effect to describe.
        interact_effect = None
    elif obj.locked and obj.can_unlock:
        interact_effect = "unlock_and_open"
    elif obj.door_open:
        interact_effect = "close"
    else:
        interact_effect = "open"
    return {
        "connects": list(obj.connects) if obj.connects else None,
        "traversable": bool(obj.door_open) if obj.door_open is not None else None,
        "interact_effect": interact_effect,
    }


def move_target_reachable(observation: Observation, obj: Any) -> bool:
    """Bugfix (navigation/exploration §10): can a MOVE to `obj` actually get
    somewhere useful from here? True when the actor is already in range, when
    an open door's crossing point is reachable, or when some cell within
    interaction range of the object is connected to the actor on the walkable
    grid. Observations without a real NavGrid (test doubles, `nav_grid=None`)
    keep the old behaviour: every target is offered."""
    grid = observation.nav_grid
    if grid is None or not hasattr(grid, "has_reachable_approach"):
        return True
    if observation.nearby_object_id == obj.id:
        return True
    waypoint = getattr(obj, "crossing_waypoint", None)
    if waypoint is not None:
        cell = grid.nearest_walkable(*waypoint)
        if cell is not None and cell in grid.reachable_cells(observation.position):
            return True
    return grid.has_reachable_approach(obj.collision_rect, observation.position)


def forget_digest_after_unreachable_move(agent: Any, boundary: str) -> None:
    """M7.1 liveness fix, shared by Native and Strands.

    A MOVE that fails at start (PATH_UNREACHABLE - the known "no-path MOVE
    offered as legal" gap, docs/known_open_questions.md #2) leaves the
    actor exactly where it was, so its next Observation has the same
    digest as the one that produced the failed decision and get_action's
    digest dedup never asks the model again. Seen live in M7.1: both spies
    idle for 164 s of game time until the deadline. Forgetting the digest
    lets the next poll decide again, still behind the normal decision
    throttle and the HP09 ready-response guard; the failure is already in
    memory (ActionExecutor.finish -> record_failure).

    Bugfix (navigation/exploration): applies to every failed MOVE boundary
    (PATH_UNREACHABLE, APPROACH_INCOMPLETE, MOVE_STALLED) - each can leave
    the observation unchanged in the same way."""
    from src.agents.action_lifecycle import FAILED_MOVE_BOUNDARIES

    if boundary in FAILED_MOVE_BOUNDARIES:
        agent._last_observation_digest = None


LIVE_STATE_REPRESENTATION_VERSION = "live-state-v1"


def serialize_observation(observation: Observation, live_state: bool = False, live_workshop: bool = False) -> dict[str, Any]:
    """Serialize only the facts the agent is legally allowed to see.

    `live_state=True` is the representation the live controllers send
    (Native `LLMController` and `StrandsAgentAdapter`, both through this
    one function, so they stay identical): MOVE offered only to reachable
    targets, plus the structured `exploration` section (bugfix
    navigation/exploration). `live_state=False` (the default) is the
    representation every stored M5/M6 snapshot and workshop fixture was
    recorded with - the offline replay and evidence-rebuild code keeps
    calling it that way, so their hashes still reproduce."""

    room_objects = [
        {
            "id": obj.id,
            "type": obj.type,
            "label": obj.label,
            "position": list(obj.position),
            "interactable": obj.interactable,
            "trappable": obj.trappable,
            "searched": obj.searched,
            "inspected": obj.inspected,
            "trap_status": obj.trap_status,
            "trap_type": obj.trap_type,
            "door_open": obj.door_open,
            "locked": obj.locked,
            "can_unlock": obj.can_unlock,
            **_door_affordance(obj),
        }
        for obj in observation.objects
    ]

    relevant_objects = []
    for obj in observation.objects:
        if obj.id == observation.nearby_object_id:
            relevant_objects.append(obj)
        elif obj.type == "door" or obj.id == "exit_door":
            relevant_objects.append(obj)
        elif obj.can_unlock or obj.locked:
            relevant_objects.append(obj)
        elif obj.id in observation.known_locked_objects:
            relevant_objects.append(obj)
    if not relevant_objects:
        relevant_objects = sorted(observation.objects, key=lambda obj: hypot(obj.position[0] - observation.position[0], obj.position[1] - observation.position[1]))[:10]

    compact_room_objects = [
        {
            "id": obj.id,
            "type": obj.type,
            "label": obj.label,
            "position": list(obj.position),
            "interactable": obj.interactable,
            "trappable": obj.trappable,
            "searched": obj.searched,
            "inspected": obj.inspected,
            "trap_status": obj.trap_status,
            "door_open": obj.door_open,
            "locked": obj.locked,
            "can_unlock": obj.can_unlock,
            **_door_affordance(obj),
            # live feedback: a saboteur trapped an object, then disarmed its
            # own trap before searching it - it could not tell its own trap
            # from an enemy one ("detected" for both). Own traps never
            # trigger on their owner. Live payloads only.
            **({"trap_is_mine": obj.trap_is_mine} if live_state and obj.trap_status == "detected" else {}),
        }
        for obj in relevant_objects
    ]
    if live_state and getattr(observation, "own_traps_dangerous", False):
        # V3: your own armed trap is a hazard to you too - say so explicitly
        by_id = {o.id: o for o in relevant_objects}
        for entry in compact_room_objects:
            obj = by_id.get(entry["id"])
            if obj is not None and obj.trap_status == "detected" and obj.trap_is_mine:
                entry["trap_status"] = constants.OWN_TRAP_ARMED

    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "self": {
            "actor_id": observation.actor_id,
            "room": observation.current_room,
            "position": list(observation.position),
            "facing": observation.facing,
            "inventory": sorted(observation.inventory),
            "knowledge": sorted(observation.knowledge),
            "status": {"stunned": observation.stunned, "busy": observation.busy},
            # M5D.4: the one object (if any) currently within interaction
            # range - GameState.current_interaction_target's own value,
            # already authoritative and actor-legitimate (not hidden
            # state). Previously never surfaced at all: the model had to
            # guess proximity from raw positions, with no way to confirm it
            # was actually near what it intended to inspect/search before
            # this milestone tied legal_actions to this same field.
            "near_object_id": observation.nearby_object_id,
            # Bugfix live flow (Part C §33-34): the actor's own trap loadout.
            # PLACE_TRAP needs a trap_type the actor holds, but before this
            # the model only ever learned which types it held from a repair
            # hint after a failed attempt. Live payloads only, so every
            # recorded payload stays byte-identical.
            **({"trap_inventory": dict(sorted(observation.trap_inventory.items()))} if live_state else {}),
            # live feedback: lives (3) and how often a trap went off under you
            **({"lives": observation.lives, "traps_triggered_on_you": observation.trap_hits}
               if live_state and observation.lives is not None else {}),
            # V3: the traps you placed yourself (they can go off under you too)
            **({"own_traps": own_trap_facts(observation)}
               if live_state and getattr(observation, "own_traps_dangerous", False) else {}),
            **({"lifecycle": observation.lifecycle}
               if live_state and getattr(observation, "own_traps_dangerous", False) else {}),
        },
        "time": {
            "elapsed": max(0.0, observation.mission_time_remaining),
            "remaining": observation.mission_time_remaining,
            "deadline": observation.mission_time_remaining,
        },
        "room": {"id": observation.current_room, "objects": compact_room_objects},
        "memory": {
            "visited_rooms": sorted(observation.known_rooms),
            "known_locked_objects": dict(sorted(observation.known_locked_objects.items())),
        },
        "opponent": {"visible": observation.visible_other_actor, "last_seen": observation.last_seen_opponent},
        # M5F.2 spec sections 7-8, 36: a structurally distinct key, never
        # merged into "opponent" or "self" - `messages_received_recently`
        # are the other actor's own unverified claims (spec section 8),
        # each entry self-labeled "sender": "opponent" so the model sees
        # exactly where this text came from without it ever being spliced
        # into anything resembling an instruction channel. Empty list
        # (never omitted) when communication is off or nothing has arrived
        # yet - never a fabricated example message.
        "communication": {
            "enabled": observation.communication_enabled,
            "recipient_id": observation.other_actor_id if observation.communication_enabled else None,
            "messages_received_recently": [
                {"sender": "opponent", "message_id": m["message_id"], "text": m["text"], "game_time": m["game_time"]}
                for m in observation.messages_received_recently
            ],
        },
        "mission": {
            "required_items": list(observation.required_items),
            "have": sorted(observation.inventory),
            "ready_to_exit": observation.has_all_required_items(),
            # live one-copy rule: what this spy knows about the contested items
            **({"one_copy_of_each_required_item": True,
                "opponent_holds": list(getattr(observation, "opponent_holds", ()))}
               if live_state and getattr(observation, "single_copy_items", False) else {}),
        },
        "legal_actions": enumerate_legal_actions(observation, reachable_only=live_state,
                                                 known_trap_blocks_place_trap=live_state,
                                                 encounter_actions=live_state,
                                                 # workshop live game only (LiveGame sets live_time_facts)
                                                 inspect_trappable_only=live_workshop),
        # Bugfix live flow V2 (Part D): the encounter rule + visible-opponent
        # facts, live payloads of encounter matches only.
        **(
            {"encounter": encounter_facts(observation)}
            if live_state and encounter_facts(observation) is not None
            else {}
        ),
        # live safe puzzle (src/puzzles.py): halves / clue / inscriptions read
        **({"puzzle": observation.puzzle}
           if live_state and getattr(observation, "puzzle", None) is not None else {}),
        # Bugfix (exploration state): the one canonical place for per-room
        # exploration facts (observation.build_exploration) - live payloads only.
        **(
            {"exploration": observation.exploration}
            if live_state and getattr(observation, "exploration", None) is not None
            else {}
        ),
    }


def compact_exploration(exploration: dict[str, Any]) -> dict[str, Any]:
    """Model-facing form of `build_exploration`: the same facts, empty lists
    and maps dropped, keys sorted. Shared by Native and Strands (both call
    `compact_observation_for_model`), so the two runtimes see identical bytes."""
    rooms = {}
    for room_id in sorted(exploration.get("rooms", {})):
        room = exploration["rooms"][room_id]
        rooms[room_id] = {key: room[key] for key in sorted(room) if key == "status" or room[key]}
    return {"current_room": exploration.get("current_room"), "rooms": rooms}


def compact_observation_for_model(
    observation_payload: dict[str, Any],
    representation_variant: str = NATIVE_CURRENT_REPRESENTATION,
) -> dict[str, Any]:
    """Produce a compact, provider-neutral observation optimized for model input.

    This keeps all legally relevant facts but reduces verbosity and numeric
    precision, and converts `legal_actions` to a compact mapping.

    `representation_variant` (M5D.7 spec sections 4-5): controls *only* how
    the already-computed `legal_actions` list is presented below - the
    underlying legal-action set, Observation content, and everything else
    in this payload is identical regardless of variant. Default
    (`native_current-v1`) is Variant A's existing compact JSON mapping,
    unchanged. `native_tool_like-v1` (Variant B) additionally attaches
    `available_actions_text`, the same data reformatted as an API-like
    block (`format_legal_actions_tool_like`) - still ordinary JSON output,
    never real tool calling.
    """
    def round_pos(p):
        try:
            return [int(round(float(p[0]))), int(round(float(p[1])))]
        except Exception:
            return p

    compact = {
        "schema_version": MODEL_OBSERVATION_SCHEMA_VERSION,
        "self": {
            "actor_id": observation_payload["self"]["actor_id"],
            "room": observation_payload["self"]["room"],
            "position": round_pos(observation_payload["self"]["position"]),
            "facing": observation_payload["self"]["facing"],
            "inventory": observation_payload["self"]["inventory"],
            "status": observation_payload["self"]["status"],
            # M5D.4: must survive compaction - it's the one fact that makes
            # legal_actions' proximity gating legible to the model (see
            # serialize_observation).
            "near_object_id": observation_payload["self"].get("near_object_id"),
            # live payloads only (see serialize_observation)
            **({"trap_inventory": observation_payload["self"]["trap_inventory"]}
               if "trap_inventory" in observation_payload["self"] else {}),
            **{k: observation_payload["self"][k] for k in ("lives", "traps_triggered_on_you", "own_traps", "lifecycle")
               if k in observation_payload["self"]},
        },
        "time": {"remaining": int(round(float(observation_payload.get("time", {}).get("remaining", 0))))},
        "room": {"id": observation_payload.get("room", {}).get("id"), "objects": []},
        "memory": {"visited_rooms": observation_payload.get("memory", {}).get("visited_rooms", []), "known_locked_objects": list(observation_payload.get("memory", {}).get("known_locked_objects", {}).keys())},
        "opponent": observation_payload.get("opponent", {}),
        **({"encounter": observation_payload["encounter"]} if "encounter" in observation_payload else {}),
        **({"puzzle": observation_payload["puzzle"]} if "puzzle" in observation_payload else {}),
        # mission time budget V2: live workshop requests only (mission_time.annotate_live_payload)
        **{key: observation_payload[key] for key in ("time_budget", "action_costs") if key in observation_payload},
        "mission": observation_payload.get("mission", {}),
        # M5F.2: already small/bounded (<= RECENT_MESSAGES_LIMIT entries) -
        # passed through unmodified rather than re-compacted.
        "communication": observation_payload.get("communication", {"enabled": False, "recipient_id": None, "messages_received_recently": []}),
    }

    # compact room objects: keep only essential flags and rounded positions
    for obj in observation_payload.get("room", {}).get("objects", []):
        compact_obj = {
            "id": obj.get("id"),
            "type": obj.get("type"),
            "position": round_pos(obj.get("position", [0, 0])),
        }
        for k in (
            "interactable", "trappable", "searched", "inspected", "trap_status", "trap_is_mine", "door_open", "locked", "can_unlock",
            "connects", "traversable", "interact_effect",  # M5E.4.2
        ):
            if k in obj:
                compact_obj[k] = obj.get(k)
        compact["room"]["objects"].append(compact_obj)

    # compact legal_actions into mapping action -> [targets]
    legal = observation_payload.get("legal_actions", [])
    compact_legal: dict[str, list[str]] = {}
    for entry in legal:
        name = _normalize_action_name(entry.get("action"))
        tgt = entry.get("target_id")
        compact_legal.setdefault(name, [])
        if tgt is not None:
            compact_legal[name].append(tgt)
    compact["legal_actions_compact"] = compact_legal
    compact["legal_actions_count"] = len(legal)
    if observation_payload.get("exploration") is not None:
        compact["exploration"] = compact_exploration(observation_payload["exploration"])
    if representation_variant == NATIVE_TOOL_LIKE_REPRESENTATION:
        door_info = {
            obj["id"]: obj
            for obj in compact["room"]["objects"]
            if obj.get("type") == "door"
        }
        compact["available_actions_text"] = format_legal_actions_tool_like(legal, door_info=door_info)

    return compact


def _coerce_model_response(response: Any, provider_name: str, model_name: str) -> ModelResponse:
    if isinstance(response, ModelResponse):
        return response
    if isinstance(response, dict):
        payload = response.get("payload") if isinstance(response.get("payload"), dict) else response
        raw_text = response.get("raw_text") or json.dumps(payload, sort_keys=True)
        # M5B.1: never fabricate {"action": "wait"} for a payload we couldn't
        # use - an empty dict here is an explicit, honest "no usable action",
        # which _validate_model_output classifies as empty_response/
        # missing_action rather than a real model-selected WAIT.
        return ModelResponse(
            payload=payload if isinstance(payload, dict) else {},
            raw_text=raw_text,
            provider=response.get("provider", provider_name),
            model=response.get("model", model_name),
            latency_ms=response.get("latency_ms"),
            time_to_first_token_ms=response.get("time_to_first_token_ms"),
            input_tokens=response.get("input_tokens"),
            output_tokens=response.get("output_tokens"),
            cached_input_tokens=response.get("cached_input_tokens"),
            finish_reason=response.get("finish_reason"),
            request_id=response.get("request_id"),
            error_type=response.get("error_type"),
            error_message=response.get("error_message"),
        )
    if isinstance(response, str):
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return ModelResponse(
                payload={},
                raw_text=response,
                provider=provider_name,
                model=model_name,
                error_type="json_parse_error",
                error_message="response was not valid JSON",
            )
        return ModelResponse(
            payload=payload if isinstance(payload, dict) else {},
            raw_text=response,
            provider=provider_name,
            model=model_name,
        )
    return ModelResponse(
        payload={},
        raw_text=str(response),
        provider=provider_name,
        model=model_name,
        error_type="provider_error",
        error_message="unexpected provider response",
    )


@dataclass
class ModelOutputValidation:
    """M5B.1: the outcome of the strict validation gate (see
    LLMController._validate_model_output). `valid=False` always carries a
    specific `error_type` from spec section 6's category list - never a
    generic catch-all, and never a fabricated WAIT.
    """

    valid: bool
    action: GameAction | None
    validated_action: dict[str, Any] | None
    error_type: str | None
    error_message: str | None


# M5B.1 spec section 7: the repair prompt "should tell the model what was
# wrong without revealing hidden information" - these are short, generic,
# category-level explanations, never anything derived from hidden GameState.
_REPAIR_REASONS: dict[str, str] = {
    "empty_response": "it was empty or contained no usable fields",
    "schema_error": "it was not valid structured JSON matching the required schema",
    "missing_action": "it did not contain a valid `action` field",
    "unknown_action": "it named an action that is not a recognized canonical action",
    "missing_target": "it was missing a required `target_id` (or `trap_type` for place_trap)",
    # M5F.2.3: found live - a hallucinated trap_type name used to pass
    # validation and then silently no-op at the engine (see
    # validate_model_output's place_trap branch for the full story).
    # M5F.3B.1: narrowed to its own real meaning now that
    # "real type, but not currently held" has a distinct code below.
    "invalid_trap_type": "the chosen trap_type is not a trap type this game supports",
    "trap_type_not_in_inventory": "the chosen trap_type is a real trap type, but you do not currently hold any",
    "invalid_target": "`target_id` must be a string",
    "illegal_action": "the chosen action is not currently legal",
    "illegal_target": "the chosen target is not currently a legal target for that action",
    "unknown_target": "the chosen target is not a real object in your current observation",
    "out_of_range": "the chosen target is real but you are not currently near it",
    # M6B.2 spec section 8: refined PLACE_TRAP-specific subtypes of
    # illegal_target - named explicitly so repair text states the real
    # reason instead of a generic rejection.
    "target_not_inspected": "you have not inspected this object yet - place_trap requires inspecting a target before trapping it",
    "target_not_trappable": "this object does not support having a trap placed on it",
    "provider_error": "the response could not be read",
    # M5F.2 spec section 39.
    "missing_message": "it did not contain a non-empty `message` field",
    "oversized_message": f"the message exceeded the {constants.MESSAGE_MAX_LENGTH}-character limit",
    # M5F.2.2 spec section 17: the ZERO_DECISION_COST envelope's optional
    # `message` object has its own distinct failure categories, since it's
    # validated separately from (and alongside) the physical action.
    "invalid_message_envelope": "the optional `message` field must be an object or null",
    "missing_message_recipient": "the message's `recipient_id` was missing or empty",
    "invalid_message_recipient": "the message's `recipient_id` was not the other spy's real actor id",
    "missing_message_text": "the message's `text` field was missing or empty",
    # M5F.2.3 spec section 8: the flat schema's own distinct failure
    # categories.
    "message_fields_incomplete": "message_to and message_text must both be present or both absent",
    "message_unavailable": "no message may be sent this decision - it is currently unavailable (cooldown active)",
}

# M5D.5: repair wording changed from M5B.1's original generic version -
# versioned explicitly per spec section 21 so telemetry can distinguish
# which repair contract produced a given repair attempt.
REPAIR_PROMPT_VERSION = "repair-v2"


def _repair_instruction(
    error_type: str | None,
    error_message: str | None,
    proposed_action: str | None = None,
    proposed_target_id: str | None = None,
    near_object_id: str | None = None,
    legal_action_count: int | None = None,
    legal_trap_types: list[str] | None = None,
    legal_place_trap_targets: list[str] | None = None,
) -> str:
    """M5D.5 spec sections 4-5: repair must fix a contract violation, not
    coach mission strategy (spec section 15) - so this states exactly what
    was rejected and why (echoing the actual previous proposal, spec
    section 4's "previous invalid proposal" + "precise contract-level
    error"), what the actor is legitimately near right now (section 7 -
    `near_object_id` is authoritative, never hidden truth), and a direct
    instruction not to repeat the same rejected pair - never "try again"
    (section 5), never a suggested next action (section 15).
    """
    reason = _REPAIR_REASONS.get(error_type or "", error_message or "it did not contain a valid action")
    if proposed_action:
        proposal_desc = f"{proposed_action}({proposed_target_id})" if proposed_target_id else proposed_action
    else:
        proposal_desc = "your previous response"
    near_desc = near_object_id if near_object_id else "nothing"
    count_desc = f" ({legal_action_count} available)" if legal_action_count is not None else ""
    # M5F.3B.1 spec section 8: a factual, runtime-derived legal-options
    # list for trap-type failures specifically - real current inventory
    # values, never a static/hardcoded catalog, and never a suggestion of
    # which one to pick (spec section 9: no automatic preference).
    trap_types_desc = ""
    # M5F.3B.1: gated on the caller having actually supplied legal trap
    # types (which it only does for a place_trap repair - see
    # `_attempt_repair`), not on `error_type` - a place_trap's very first
    # rejection is often `missing_target` (trap_type omitted entirely),
    # not yet `invalid_trap_type`, and that's exactly the attempt that
    # most needs this hint before the model guesses at a second value.
    if legal_trap_types is not None:
        trap_types_desc = (
            f" Allowed trap types in your current inventory: {legal_trap_types}. "
            "Choose one of these or choose another legal action."
        )
    # M6B.2 spec sections 9, 14 (RP12_LEGAL_TARGET_HINTS): a factual,
    # runtime-derived legal-target list for PLACE_TRAP target-legality
    # failures specifically (target_not_inspected/target_not_trappable/
    # illegal_target) - real current legal_actions membership, never a
    # strategic recommendation of which one to pick (spec section 9: "Do
    # not recommend which one is strategically preferable"). Since
    # PLACE_TRAP is proximity-gated to at most the one current
    # nearby_object_id (see enumerate_legal_actions), this list is almost
    # always `[]` (nothing currently qualifies - e.g. inspect it first) or
    # a single id, which is itself useful, honest information: it tells
    # the model plainly whether ANY place_trap target currently exists,
    # rather than leaving it to keep guessing at the same rejected one.
    target_hint_desc = ""
    if legal_place_trap_targets is not None:
        target_hint_desc = f" Legal PLACE_TRAP targets right now: {legal_place_trap_targets}."
    return (
        f"Your previous proposal {proposal_desc} was rejected: {error_type or 'invalid'} - {reason}. "
        f"You are currently near: {near_desc}. "
        f"Choose exactly one action from current legal_actions{count_desc}.{trap_types_desc}{target_hint_desc} "
        "Do not repeat the same rejected action/target."
    )


def place_trap_repair_hints(observation: Observation, legal_actions: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """M6B.2 spec sections 13, 24-25 (shared harness semantics audit): the
    factual PLACE_TRAP repair hints (RP06's legal trap types, RP12's legal
    targets) computed from the exact same inputs (`observation.
    trap_inventory`, the already-computed `legal_actions`) validation
    itself used - one definition, called identically by both
    LLMController._attempt_repair and StrandsAgentAdapter._request_action,
    so the two hints can never independently drift the way M6B's own
    legal_trap_types omission did before this milestone existed."""
    legal_trap_types = sorted(t for t, count in observation.trap_inventory.items() if count > 0)
    legal_place_trap_targets = sorted(
        entry.get("target_id") for entry in legal_actions
        if entry.get("action") == "place_trap" and entry.get("target_id")
    )
    return legal_trap_types, legal_place_trap_targets


def invalid_action_signature(action: str | None, target_id: str | None, trap_type: str | None) -> str:
    """M6B.2 spec section 11: a compact, deterministic string identifying
    one specific (action, target, trap_type) proposal - e.g.
    "PLACE_TRAP|target=desk_03|trap_type=electric" - used only to COUNT how
    many times an actor has now proposed this exact invalid combination
    within one episode (diagnosis of the M6B.1 never-inspected-target
    repetition pattern, never auto-suppression - spec section 11 is
    explicit that behavior shaping is out of scope here). Module-level so
    both LLMController and StrandsAgentAdapter compute it identically
    (see src/agents/strands_agent.py), never two independently-drifting
    definitions.
    """
    action_name = (action or "wait").strip().upper() or "WAIT"
    target_desc = target_id if target_id else "none"
    trap_desc = trap_type if trap_type else "none"
    return f"{action_name}|target={target_desc}|trap_type={trap_desc}"


# M5D.5 spec section 18: a small, closed vocabulary describing how a
# decision resolved - computed synchronously from facts available within
# one _request_action()/_request_action() call. STALE_ABORT is
# deliberately not produced here: within one synchronous decision request
# there is no later Observation to go stale against (repair reuses the
# same captured Observation as the initial attempt) - genuine async
# staleness (the world moved on *between* decisions) remains
# AgentMemoryTracker._check_pending_decision_failure's job, one tick
# later, against a real new Observation (see CHANGELOG.md's M5D.5 entry).
RECOVERY_FIRST_PASS_VALID = "FIRST_PASS_VALID"
RECOVERY_REPAIRED_VALID = "REPAIRED_VALID"
RECOVERY_REPAIR_FAILED_FALLBACK = "REPAIR_FAILED_FALLBACK"
RECOVERY_RUNTIME_FAILURE = "RUNTIME_FAILURE"


def _recovery_outcome(
    initial_valid: bool,
    repair_attempted: bool,
    repair_valid: bool | None,
    runtime_failure: bool = False,
) -> str:
    if runtime_failure:
        return RECOVERY_RUNTIME_FAILURE
    if initial_valid:
        return RECOVERY_FIRST_PASS_VALID
    if repair_attempted and repair_valid:
        return RECOVERY_REPAIRED_VALID
    return RECOVERY_REPAIR_FAILED_FALLBACK


# M5E.1 spec section 10: disjoint per-call error categories, computed from
# the error_type/error_message strings both runtimes already produce
# (never a new source of truth) - so a benchmark can report separate
# counts instead of one undifferentiated "provider/runtime failure"
# bucket. Shared between Native and Strands (imported by strands_agent.py)
# for the same reason `_repair_instruction`/`_recovery_outcome` are.
PROVIDER_TIMEOUT = "provider_timeout"
PROVIDER_TRANSPORT_ERROR = "provider_transport_error"
PROVIDER_HTTP_ERROR = "provider_http_error"
PROVIDER_INVALID_RESPONSE = "provider_invalid_response"
RUNTIME_EXCEPTION = "runtime_exception"
# M5E.3: Bedrock's own taxonomy (spec section 21) is a superset of
# Ollama's - two categories Ollama never needed (auth is keyless locally;
# Ollama has no per-request quota to throttle). `BedrockClient` classifies
# its own errors directly into these final category names (see
# bedrock_client.py's classify_bedrock_exception) rather than producing
# raw error_type strings for classify_provider_call_error to re-derive -
# handled below as an identity pass-through so both providers share one
# taxonomy surface without Bedrock's AWS-specific error codes leaking into
# this shared, provider-neutral function.
PROVIDER_AUTH_ERROR = "provider_auth_error"
PROVIDER_ACCESS_ERROR = "provider_access_error"
PROVIDER_THROTTLING = "provider_throttling"
PROVIDER_SERVICE_ERROR = "provider_service_error"
_FINAL_ERROR_CATEGORIES = {
    PROVIDER_TIMEOUT,
    PROVIDER_TRANSPORT_ERROR,
    PROVIDER_HTTP_ERROR,
    PROVIDER_INVALID_RESPONSE,
    RUNTIME_EXCEPTION,
    PROVIDER_AUTH_ERROR,
    PROVIDER_ACCESS_ERROR,
    PROVIDER_THROTTLING,
    PROVIDER_SERVICE_ERROR,
}


_SECRET_LIKE_PATTERN = re.compile(r"(api[_-]?key|authorization|token|secret|password)\s*[:=]\s*\S+", re.IGNORECASE)


def normalize_provider_error_message(message: str | None, max_len: int = 300) -> str | None:
    """M5E.1 spec section 9: "preserve the raw provider error safely...
    do not expose secrets or API credentials." Ollama is local/keyless so
    the practical risk is low, but this keeps the same hygiene regardless
    of which provider produced the error, and bounds telemetry size."""
    if message is None:
        return None
    redacted = _SECRET_LIKE_PATTERN.sub(lambda m: m.group(1) + "=[redacted]", message)
    return redacted[:max_len]


# Validation-only rejections: the call/response itself was fine (real,
# parseable content) - the model just proposed something game-illegal.
# Never a call-level failure category (spec section 10 keeps
# validation_failure disjoint from the provider/runtime categories).
# M6B.2: target_not_inspected/target_not_trappable are refined subtypes of
# illegal_target (spec sections 3, 8) - still game-rule validation, never a
# call-level failure.
_GAME_RULE_VALIDATION_ERRORS = {
    "missing_target", "illegal_target", "unknown_target", "out_of_range",
    "target_not_inspected", "target_not_trappable",
}


def classify_provider_call_error(error_type: str | None, error_message: str | None) -> str | None:
    """None means the call itself succeeded (a validation-level rejection
    of a well-formed response is not a *call* failure - see
    `_GAME_RULE_VALIDATION_ERRORS`/`validate_model_output` for that
    separate, already-existing classification)."""
    if not error_type or error_type in _GAME_RULE_VALIDATION_ERRORS:
        return None
    if error_type in _FINAL_ERROR_CATEGORIES:
        return error_type
    message = (error_message or "").lower()
    if "timed out" in message or "timeout" in message:
        return PROVIDER_TIMEOUT
    if error_type == "provider_error":
        if any(token in message for token in ("connection", "reset", "refused", "broken pipe", "unreachable")):
            return PROVIDER_TRANSPORT_ERROR
        if "http" in message or "status" in message:
            return PROVIDER_HTTP_ERROR
        return PROVIDER_TRANSPORT_ERROR
    if error_type == "strands_runtime_error":
        return RUNTIME_EXCEPTION
    if error_type in ("empty_response", "missing_action", "no_tool_selected", "json_parse_error"):
        return PROVIDER_INVALID_RESPONSE
    return RUNTIME_EXCEPTION


def validate_model_output(
    observation: Observation,
    response: Any,
    legal_actions: list[dict[str, Any]],
    zero_decision_cost: bool = False,
    flat_message_schema: bool = False,
    message_available: bool = True,
) -> ModelOutputValidation:
    """M5B.1 strict gate: a response is valid only if it parses into a
    canonical, schema-valid, currently legal GameAction. Never defaults a
    missing/unrecognized action to WAIT - only an explicit
    `{"action": "wait"}` counts as a genuine WAIT decision (spec sections
    1-3). Returns a specific error category on failure rather than one
    generic "invalid" bucket (spec section 6).

    Module-level (not a method): M5C.1's StrandsAgentAdapter reuses this
    exact function so the strict-validity semantics are identical across
    every agent runtime, not reimplemented per runtime (M5C.1 spec section
    12: "M5B.1 remains mandatory").

    `zero_decision_cost` (M5F.2.2 spec section 1, default False - every
    pre-M5F.2.2 caller's implicit value, byte-identical behavior): when
    True, `response["message"]` is interpreted as an optional ENVELOPE
    object (`{"recipient_id", "text"}`) that may accompany any physical
    action, validated as one unit alongside that action (spec section 17
    - either both are valid or the whole decision is invalid) - a
    completely different meaning from the same key's FULL_COST role
    (plain text, only for the standalone `send_message` action), so the
    two code paths below are mutually exclusive on this flag.

    `flat_message_schema` (M5F.2.3 spec section 6, only consulted when
    `zero_decision_cost` is True): when True, the envelope is instead two
    flat top-level fields (`message_to`/`message_text`) rather than a
    nested `message` object - the nested-vs-flat branches are themselves
    mutually exclusive. `message_available` (M5F.2.3 spec section 3,
    default True - unbounded, every pre-M5F.2.3 caller's implicit value):
    when False, `message_to`/`message_text` being present at all is
    itself invalid (spec section 8's "message unavailable + message
    fields present -> invalid"), regardless of whether their content
    would otherwise have been legal - not silently dropped (spec section
    29), a genuine validation failure that goes through the normal
    repair-then-fallback path like any other.
    """

    def invalid(error_type: str, error_message: str) -> ModelOutputValidation:
        return ModelOutputValidation(False, None, None, error_type, error_message)

    if not isinstance(response, dict):
        return invalid("schema_error", "response is not a JSON object")
    if not response:
        return invalid("empty_response", "response contained no fields")

    raw_action = response.get("action")
    if raw_action is None:
        return invalid("missing_action", "response has no 'action' field")
    if not isinstance(raw_action, str) or not raw_action.strip():
        return invalid("schema_error", "'action' must be a non-empty string")

    normalized_name = _normalize_action_name(raw_action)
    encounters = bool(getattr(observation, "encounters_enabled", False))
    if encounters:
        normalized_name = normalize_encounter_action(normalized_name)
    is_encounter_action = encounters and normalized_name in ENCOUNTER_ACTIONS
    riddle = (getattr(observation, "puzzle", None) or {}).get("type") == "riddle"
    raw_key = raw_action.strip().lower().replace("-", "_").replace(" ", "_")
    if riddle and raw_key in _ENTER_CODE_ALIASES:
        normalized_name = "enter_safe_code"  # live riddle action (not one of the recorded action names)
        response = {**_normalize_target_id(response), "action": "enter_safe_code"}
        if not response.get("target_id"):
            response["target_id"] = "ambassador_safe"
        if not _valid_code(response.get("code")):
            return invalid("missing_code", "enter_safe_code needs code: the digits you worked out, e.g. \"1947\"")
    elif normalized_name not in VALID_ACTION_NAMES and not is_encounter_action:
        return invalid("unknown_action", f"unrecognized action {raw_action!r}")
    if is_encounter_action and not response.get("target_id"):
        # the only possible target is the opponent - accept it implicitly
        response = {**response, "target_id": observation.other_actor_id}

    target_id = response.get("target_id")
    if target_id is not None and not isinstance(target_id, str):
        return invalid("invalid_target", "'target_id' must be a string")
    if _action_requires_target(normalized_name) and not target_id:
        return invalid("missing_target", f"{normalized_name} requires a target_id")

    if normalized_name == "place_trap":
        trap_type = response.get("trap_type")
        if not trap_type or not isinstance(trap_type, str):
            return invalid("missing_target", "place_trap requires a trap_type")
        # M5F.2.3: a real, pre-existing gap found live (not introduced by
        # this milestone's own work, but exposed by its multi-decision
        # sequence testing) - this only ever checked that `trap_type` was
        # a non-empty string, never that the actor actually HELD one of
        # that type. A hallucinated name (e.g. "stun" - not one of this
        # game's four real trap types, bomb/electric/snare/alarm) used to
        # pass validation, then silently no-op at the engine
        # (`GameState._place_trap`'s own inventory check rejects it,
        # spending no time and mutating nothing) - leaving the
        # Observation digest unchanged and permanently starving the actor
        # of any further decision (LLMController.get_action's digest-
        # dedup has nothing new to react to). `observation.trap_inventory`
        # is exactly the actor's own legitimate self-knowledge already
        # used to gate PLACE_TRAP's own legality in `enumerate_legal_actions`
        # (`has_any_trap`), so checking the specific type here needs no
        # new data source.
        #
        # M5F.3B.1 spec section 7: two distinct failure modes, kept as two
        # distinct error_type codes rather than one conflated message -
        # `invalid_trap_type` stays exactly what it always meant (a string
        # that isn't one of this Engine's real trap types at all, e.g. the
        # recurring "stun" hallucination - `constants.TRAP_TYPES` is the
        # authoritative set, never a hardcoded duplicate of it here) so
        # every pre-M5F.3B.1 caller/test keeps its exact prior behavior;
        # `trap_type_not_in_inventory` is new - a real trap type this game
        # does support, just not one this actor currently holds any of.
        if trap_type not in constants.TRAP_TYPES:
            return invalid("invalid_trap_type", f"{trap_type!r} is not a trap type this game supports")
        if observation.trap_inventory.get(trap_type, 0) <= 0:
            return invalid("trap_type_not_in_inventory", f"you do not currently hold any {trap_type!r} traps")

    if not zero_decision_cost and normalized_name == "send_message":
        # M5F.2 spec section 39: message-content problems get their own
        # precise categories, checked before recipient legality below (via
        # convert_response_to_action's legal-target check) - so a repair
        # instruction can tell "your message was empty/too long" apart
        # from "you named an illegal recipient" instead of one generic
        # rejection either way.
        message_text = response.get("message")
        if message_text is None or not isinstance(message_text, str) or not message_text.strip():
            return invalid("missing_message", "send_message requires a non-empty message field")
        if len(message_text) > constants.MESSAGE_MAX_LENGTH:
            return invalid("oversized_message", f"message exceeds the {constants.MESSAGE_MAX_LENGTH}-character limit")

    # M5F.2.2 spec sections 2, 17 / M5F.2.3 spec sections 6, 8: the
    # ZERO_DECISION_COST envelope's optional message - validated as part
    # of the same unit as the physical action below (never silently
    # stripped either way). `accompanying_message` stays None for every
    # FULL_COST call and for a ZERO_DECISION_COST call that simply omits
    # the message fields (both legal, honest "no message this decision"
    # cases).
    accompanying_message: dict[str, str] | None = None
    if zero_decision_cost and flat_message_schema:
        # M5F.2.3 spec section 8: message_to/message_text must both be
        # present or both absent - never just one (an incomplete pair is
        # its own precise error category, distinct from a bad recipient
        # or bad text, so repair instructions stay specific).
        message_to = response.get("message_to")
        message_text = response.get("message_text")
        has_to = message_to is not None
        has_text = message_text is not None
        if has_to != has_text:
            return invalid("message_fields_incomplete", "message_to and message_text must both be present or both absent")
        if has_to and has_text:
            if not message_available:
                return invalid("message_unavailable", "no message may be sent this decision - cooldown active")
            if not isinstance(message_to, str) or not message_to.strip():
                return invalid("missing_message_recipient", "message_to must be a non-empty string")
            if message_to == observation.actor_id or message_to != observation.other_actor_id:
                return invalid("invalid_message_recipient", "message_to is not the other spy's real actor id")
            if not isinstance(message_text, str) or not message_text.strip():
                return invalid("missing_message_text", "message_text must be non-empty")
            if len(message_text) > constants.MESSAGE_MAX_LENGTH:
                return invalid("oversized_message", f"message_text exceeds the {constants.MESSAGE_MAX_LENGTH}-character limit")
            # M5F.2.3 spec section 7: normalized into the exact same
            # internal shape the nested envelope already used - every
            # downstream delivery/ActionExecutor/GameState code path
            # needs zero changes for the flat schema.
            accompanying_message = {"recipient_id": message_to, "text": message_text.strip()[: constants.MESSAGE_MAX_LENGTH]}
    elif zero_decision_cost:
        raw_message = response.get("message")
        if raw_message is not None:
            if not isinstance(raw_message, dict):
                return invalid("invalid_message_envelope", "'message' must be an object or null")
            recipient_id = raw_message.get("recipient_id")
            message_text = raw_message.get("text")
            if recipient_id is None or not isinstance(recipient_id, str) or not recipient_id.strip():
                return invalid("missing_message_recipient", "message.recipient_id is required")
            if recipient_id == observation.actor_id or recipient_id != observation.other_actor_id:
                return invalid("invalid_message_recipient", "message.recipient_id is not the other spy's real actor id")
            if message_text is None or not isinstance(message_text, str) or not message_text.strip():
                return invalid("missing_message_text", "message.text must be non-empty")
            if len(message_text) > constants.MESSAGE_MAX_LENGTH:
                return invalid("oversized_message", f"message.text exceeds the {constants.MESSAGE_MAX_LENGTH}-character limit")
            accompanying_message = {"recipient_id": recipient_id, "text": message_text.strip()[: constants.MESSAGE_MAX_LENGTH]}

    action = convert_response_to_action(observation, response, legal_actions)
    if action is None:
        # M5D.4 spec section 12 / M5D.5 spec section 5: distinguish three
        # ways a target can fail, not one catch-all "illegal_target":
        # - unknown_target: the named object doesn't exist in this actor's
        #   own Observation at all;
        # - out_of_range: a real, known object, for a proximity-gated
        #   action (inspect/interact/place_trap/disarm), that simply isn't
        #   the actor's current nearby_object_id - by far the most common
        #   failure observed live (M5D.4/M5D.5 telemetry audits);
        # - illegal_target: a real, in-range object still rejected for
        #   another legitimate reason (locked, needs prior inspection, no
        #   trap inventory, no detected trap to disarm).
        # Losing this distinction is exactly what made every repair
        # instruction generic ("not currently legal") instead of precise.
        if is_encounter_action:
            category = "encounter_not_available"
        elif target_id is not None and observation.get(target_id) is None:
            category = "unknown_target"
        elif (
            target_id is not None
            and normalized_name in _PROXIMITY_GATED_ACTIONS
            and target_id != observation.nearby_object_id
        ):
            category = "out_of_range"
        elif target_id is not None and normalized_name == "place_trap":
            # M6B.2 spec sections 3, 8: a real, in-range object still
            # rejected for PLACE_TRAP specifically has exactly two possible
            # actor-visible causes left at this point (trap_type/inventory
            # were already validated above, and out_of_range was already
            # ruled out) - `obj.trappable`/`obj.inspected` are both the
            # actor's own legitimate self-knowledge (Observation fields,
            # never hidden GameState), so the taxonomy can name the real
            # reason instead of one generic "illegal_target" bucket. A
            # target this actor has never inspected is exactly the pattern
            # M6B.1 traced 23/25 of its new "other validation failure"
            # occurrences to (see outlier_target_legality_analysis.md) -
            # this makes that failure mode a first-class, reportable
            # category rather than folded into an unexplained catch-all.
            target_obj = observation.get(target_id)
            if target_obj is not None and not target_obj.trappable:
                category = "target_not_trappable"
            elif target_obj is not None and not target_obj.inspected:
                category = "target_not_inspected"
            else:
                # Anything else (e.g. a hidden pre-existing trap - spec
                # section 7: never leak that truth here) stays the
                # existing generic category; the engine itself is the
                # final authority for that case, not this validator.
                category = "illegal_target"
        elif target_id is not None:
            category = "illegal_target"
        else:
            category = "illegal_action"
        return invalid(category, f"{normalized_name} is not currently legal")

    if accompanying_message is not None:
        action = dataclasses_replace(action, accompanying_message=accompanying_message)

    validated_action = {
        "action": normalized_name,
        "target_id": target_id,
        "trap_type": response.get("trap_type"),
        "message": response.get("message") if (not zero_decision_cost and normalized_name == "send_message") else None,
        "accompanying_message": accompanying_message,
        "reason": response.get("reason"),
        **({"code": str(response.get("code")).strip()} if normalized_name == "enter_safe_code" else {}),
    }
    return ModelOutputValidation(True, action, validated_action, None, None)


class LLMController:
    """Minimal provider-neutral controller for a legal-action, structured-output LLM."""

    def __init__(
        self,
        actor_id: str,
        model_client: ModelClient | None = None,
        config: LLMControllerConfig | None = None,
        telemetry: TelemetryLogger | None = None,
        episode_id: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        execution_mode: str | None = None,
        agent_runtime: str = "native",
        agent_runtime_version: str = "native-v1",
    ) -> None:
        self.actor_id = actor_id
        # M5C: which agent runtime produced this decision - a separate
        # telemetry dimension from provider/model (see src/agents/base.py).
        # LLMController itself is always the native implementation, whether
        # constructed directly (pre-M5C call sites, most tests) or through
        # NativeAgentAdapter, so this defaults to "native" either way rather
        # than requiring every caller to opt in.
        self.agent_runtime = agent_runtime
        self.agent_runtime_version = agent_runtime_version
        self.config = config or LLMControllerConfig()
        if execution_mode is not None:
            self.config.execution_mode = execution_mode
        # M5F.1: validated and rendered once here, not re-read per call -
        # an invalid profile name fails loudly at construction (spec
        # section 13), never silently at decision time.
        from src.llm.personality import render_personality_block

        self._personality_block = render_personality_block(self.config.personality_profile)
        # M5F.2: same frozen-at-construction discipline as the personality
        # block immediately above - `config.communication_enabled` is read
        # exactly once, here, never re-checked per call.
        from src.llm.communication import render_communication_block

        self._communication_block = render_communication_block(
            self.config.communication_enabled, zero_decision_cost=self.config.zero_decision_cost_messaging,
            flat_message_schema=self.config.flat_message_schema,
        )
        # M5F.2.3 spec sections 3, 11: actor-specific (this controller
        # instance already is - one per actor, never shared, same
        # isolation M5F established), decision-count-based (never wall
        # clock/latency/ticks) message-availability tracking. `None` means
        # "never sent, available from the start" (spec section 3's "message
        # initially available") - distinct from `0`, which specifically
        # means "just sent, cooldown running."
        self._decisions_since_last_message: int | None = None
        self.telemetry = telemetry
        self.episode_id = episode_id or "ep_unknown"
        self.model_client = model_client or MockModelClient(provider=provider_name or "mock", model=model_name or "mock-model")
        self.provider_name = provider_name or getattr(self.model_client, "provider", "mock")
        self.model_name = model_name or getattr(self.model_client, "model", "mock-model")
        self.execution_mode = str(self.config.execution_mode).lower()
        # Bugfix live flow (Part A): live-play-only async timing - see
        # src/agents/live_async.py. Evaluation/playable modes never touch it.
        self._live_async = self.execution_mode == live_async.LIVE_ASYNC
        self._live_pending: live_async.PendingLiveDecision | None = None
        self._live_start_with_messages = True
        self.live_async_stats = live_async.LiveAsyncStats()
        self.status = self.config.status
        self.last_action = GameAction(ActionType.WAIT)
        self.last_reason = "No model decision yet."
        self._last_observation_digest: str | None = None
        self.calls = 0
        self.invalid_calls = 0
        # M5F.3B.1 spec section 12: `fallback_count` used to be the ONLY
        # failure counter, incremented identically for a genuine provider/
        # network exception and for a model-chosen-invalid-action-that-
        # never-repaired - conflating infrastructure failures with model
        # behavior. Kept as the historical total (every pre-M5F.3B.1
        # reader of `fallback_count` still gets the same combined number),
        # but split into its two real components below so a caller (e.g.
        # `duel_runner.run_duel_episode`'s episode-ending failure cap) can
        # react to genuine infrastructure instability without an episode
        # of otherwise-healthy, cleanly-repaired-or-fallen-back model
        # mistakes counting against it.
        self.fallback_count = 0
        self.provider_failure_count = 0
        self.validation_failure_count = 0
        self.repair_failure_count = 0
        self.repair_count = 0
        # M6B.2 spec section 11: per-episode counts of each distinct
        # invalid-proposal signature seen so far (diagnosis only - see
        # `invalid_action_signature`).
        self._invalid_signature_counts: dict[str, int] = {}
        # M5D.2: strategic decisions vs. accepted-action execution, kept
        # explicitly distinct (spec section 11) - see action_lifecycle.py.
        self._executor = ActionExecutor()
        self.decisions_requested = 0
        self.decisions_completed = 0
        self._decision_seq = 0
        self._decision_interval_seconds = 0.5
        self._next_decision_allowed_at = 0.0
        self._thinking_thread: threading.Thread | None = None
        self._thinking_lock = threading.Lock()
        self._thinking_in_progress = False
        # M6G.3: the explicit RESPONSE_READY_UNCOMMITTED state. Set at the
        # single ready point (_request_action_background), cleared when
        # get_action hands the decision to the game loop - which sets the
        # per-call marker - or counted as cancelled when the actor is
        # busy/stunned at that moment (the existing discard). Game._step's
        # response barrier reads these; bookkeeping only, in every mode.
        self._response_ready_uncommitted = False
        self.response_ready_at: float | None = None
        self.last_call_committed_ready_response = False
        self.ready_responses_cancelled = 0
        self.last_ready_response_cancel_reason: str | None = None
        
        # M5B: Agent decision memory and strategic context. The diffing/
        # pending-decision logic lives in AgentMemoryTracker (M5C.1) so
        # every runtime adapter shares one implementation - see
        # src/llm/agent_memory.py.
        self._memory_tracker = AgentMemoryTracker(actor_id)
        self._last_decision_id: str | None = None
        # M5C: mirrors the action_source of the most recent memory.record_decision
        # call, so NativeAgentAdapter.decide() can report it without re-deriving
        # it from the returned GameAction.
        self._last_action_source: str = "llm"

    def _effective_system_prompt(self) -> str:
        """Spec section 9: the personality block is appended after the
        base system prompt, never replacing it - `NativeToolLikeAgentAdapter`
        overwrites `config.system_prompt` itself (see native_agent.py), so
        keeping the personality text on a separate field means it survives
        that unconditionally regardless of which representation variant is
        in play. `None` (DEFAULT profile) returns the base prompt
        byte-identical to every pre-M5F.1 caller."""
        prompt = self.config.system_prompt
        if self._personality_block is not None:
            prompt = f"{prompt}\n\n{self._personality_block}"
        # M5F.2: appended last, after personality - never conditioned on
        # personality/provider/anything else (spec section 9: identical
        # for both actors whenever communication is on at all).
        if self._communication_block is not None:
            prompt = f"{prompt}\n\n{self._communication_block}"
        if self.config.live_prompt_addendum:
            prompt = f"{prompt}\n\n{self.config.live_prompt_addendum}"
        return prompt

    def _message_cooldown_active(self) -> bool:
        """M5F.2.3 spec section 3: True while the bounded free-message
        rule blocks a new message THIS decision. `message_cooldown_decisions
        <= 0` (default) means no cooldown at all - always available,
        M5F.2.2's own unbounded behavior. `_decisions_since_last_message`
        is `None` until the first message is ever sent (available), then
        counts decisions elapsed since the last send - available again
        once it reaches `message_cooldown_decisions` (see
        `_request_action`'s increment-then-check ordering for the exact
        off-by-one reasoning: sending at decision K, counter reads 1 at
        K+1 and 2 at K+2 - both still below 3 - and 3 at K+3, the first
        available decision again)."""
        if self.config.message_cooldown_decisions <= 0:
            return False
        if self._decisions_since_last_message is None:
            return False
        return self._decisions_since_last_message < self.config.message_cooldown_decisions

    def _decisions_until_available(self) -> int:
        """M5F.2.3 spec section 10: the exact `next_available_in_decisions`
        figure shown mechanically in the model-facing payload - `0` when
        already available (never negative)."""
        if not self._message_cooldown_active():
            return 0
        return self.config.message_cooldown_decisions - (self._decisions_since_last_message or 0)

    @property
    def memory(self) -> AgentMemory:
        return self._memory_tracker.memory

    # M5D.2: actions_* counters live on the shared ActionExecutor (spec
    # section 11) so both LLMController and StrandsAgentAdapter read one
    # definition - these delegate for existing/external callers (tests,
    # telemetry aggregation) that read `controller.actions_completed` etc.
    @property
    def actions_accepted(self) -> int:
        return self._executor.actions_accepted

    @property
    def actions_started(self) -> int:
        return self._executor.actions_started

    @property
    def actions_completed(self) -> int:
        return self._executor.actions_completed

    @property
    def actions_failed(self) -> int:
        return self._executor.actions_failed

    @property
    def actions_interrupted(self) -> int:
        return self._executor.actions_interrupted

    @property
    def actions_cancelled(self) -> int:
        return self._executor.actions_cancelled

    def _update_memory_from_observation(self, observation: Observation) -> None:
        """Update persistent strategic context from the authoritative observation."""
        self._memory_tracker.update_from_observation(observation)

    def _get_memory_telemetry_kwargs(self) -> dict[str, Any]:
        """Get memory metadata for telemetry reporting."""
        return self.memory.to_telemetry_dict()

    def _provisional_action_for(self, observation: Observation, observation_payload: dict[str, Any]) -> GameAction:
        if self.last_action is not None and self.last_action.type != ActionType.WAIT:
            return self.last_action

        legal_actions = observation_payload.get("legal_actions", [])
        for entry in legal_actions:
            if _normalize_action_name(entry.get("action")) == "move":
                candidate = self._convert_response_to_action(observation, entry, legal_actions)
                if candidate is not None and candidate.type != ActionType.WAIT:
                    return candidate

        for entry in legal_actions:
            if _normalize_action_name(entry.get("action")) == "wait":
                continue
            candidate = self._convert_response_to_action(observation, entry, legal_actions)
            if candidate is not None and candidate.type != ActionType.WAIT:
                return candidate
        return GameAction(ActionType.WAIT)

    def _inflight_action(self, observation: Observation, observation_payload: dict[str, Any], *,
                         handing_over: bool = False) -> GameAction:
        if self.execution_mode == "playable":
            provisional = self._provisional_action_for(observation, observation_payload)
            if provisional.type != ActionType.WAIT:
                self.last_action = provisional
                return provisional
        if self._live_async:
            # Live async: a thinking actor has no active action (requests
            # launch only at boundaries), so it stands still while the
            # world - and the other spy - keep going.
            return GameAction(ActionType.WAIT)
        action = self.last_action
        # M6G.3 (HP09): a decision published a moment ago (its flag is set
        # before `last_action` at the ready point, and read after it here)
        # is never returned as the in-flight action - it is handed over by
        # the next call, so the game loop sees exactly one hand-over.
        if not handing_over and self.config.ready_response_guard and self._response_ready_uncommitted:
            return GameAction(ActionType.WAIT)
        if action.type != ActionType.WAIT:
            return action
        return GameAction(ActionType.WAIT)

    @property
    def response_ready_uncommitted(self) -> bool:
        """M6G.3: a completed decision is waiting to be handed to the game loop."""
        return self._response_ready_uncommitted

    def _hand_over(self, action: GameAction) -> GameAction:
        """M6G.3: this call returns the completed decision to the game loop
        (RESPONSE_READY_UNCOMMITTED -> handed over; the scheduler commits it)."""
        if self._response_ready_uncommitted and not self._thinking_in_progress:
            self._response_ready_uncommitted = False
            self.last_call_committed_ready_response = True
        return action

    def _cancel_ready_response(self, reason: str) -> None:
        """M6G.3: the existing discard of a ready decision (busy/stunned/
        mission complete when it would be handed over) - recorded, not changed."""
        if self._response_ready_uncommitted and not self._thinking_in_progress:
            self._response_ready_uncommitted = False
            self.ready_responses_cancelled += 1
            self.last_ready_response_cancel_reason = reason

    def note_live_tick(self, *, self_moved: bool, opponent_moved: bool, opponent_thinking: bool) -> None:
        """Live async only (called by Game._step): per-frame flow telemetry."""
        live_async.note_world_tick(self, self_moved=self_moved, opponent_moved=opponent_moved,
                                   opponent_thinking=opponent_thinking)

    def get_action(self, observation: Observation) -> GameAction:
        self.last_call_committed_ready_response = False
        if observation.mission_complete or observation.stunned or observation.busy or observation.time_expired:
            if self._live_async and self._response_ready_uncommitted and not self._thinking_in_progress:
                live_async.cancel_ready_decision(self, observation, live_async.cancel_reason(observation))
            self._cancel_ready_response("busy_stunned_or_mission_complete_at_hand_over")
            self.status = "IDLE"
            self.last_action = GameAction(ActionType.WAIT)
            self.last_reason = "Mission complete, stunned, or busy."
            if self._executor.in_progress:
                _, boundary = self._executor.continue_(observation)
                if boundary is not None:
                    self._finish_action_lifecycle(observation, boundary)
            return self.last_action

        # M5B: Update persistent memory from authoritative observation
        self._update_memory_from_observation(observation)

        # M5D.2: an accepted action already in progress (currently only
        # MOVE spans multiple ticks - see action_lifecycle.py) keeps
        # executing without requesting a new strategic decision. This is
        # the actual fix for M5D.1's no-progress finding: previously, the
        # actor's own natural position drift *while executing an already-
        # accepted move* looked just as "new" to the observation-digest
        # check below as any genuinely new situation, so almost every tick
        # triggered a brand-new (and, for a real model, extremely
        # expensive) decision cycle instead of letting the move continue.
        if self._executor.in_progress:
            if (not self._live_async and self._response_ready_uncommitted and not self._thinking_in_progress
                    and self._executor.fired_at_start()):
                # Evaluation timing with the live close approach: the close-range
                # action fired in start() on the request thread; this hand-over
                # frame performs it (continue_ would end it unperformed).
                return self._hand_over(self.last_action)
            action, boundary = self._executor.continue_(observation)
            self.last_action = action
            if boundary is None:
                return self._hand_over(action)
            self._finish_action_lifecycle(observation, boundary)
            # Boundary just reached - fall through so a new decision can be
            # considered this same tick instead of wasting one doing nothing.

        observation_payload = serialize_observation(observation, live_state=self.config.live_state_representation,
                                                    live_workshop=self.config.live_time_facts)
        # Exclude ephemeral time fields from the observation digest so the
        # controller doesn't re-query the model on every tick as time elapses.
        payload_for_digest = dict(observation_payload)
        payload_for_digest.pop("time", None)
        observation_digest = hashlib.sha256(json.dumps(payload_for_digest, sort_keys=True).encode("utf-8")).hexdigest()

        # Live async: a ready answer is classified against the world as it
        # is now, then started or discarded (src/agents/live_async.py).
        if self._live_async and self._live_pending is not None and self._response_ready_uncommitted and not self._thinking_in_progress:
            return live_async.commit_ready_decision(self, observation, observation_payload["legal_actions"], observation_digest)

        # simulated-clock lab: pacing on the game clock (elapsed = -remaining)
        game_now = -observation.mission_time_remaining if self.config.game_clock_pacing else None
        if self._live_async:
            live_async.idle_redecide(self)
        elif game_now is not None:
            live_async.idle_redecide(self, now=game_now)
        if self._last_observation_digest == observation_digest:
            if self._thinking_in_progress:
                return self._inflight_action(observation, observation_payload)
            return self._hand_over(self.last_action)

        # the game-clock throttle keeps its own deadline (game seconds, from -inf)
        now = time.monotonic() if game_now is None else game_now
        if now < (self._next_decision_allowed_at if game_now is None
                  else getattr(self, "_next_game_decision_allowed_at", float("-inf"))):
            if self._thinking_in_progress:
                return self._inflight_action(observation, observation_payload)
            return self._hand_over(self.last_action)

        # M6G.3 (HP09): a completed decision that has not been handed over
        # yet is handed over now instead of being replaced by a new request.
        # The digest is deliberately not stored, so the next call sees the
        # change and launches normally. Closes both M6G.1/M6G.2 routes: an
        # answer landing next to an observation change, and a change seen
        # inside the throttle window but consumed after it.
        if self.config.ready_response_guard and self._response_ready_uncommitted and not self._thinking_in_progress:
            return self._hand_over(self.last_action)

        self._last_observation_digest = observation_digest
        if game_now is None:
            self._next_decision_allowed_at = now + self._decision_interval_seconds
        else:
            self._next_game_decision_allowed_at = now + self._decision_interval_seconds
        self._decision_seq += 1

        if self._thinking_in_progress:
            return self._inflight_action(observation, observation_payload)

        if self._response_ready_uncommitted:
            # Only reachable with the guard off (pre-M6G.3 lifecycle): the
            # in-flight action this very call returns is that ready decision
            # while a replacement is launched - the HP09 race, recorded.
            self._hand_over(self.last_action)

        # Evaluation mode is the default: while an LLM request is in flight we
        # wait and let the engine continue only with an already accepted action.
        self._thinking_in_progress = True
        self.status = "THINKING"
        # M5D.2: incremented exactly once per genuinely-spawned strategic
        # request - not per tick/poll, and not for the (rare, non-move)
        # digest changes above that arrive while already thinking and
        # therefore never reach this point (spec section 11).
        self.decisions_requested += 1
        if self._live_async:
            live_async.on_request_launched(self, observation, observation_digest)
        thread = threading.Thread(target=self._request_action_background, args=(observation, observation_payload))
        thread.daemon = True
        thread.start()
        self._thinking_thread = thread
        thread.join(timeout=0.05)
        if self._response_ready_uncommitted and not self._thinking_in_progress:
            if self._live_async and self._live_pending is not None:
                return live_async.commit_ready_decision(self, observation, observation_payload["legal_actions"], observation_digest)
            # answered within the join: this very call hands the decision over
            return self._hand_over(self._inflight_action(observation, observation_payload, handing_over=True))
        return self._inflight_action(observation, observation_payload)

    def _validate_model_output(
        self,
        observation: Observation,
        response: Any,
        legal_actions: list[dict[str, Any]],
    ) -> ModelOutputValidation:
        return validate_model_output(
            observation, response, legal_actions, zero_decision_cost=self.config.zero_decision_cost_messaging,
            flat_message_schema=self.config.flat_message_schema,
            message_available=not self._message_cooldown_active(),
        )

    def _attempt_repair(
        self,
        observation: Observation,
        legal_actions: list[dict[str, Any]],
        output_schema: dict[str, Any],
        model_payload: dict[str, Any],
        initial_validation: ModelOutputValidation,
        raw_response: dict[str, Any] | None = None,
        decision_id: str | None = None,
        decision_number: int = 0,
    ) -> tuple[ModelOutputValidation, dict[str, Any], ModelResponse | None, int]:
        """One real re-prompt (spec section 7) - never a locally-fabricated
        "fix". Tells the model what was wrong (generic, category-level, no
        hidden information) and re-validates its second response through
        the exact same strict gate. Returns
        (repair_validation, repair_raw_response, repair_model_response,
        repair_context_chars).
        """
        repair_payload = dict(model_payload)
        raw_response = raw_response or {}
        # M5F.3B.1: found live (Stage 2) - the model's INITIAL place_trap
        # proposal often omits `trap_type` entirely (`missing_target`, not
        # `invalid_trap_type`), and only the REPAIR attempt adds one -
        # landing on the same hallucinated "stun" every time, because the
        # original condition here only fired for an initial
        # `invalid_trap_type`/`trap_type_not_in_inventory` error, so the
        # very attempt that needed the factual legal-values hint never
        # got it. Broadened to any place_trap repair, regardless of which
        # specific error triggered it.
        legal_trap_types = None
        legal_place_trap_targets = None
        if raw_response.get("action") == "place_trap":
            legal_trap_types, legal_place_trap_targets = place_trap_repair_hints(observation, legal_actions)
        repair_instruction = _repair_instruction(
            initial_validation.error_type,
            initial_validation.error_message,
            proposed_action=raw_response.get("action"),
            proposed_target_id=raw_response.get("target_id"),
            near_object_id=observation.nearby_object_id,
            legal_action_count=len(legal_actions),
            legal_trap_types=legal_trap_types,
            legal_place_trap_targets=legal_place_trap_targets,
        )
        repair_payload["repair_instruction"] = repair_instruction
        repair_payload["repair_prompt_version"] = REPAIR_PROMPT_VERSION
        repair_prompt_json = json.dumps(repair_payload, sort_keys=True,
                                        **({"separators": (",", ":")} if self.config.compact_payload_json else {}))
        repair_context_chars = len(repair_instruction)
        start = time.monotonic()
        start_iso = datetime.now(timezone.utc).isoformat()
        try:
            raw = self.model_client.generate_action(
                self._effective_system_prompt(),
                repair_prompt_json,
                output_schema,
                request_context={"episode_id": self.episode_id, "actor_id": self.actor_id, "repair": True},
            )
        except Exception as exc:  # pragma: no cover - defensive safety boundary
            self._emit_provider_call(
                decision_id=decision_id,
                decision_number=decision_number,
                call_index=2,
                call_type="repair",
                start_iso=start_iso,
                latency_ms=(time.monotonic() - start) * 1000.0,
                success=False,
                error_type="provider_error",
                error_message=str(exc),
            )
            return ModelOutputValidation(False, None, None, "provider_error", str(exc)), {}, None, repair_context_chars

        normalized = _coerce_model_response(raw, self.provider_name, self.model_name)
        repair_raw_response = _normalize_target_id(normalized.payload or {})
        if self.config.model_response_transform is not None:
            repair_raw_response = self.config.model_response_transform(repair_raw_response)
        repair_validation = self._validate_model_output(observation, repair_raw_response, legal_actions)
        # A well-formed HTTP/provider response whose *content* turned out
        # unusable (e.g. empty_response) is a call-level fact too (spec
        # section 10's provider_invalid_response), not just a validation
        # outcome - same fallback pattern _request_action already uses for
        # initial_error_type below.
        call_error_type = normalized.error_type or (None if repair_validation.valid else repair_validation.error_type)
        call_error_message = normalized.error_message or (None if repair_validation.valid else repair_validation.error_message)
        self._emit_provider_call(
            decision_id=decision_id,
            decision_number=decision_number,
            call_index=2,
            call_type="repair",
            start_iso=start_iso,
            latency_ms=normalized.latency_ms or ((time.monotonic() - start) * 1000.0),
            success=(call_error_type is None),
            error_type=call_error_type,
            error_message=call_error_message,
            input_tokens=normalized.input_tokens,
            output_tokens=normalized.output_tokens,
        )
        return repair_validation, repair_raw_response, normalized, repair_context_chars

    def _emit_provider_call(
        self,
        *,
        decision_id: str | None,
        decision_number: int,
        call_index: int,
        call_type: str,
        start_iso: str,
        latency_ms: float,
        success: bool,
        error_type: str | None = None,
        error_message: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """M5E.1 spec section 9: one record per real model call, separate
        from the per-decision aggregate `emit_llm_decision` already
        records. No-ops without a telemetry sink, same as every other
        emit_* call site in this class."""
        if not self.telemetry:
            return
        normalized_message = normalize_provider_error_message(error_message)
        self.telemetry.emit_provider_call(
            episode_id=self.episode_id,
            actor_id=self.actor_id,
            decision_id=decision_id or "unknown",
            decision_number=decision_number,
            call_index=call_index,
            call_type=call_type,
            runtime=self.agent_runtime,
            provider=self.provider_name,
            model=self.model_name,
            condition=self.config.condition,
            start_time=start_iso,
            end_time=datetime.now(timezone.utc).isoformat(),
            latency_ms=round(latency_ms, 2),
            success=success,
            provider_error_type=classify_provider_call_error(error_type, error_message),
            provider_error_message_normalized=normalized_message,
            timeout=(classify_provider_call_error(error_type, error_message) == PROVIDER_TIMEOUT),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _record_invalid_signature(self, response: dict[str, Any] | None) -> int:
        """M6B.2 spec section 11: increments and returns the running count
        for this exact invalid (action, target, trap_type) signature within
        this episode - see `invalid_action_signature`."""
        response = response or {}
        signature = invalid_action_signature(response.get("action"), response.get("target_id"), response.get("trap_type"))
        self._invalid_signature_counts[signature] = self._invalid_signature_counts.get(signature, 0) + 1
        return self._invalid_signature_counts[signature]

    def _emit_decision_telemetry(self, **kwargs: Any) -> None:
        if not self.telemetry:
            return
        memory_kw = self._get_memory_telemetry_kwargs()
        self.telemetry.emit_llm_decision(
            episode_id=self.episode_id,
            actor_id=self.actor_id,
            provider=self.provider_name,
            model=self.model_name,
            prompt_version=self.config.prompt_version,
            observation_schema_version=OBSERVATION_SCHEMA_VERSION,
            action_schema_version=self.config.action_schema_version,
            agent_runtime=self.agent_runtime,
            agent_runtime_version=self.agent_runtime_version,
            **kwargs,
            **memory_kw,
        )

    def _finish_action_lifecycle(self, observation: Observation, boundary: str) -> None:
        """Called once an accepted action reaches a lifecycle boundary
        (spec section 5). Delegates to the shared ActionExecutor.finish()
        (spec section 11-12) so LLMController and StrandsAgentAdapter apply
        identical counter/memory/telemetry semantics from one definition.
        """
        self._executor.finish(
            memory=self.memory,
            telemetry=self.telemetry,
            episode_id=self.episode_id,
            actor_id=self.actor_id,
            agent_runtime=self.agent_runtime,
            agent_runtime_version=self.agent_runtime_version,
            observation=observation,
            boundary=boundary,
        )
        forget_digest_after_unreachable_move(self, boundary)

    def _request_action(self, observation: Observation, observation_payload: dict[str, Any]) -> GameAction | None:
        # Bugfix mission time budget V2: time budget + action costs, live
        # workshop payloads only (set by LiveGame); built here, on the request
        # thread, because the MOVE ETAs need path searches.
        self._last_request_facts = None
        if self.config.live_time_facts:
            observation_payload = annotate_live_payload(observation, observation_payload)
            self._last_request_facts = {"time_budget": observation_payload["time_budget"],
                                        "action_costs": observation_payload["action_costs"]}
        legal_actions = observation_payload["legal_actions"]
        output_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(VALID_ACTION_NAMES) + (
                    list(ENCOUNTER_ACTIONS) if getattr(observation, "encounters_enabled", False) else []) + (
                    ["enter_safe_code"] if (getattr(observation, "puzzle", None) or {}).get("type") == "riddle" else [])},
                "target_id": {"type": ["string", "null"]},
                "trap_type": {"type": ["string", "null"]},
                "reason": {"type": ["string", "null"]},
            },
            "required": ["action"],
        }
        if (getattr(observation, "puzzle", None) or {}).get("type") == "riddle":
            output_schema["properties"]["code"] = {"type": ["string", "null"]}  # live riddle: enter_safe_code
        # M5F.2.2/M5F.2.3: `message` means up to three different things
        # depending on cost mode, so its schema shape is mode-specific -
        # FULL_COST keeps the exact pre-M5F.2.2 string shape (unchanged,
        # spec section 3's backward-compatibility requirement);
        # ZERO_DECISION_COST (nested) replaces it with the optional
        # envelope object (M5F.2.2 spec section 2); flat mode (M5F.2.3
        # spec section 6) drops the nested `message` key entirely in favor
        # of two flat top-level fields, for model-output reliability.
        #
        # M5F.3B bug fix: the pre-existing `else` branch below used to add
        # a legacy `message` field to the schema unconditionally, even when
        # `communication_enabled` is False - SOCIAL_OFF's own legality gate
        # (see `enumerate_legal_actions`) already made choosing it illegal,
        # but the model-facing JSON schema itself still advertised message
        # capability that didn't exist. M5F.3B section 6 requires SOCIAL_OFF
        # to "genuinely omit message capability from model output," so the
        # field is now omitted entirely rather than merely made illegal.
        if not self.config.communication_enabled:
            pass
        elif self.config.zero_decision_cost_messaging and self.config.flat_message_schema:
            output_schema["properties"]["message_to"] = {"type": ["string", "null"]}
            output_schema["properties"]["message_text"] = {"type": ["string", "null"]}
        elif self.config.zero_decision_cost_messaging:
            output_schema["properties"]["message"] = {
                "type": ["object", "null"],
                "properties": {
                    "recipient_id": {"type": ["string", "null"]},
                    "text": {"type": ["string", "null"]},
                },
            }
        else:
            output_schema["properties"]["message"] = {"type": ["string", "null"]}

        # M5F.2.3 spec section 3: one increment per genuine strategic
        # decision (this method, called once per decision - repairs stay
        # within the same call, never a second increment). Checked/shown
        # further below using the value AFTER this increment, matching
        # the "available again at decision K+3" arithmetic documented on
        # `_message_cooldown_active`.
        if self._decisions_since_last_message is not None:
            self._decisions_since_last_message += 1
        # M5F.2.3: computed once, used both for the actual model-payload
        # injection below and for the telemetry fields at both emission
        # call sites further down - `None` (not `False`/`0`) unless the
        # bounded rule is actually active, so a query over historical
        # events can tell "unbounded/FULL_COST, this field is meaningless"
        # apart from "bounded and available right now."
        comm_available = not self._message_cooldown_active() if (self.config.zero_decision_cost_messaging and self.config.message_cooldown_decisions > 0) else None
        comm_next_available_in = self._decisions_until_available() if (self.config.zero_decision_cost_messaging and self.config.message_cooldown_decisions > 0) else None
        start = time.monotonic()
        decision_tick = self._decision_seq
        decision_game_time = float(self._decision_seq) * self._decision_interval_seconds
        observation_hash = hashlib.sha256(json.dumps(observation_payload, sort_keys=True).encode("utf-8")).hexdigest()

        # Build a compact model-facing payload to reduce prompt size and
        # token usage, keeping the authoritative `observation_payload` intact
        # for in-process validation.
        model_payload = compact_observation_for_model(observation_payload, self.config.representation_variant)
        # M5B: Include strategic memory in model payload
        model_payload["strategy_memory"] = self.memory.to_compact_dict()
        # M5F.2.3 spec section 10: availability shown mechanically, never
        # a behavioral nudge - only when the bounded rule is actually
        # active (message_cooldown_decisions > 0); every other caller's
        # "communication" payload is untouched by this.
        if comm_available is not None:
            model_payload["communication"] = dict(model_payload.get("communication") or {})
            model_payload["communication"]["available_now"] = comm_available
            model_payload["communication"]["next_available_in_decisions"] = comm_next_available_in
        if self.config.model_payload_transform is not None:
            model_payload = self.config.model_payload_transform(model_payload)
        model_payload_json = json.dumps(model_payload, sort_keys=True,
                                        **({"separators": (",", ":")} if self.config.compact_payload_json else {}))
        observation_size = len(json.dumps(observation_payload, sort_keys=True))
        model_payload_size = len(model_payload_json)
        serialized_legal_actions_chars = len(json.dumps(legal_actions, sort_keys=True))

        # Generate a decision ID for this LLM request
        decision_id = str(uuid.uuid4())
        self._last_decision_id = decision_id
        call_start_iso = datetime.now(timezone.utc).isoformat()

        try:
            response = self.model_client.generate_action(
                self._effective_system_prompt(),
                model_payload_json,
                output_schema,
                request_context={"episode_id": self.episode_id, "actor_id": self.actor_id},
            )
        except Exception as exc:  # pragma: no cover - defensive safety boundary
            self.status = "ERROR"
            self.last_reason = f"provider exception: {exc}"
            self.invalid_calls += 1
            self.fallback_count += 1
            self.provider_failure_count += 1
            self._emit_provider_call(
                decision_id=decision_id,
                decision_number=decision_tick,
                call_index=1,
                call_type="initial",
                start_iso=call_start_iso,
                latency_ms=(time.monotonic() - start) * 1000.0,
                success=False,
                error_type="provider_error",
                error_message=str(exc),
            )
            self._emit_decision_telemetry(
                tick=decision_tick,
                game_time=decision_game_time,
                observation_hash=observation_hash,
                observation=observation_payload,
                legal_actions=legal_actions,
                raw_response={"error": str(exc)},
                observation_size_chars=observation_size,
                model_payload_size_chars=model_payload_size,
                validated_action=None,
                valid=False,
                repair_attempted=False,
                repair_raw_response=None,
                repair_valid=None,
                repair_error_type=None,
                initial_model_output_valid=False,
                initial_error_type="provider_error",
                final_model_action_valid=False,
                fallback_used=True,
                fallback_action={"action": "wait"},
                executed_action={"action": "wait"},
                executed_action_valid=True,
                action_source="fallback",
                latency_ms=round((time.monotonic() - start) * 1000.0, 2),
                error_type="provider_error",
                error_message=str(exc),
                decision_id=decision_id,
                legal_actions_count=len(legal_actions),
                initial_proposal={"error": str(exc)},
                recovery_outcome=_recovery_outcome(False, False, None, runtime_failure=True),
                representation_variant=self.config.representation_variant,
                decision_number=decision_tick,
                condition=self.config.condition,
                personality_profile=self.config.personality_profile,
                serialized_legal_actions_chars=serialized_legal_actions_chars,
                communication_available=comm_available,
                communication_next_available_in_decisions=comm_next_available_in,
            )
            # M5B: Record fallback decision in memory (independent of telemetry).
            self.memory.record_decision(
                game_time=decision_game_time,
                decision_id=decision_id,
                selected_action="wait",
                target_id=None,
                valid=False,
                action_source="fallback",
                observation_hash=observation_hash,
            )
            self._last_action_source = "fallback"
            self._memory_tracker.clear_pending_decision()
            self.decisions_completed += 1
            return GameAction(ActionType.WAIT)

        self.calls += 1
        normalized = _coerce_model_response(response, self.provider_name, self.model_name)
        raw_response = _normalize_target_id(normalized.payload or {})
        if self.config.model_response_transform is not None:
            raw_response = self.config.model_response_transform(raw_response)
        latency_ms = normalized.latency_ms or 0.0
        time_to_first_token_ms = normalized.time_to_first_token_ms
        input_tokens = normalized.input_tokens or 0
        output_tokens = normalized.output_tokens or 0
        cached_input_tokens = normalized.cached_input_tokens
        finish_reason = normalized.finish_reason
        provider_request_id = normalized.request_id

        # M5B.1: strict validation gate - a malformed/missing/unrecognized
        # response is never silently equivalent to {"action": "wait"} here.
        initial_validation = self._validate_model_output(observation, raw_response, legal_actions)
        # Prefer the provider's own diagnostic (e.g. "json_parse_error") when
        # it has one - it's more specific than the generic "empty_response"
        # the strict validator falls back to for an empty payload.
        initial_error_type = normalized.error_type or initial_validation.error_type
        self._emit_provider_call(
            decision_id=decision_id,
            decision_number=decision_tick,
            call_index=1,
            call_type="initial",
            start_iso=call_start_iso,
            latency_ms=latency_ms,
            success=(classify_provider_call_error(initial_error_type, initial_validation.error_message) is None),
            error_type=initial_error_type,
            error_message=initial_validation.error_message,
            input_tokens=normalized.input_tokens,
            output_tokens=normalized.output_tokens,
        )

        repair_attempted = False
        repair_raw_response: dict[str, Any] | None = None
        repair_validation: ModelOutputValidation | None = None
        repair_context_chars: int | None = None
        final_validation = initial_validation

        if not initial_validation.valid and self.config.max_retries > 0:
            repair_attempted = True
            self.repair_count += 1
            repair_validation, repair_raw_response, repair_model_response, repair_context_chars = self._attempt_repair(
                observation, legal_actions, output_schema, model_payload, initial_validation, raw_response,
                decision_id=decision_id, decision_number=decision_tick,
            )
            if repair_model_response is not None:
                self.calls += 1
                latency_ms += repair_model_response.latency_ms or 0.0
                input_tokens += repair_model_response.input_tokens or 0
                output_tokens += repair_model_response.output_tokens or 0
                finish_reason = repair_model_response.finish_reason or finish_reason
                provider_request_id = repair_model_response.request_id or provider_request_id
            if repair_validation.valid:
                final_validation = repair_validation

        final_model_action_valid = final_validation.valid
        fallback_used = not final_model_action_valid
        action_source = "llm" if final_model_action_valid else "fallback"
        executed_action_dict = final_validation.validated_action if final_model_action_valid else {"action": "wait"}

        # M5F.2.3 spec section 3: reset the moment a message is actually
        # accepted as part of the final validated decision - this is the
        # one and only place `_decisions_since_last_message` goes back to
        # 0, regardless of nested vs. flat representation (both funnel
        # into the same `accompanying_message` shape).
        if final_model_action_valid and executed_action_dict.get("accompanying_message") is not None:
            self._decisions_since_last_message = 0

        # M5D.2: hand the accepted decision to the ActionExecutor rather
        # than using ModelOutputValidation's own one-shot GameAction
        # directly - for MOVE this replaces a straight-line direction
        # computed once with real multi-tick A* pathfinding (the actual
        # M5D.1 root-cause fix; see action_lifecycle.py). For every other
        # action type the executor's result is equivalent to
        # final_validation.action, just routed through the same lifecycle
        # bookkeeping for consistency.
        if final_model_action_valid and self._live_async:
            # Live async: not started here, on the request thread against the
            # request-time world - the game thread classifies and starts it
            # at commit (live_async.commit_ready_decision).
            self._live_pending = live_async.make_pending(
                self, decision_id=decision_id, action_dict=executed_action_dict, valid=True,
                decision_game_time=decision_game_time,
            )
            executed_action = GameAction(ActionType.WAIT)
        elif final_model_action_valid:
            normalized_action = executed_action_dict.get("action", "wait")
            target_id = executed_action_dict.get("target_id")
            trap_type = executed_action_dict.get("trap_type")
            message_text = executed_action_dict.get("message")
            accompanying_message = executed_action_dict.get("accompanying_message")
            executed_action, start_boundary, _action_id = self._executor.start(
                observation, decision_id, normalized_action, target_id, trap_type, decision_game_time,
                message=message_text, accompanying_message=accompanying_message,
                # live riddle: the code entered at the safe (live_async passes it via _message_kwargs)
                code=executed_action_dict.get("code"),
            )
            if start_boundary is not None:
                self._finish_action_lifecycle(observation, start_boundary)
        else:
            executed_action = GameAction(ActionType.WAIT)

        if final_model_action_valid:
            reason_source = repair_raw_response if repair_attempted and final_validation is repair_validation else raw_response
            self.last_reason = (reason_source or {}).get("reason") or (
                "repaired model action" if repair_attempted else "validated model action"
            )
            final_error_type = None
            final_error_message = None
            same_invalid_action_signature_count = None
        else:
            self.invalid_calls += 1
            self.fallback_count += 1
            self.status = "ERROR"
            self.last_reason = "invalid model output rejected"
            if repair_attempted and repair_validation is not None:
                final_error_type = repair_validation.error_type
                final_error_message = repair_validation.error_message
                same_invalid_action_signature_count = self._record_invalid_signature(repair_raw_response)
            else:
                final_error_type = initial_error_type
                final_error_message = initial_validation.error_message
                same_invalid_action_signature_count = self._record_invalid_signature(raw_response)
            # M5F.3B.1 spec section 12: a repair attempt can itself fail
            # via a genuine provider exception (`_attempt_repair`'s own
            # `except Exception` branch returns a validation object with
            # `error_type="provider_error"`) - that must still count as a
            # provider/infrastructure failure, not a model-behavior
            # validation failure, even though it's discovered here in the
            # same "final action still invalid" branch as a real
            # hallucinated-parameter rejection.
            if final_error_type == "provider_error":
                self.provider_failure_count += 1
            else:
                self.validation_failure_count += 1
            if repair_attempted:
                self.repair_failure_count += 1

        self._emit_decision_telemetry(
            tick=decision_tick,
            game_time=decision_game_time,
            observation_hash=observation_hash,
            observation=observation_payload,
            legal_actions=legal_actions,
            raw_response=raw_response,
            validated_action=executed_action_dict if final_model_action_valid else None,
            valid=final_model_action_valid,
            repair_attempted=repair_attempted,
            repair_raw_response=repair_raw_response,
            repair_valid=(repair_validation.valid if repair_validation is not None else None),
            repair_error_type=(repair_validation.error_type if repair_validation is not None else None),
            initial_model_output_valid=initial_validation.valid,
            initial_error_type=initial_error_type,
            final_model_action_valid=final_model_action_valid,
            fallback_used=fallback_used,
            fallback_action=({"action": "wait"} if fallback_used else None),
            executed_action=executed_action_dict,
            executed_action_valid=True,
            action_source=action_source,
            latency_ms=latency_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            finish_reason=finish_reason,
            provider_request_id=provider_request_id,
            error_type=final_error_type,
            error_message=final_error_message,
            observation_size_chars=observation_size,
            model_payload_size_chars=model_payload_size,
            legal_actions_count=len(legal_actions),
            decision_id=decision_id,
            initial_proposal=raw_response,
            repair_context_version=(REPAIR_PROMPT_VERSION if repair_attempted else None),
            repair_legal_action_count=(len(legal_actions) if repair_attempted else None),
            repair_near_object_id=(observation.nearby_object_id if repair_attempted else None),
            recovery_outcome=_recovery_outcome(
                initial_validation.valid,
                repair_attempted,
                repair_validation.valid if repair_validation is not None else None,
            ),
            representation_variant=self.config.representation_variant,
            decision_number=decision_tick,
            condition=self.config.condition,
            personality_profile=self.config.personality_profile,
            communication_available=comm_available,
            communication_next_available_in_decisions=comm_next_available_in,
            serialized_legal_actions_chars=serialized_legal_actions_chars,
            repair_context_chars=repair_context_chars,
            same_invalid_action_signature_count=same_invalid_action_signature_count,
            total_tokens=(input_tokens + output_tokens) if (input_tokens or output_tokens) else None,
        )

        # M5B: Record the decision in memory. Independent of telemetry being
        # configured - memory is a first-class subsystem, not a byproduct of
        # logging. Per M5B.1 section 13: a fallback WAIT is recorded with
        # action_source="fallback", never misattributed as "LLM chose wait".
        selected_action = executed_action_dict.get("action", "wait") if final_model_action_valid else "wait"
        target_for_memory = executed_action_dict.get("target_id") if final_model_action_valid else None
        self.memory.record_decision(
            game_time=decision_game_time,
            decision_id=decision_id,
            selected_action=selected_action,
            target_id=target_for_memory,
            valid=final_model_action_valid,
            action_source=action_source,
            observation_hash=observation_hash,
        )
        self._last_action_source = action_source
        if final_model_action_valid:
            self._memory_tracker.set_pending_decision(decision_id, selected_action, target_for_memory, observation.nearby_object_id)
        else:
            self._memory_tracker.clear_pending_decision()
        self.decisions_completed += 1

        return executed_action

    def _request_action_background(self, observation: Observation, observation_payload: dict[str, Any]) -> None:
        try:
            decision = self._request_action(observation, observation_payload)
        except Exception:
            decision = GameAction(ActionType.WAIT)
        with self._thinking_lock:
            if decision is None:
                decision = GameAction(ActionType.WAIT)
            if self._live_async and self._live_pending is None:
                # rejected output / provider failure: the fallback WAIT is
                # committed through the same live path (and recorded)
                self._live_pending = live_async.make_pending(
                    self, decision_id=self._last_decision_id, action_dict={"action": "wait"}, valid=False,
                    decision_game_time=float(self._decision_seq) * self._decision_interval_seconds,
                )
            # M6G.3: RESPONSE_READY_UNCOMMITTED is published before the
            # decision itself, so any reader that sees the new `last_action`
            # also sees the flag (see _inflight_action), and before the
            # in-flight flag clears.
            self.response_ready_at = time.monotonic()
            self._response_ready_uncommitted = True
            self.last_action = decision
            self.status = "EXECUTING" if decision.type != ActionType.WAIT else "IDLE"
            self._thinking_in_progress = False

    def _convert_response_to_action(
        self,
        observation: Observation,
        response: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> GameAction | None:
        return convert_response_to_action(observation, response, legal_actions)
