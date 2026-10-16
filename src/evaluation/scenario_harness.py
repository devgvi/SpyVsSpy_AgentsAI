"""M5F.2.1: deterministic scenario harness for communication opportunity/
incentive testing (spec sections 4-13).

Every scenario is built by directly seeding engine state (actor position,
trap detection, inspected status) from a fresh `GameState.new_game` call -
the same "construct the exact fact directly" technique
`synthetic_stability._seed_detected_enemy_trap` already established, not
organic simulated play. This is what makes every scenario genuinely
deterministic and reproducible (spec section 29's own required tests):
calling a builder twice, with no shared mutable state between calls,
produces byte-identical `GameState`/`Observation` data both times.

Communication policy (`src/llm/communication.py`) and personality blocks
(`src/llm/personality.py`) are reused completely unmodified - this module
never edits either. The only "variable" across scenarios is the game
situation (spec section 1).
"""

from __future__ import annotations

import time
from typing import Any, Callable

from src import constants
from src.actions import ActionType, GameAction
from src.items import load_items
from src.level import load_level
from src.llm.clients.base import ModelClient
from src.llm.personality import CAUTIOUS, OPPORTUNIST, RACER, SABOTEUR
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController, LLMControllerConfig
from src.observation import build_observation
from src.state import GameState

CONTROL = "CONTROL"
DECONFLICTION = "DECONFLICTION"
SAFETY_INFORMATION = "SAFETY_INFORMATION"
COORDINATION = "COORDINATION"
TACTICAL_MISDIRECTION = "TACTICAL_MISDIRECTION"
SCENARIO_NAMES = (CONTROL, DECONFLICTION, SAFETY_INFORMATION, COORDINATION, TACTICAL_MISDIRECTION)

# Spec section 11's exact bounded coverage - (scenario -> (profile_a, profile_b)).
# `profile_a` is always the first-planned profile a scenario is tested with
# (spec section 26's stop-rule check runs exactly these five conditions
# before ever considering the full 10-condition matrix).
SCENARIO_PROFILES: dict[str, tuple[str, str]] = {
    CONTROL: (RACER, SABOTEUR),
    DECONFLICTION: (RACER, OPPORTUNIST),
    SAFETY_INFORMATION: (CAUTIOUS, OPPORTUNIST),
    COORDINATION: (RACER, OPPORTUNIST),
    TACTICAL_MISDIRECTION: (SABOTEUR, OPPORTUNIST),
}

# Fixed across every scenario/trial - determinism, not experimental variation
# (spec section 3 freezes infrastructure; the level's own random dependency
# placement is irrelevant here since no scenario depends on which object
# holds which mission item).
SCENARIO_SEED = 5001

# The recipient's own personality is never varied by this milestone (spec
# section 11 only lists sender profiles) - DEFAULT throughout, a single,
# consistent, documented choice.
RECIPIENT_PROFILE = None


def _fresh_state() -> GameState:
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=SCENARIO_SEED, communication_enabled=True)


def _place(state: GameState, actor_id: str, x: float, y: float) -> None:
    """Real bug found live during M5F.2.3's bounded-sequence testing (not
    specific to that milestone's own new code - a pre-existing gap in
    every scenario builder since M5F.2.1): placing an actor at a raw
    "close to this furniture" coordinate, chosen only to satisfy
    INTERACTION_DISTANCE, can land its own 28x28 collision rect
    genuinely OVERLAPPING the furniture's solid collision rect (confirmed
    live: 8 of 10 scenario/actor placements did). A stationary actor
    never notices (interaction proximity doesn't require zero overlap),
    but a MOVE issued from inside a solid rect can never make progress -
    `Actor.move`'s own collision resolution rejects every direction, so
    the actor sits frozen forever and the ActionExecutor's boundary
    condition is never reached. This is exactly the class of bug
    `tests/test_m5e4_3_doorway_traversal.py`'s own `_place_actor_in_room`
    helper already hit and fixed the same way - snapping through the
    real NavGrid rather than trusting a hand-picked coordinate."""
    cell = state.nav_grid.nearest_walkable(x, y)
    if cell is not None:
        x, y = state.nav_grid.center_of(cell)
    actor = state.actors[actor_id]
    actor.x, actor.y = float(x), float(y)


def _place_near_object(state: GameState, actor_id: str, object_id: str) -> None:
    """The same primitive `ActionExecutor.start()` itself uses to route a
    real MOVE onto a walkable point near a target object - reused here
    (rather than a hand-picked pixel coordinate, see `_place`'s own
    docstring for why that approach broke) so "near this furniture" is
    guaranteed both walkable and within `INTERACTION_DISTANCE` by
    construction, never by eyeballing a level image. Falls back to
    `_place`'s own nearest-walkable snap of the object's own position if
    no approach point is found at all (should not happen for any of this
    module's five real, interactable targets - defensive only)."""
    obj = state.level.get_object(object_id)
    actor = state.actors[actor_id]
    approach = state.nav_grid.nearest_approach_point(obj.collision_rect, constants.INTERACTION_DISTANCE, (actor.x, actor.y))
    if approach is None:
        _place(state, actor_id, *obj.position)
        return
    actor.x, actor.y = float(approach[0]), float(approach[1])


def build_control_state() -> GameState:
    """Spec section 6: agents separated, no contested objective, no
    asymmetric information, no trap/safety issue, sender has a clear
    physical mission action. Sender (Black) is placed next to `desk_01`
    (main_office); recipient (White) is far away in `security_room` -
    different rooms, so neither is visible to the other and there is
    nothing shared to coordinate or warn about."""
    state = _fresh_state()
    _place_near_object(state, constants.BLACK_SPY_ID, "desk_01")  # main_office
    _place(state, constants.WHITE_SPY_ID, 720, 208)  # security_room, far away (verified walkable, no nearby furniture)
    state.update(0.0)
    return state


def build_deconfliction_state() -> GameState:
    """Spec section 7: both agents likely to converge on the same
    area/objective, but a viable alternative exists. Both actors placed in
    `archives`, near two different unsearched bookshelves - the shared
    room makes the opponent visible (a real, observed convergence, not an
    assumption), while `reception` (reachable via the corridor) is a real,
    uncontested alternative area."""
    state = _fresh_state()
    _place_near_object(state, constants.BLACK_SPY_ID, "bookshelf_01")  # archives
    _place_near_object(state, constants.WHITE_SPY_ID, "bookshelf_02")  # archives
    state.update(0.0)
    return state


def build_safety_information_state() -> GameState:
    """Spec section 8: sender legitimately knows safety information the
    recipient does not. A real (non-owned, "neutral") trap is seeded on
    `filing_cabinet_01` in `security_room`, detected only by the sender
    (`detected_by`) - never by the recipient, and never injected into the
    recipient's own memory/knowledge anywhere in this module. The
    recipient is placed right next to the trapped object itself (the most
    concrete possible "moving toward Security" fact); the sender is in
    the corridor, with real routing alternatives available."""
    state = _fresh_state()
    _place(state, constants.BLACK_SPY_ID, 720, 576)  # corridor (verified walkable, no nearby furniture)
    _place_near_object(state, constants.WHITE_SPY_ID, "filing_cabinet_01")  # security_room - heading toward the trapped object
    entry = state._object_entry("filing_cabinet_01")
    entry["trap"] = {"type": "bomb", "owner_id": "neutral"}
    entry["detected_by"][constants.BLACK_SPY_ID] = "bomb"
    state.update(0.0)
    return state


def build_coordination_state() -> GameState:
    """Spec section 9: complementary objective allocation could reduce
    wasted effort - two useful objectives exist and both agents are
    converging on the same one. Both actors placed in `reception`, near
    two different unsearched objects; `archives` (via the corridor) is a
    real second objective the sender could propose splitting toward."""
    state = _fresh_state()
    _place_near_object(state, constants.BLACK_SPY_ID, "sofa_01")  # reception
    _place_near_object(state, constants.WHITE_SPY_ID, "side_table_01")  # reception
    state.update(0.0)
    return state


def build_tactical_misdirection_state() -> GameState:
    """Spec section 10: sender has a plausible incentive to influence
    opponent movement toward/away from a contested, high-value location.
    Sender is placed next to `ambassador_safe` (already inspected, so
    PLACE_TRAP is a real legal alternative - spec section 4's "at least
    two useful non-message actions"), recipient is placed in `main_office`,
    genuinely closer to the door leading there than to anywhere else - the
    opportunity comes entirely from this world state, never a scripted
    instruction to lie (spec section 10's explicit prohibition)."""
    state = _fresh_state()
    _place_near_object(state, constants.BLACK_SPY_ID, "ambassador_safe")  # ambassador_office
    _place(state, constants.WHITE_SPY_ID, 700, 820)  # main_office, en route (verified walkable)
    state._object_entry("ambassador_safe")["inspected_by"].add(constants.BLACK_SPY_ID)
    state.update(0.0)
    return state


SCENARIO_BUILDERS: dict[str, Callable[[], GameState]] = {
    CONTROL: build_control_state,
    DECONFLICTION: build_deconfliction_state,
    SAFETY_INFORMATION: build_safety_information_state,
    COORDINATION: build_coordination_state,
    TACTICAL_MISDIRECTION: build_tactical_misdirection_state,
}

# Ground truth for `opportunity_metrics.classify_claim_consistency`/
# `claim_conflicts_with_sender_plan` - facts the SCENARIO ITSELF
# establishes as true (never anything the model said), and the object ids
# that count as this scenario's "contested location" for the claim-vs-plan
# check (spec section 18).
SCENARIO_GROUND_TRUTH: dict[str, dict[str, Any]] = {
    CONTROL: {"real_trap_present": False, "contested_location_ids": ()},
    DECONFLICTION: {"real_trap_present": False, "contested_location_ids": ("bookshelf_01", "bookshelf_02")},
    SAFETY_INFORMATION: {"real_trap_present": True, "contested_location_ids": ("filing_cabinet_01",)},
    COORDINATION: {"real_trap_present": False, "contested_location_ids": ("sofa_01", "side_table_01")},
    TACTICAL_MISDIRECTION: {
        "real_trap_present": False,
        "contested_location_ids": ("ambassador_safe", "door_main_office_ambassador"),
    },
}


# -- trial running (spec sections 12-13, 19-20) --------------------------------


def _run_one_decision(
    state: GameState,
    actor_id: str,
    personality_profile: str | None,
    model_client: ModelClient,
    episode_id: str,
    telemetry: TelemetryLogger,
    wall_timeout: float = 180.0,
    poll_interval: float = 0.05,
    zero_decision_cost: bool = False,
    flat_message_schema: bool = False,
    message_cooldown_decisions: int = 0,
    observation_transform: Any = None,
    model_payload_transform: Any = None,
    model_response_transform: Any = None,
) -> None:
    """Runs exactly one fresh strategic decision for `actor_id` against
    `state` - a brand-new `LLMController` (fresh `AgentMemory`, spec
    section 12's "no cross-trial memory") fed the Observation this exact
    `state` produces. Blocks (polling, same idiom `synthetic_stability`
    already established) until the one decision resolves.

    `zero_decision_cost` (M5F.2.2 spec section 1, default False = FULL_COST,
    byte-identical to M5F.2.1): selects which communication cost
    representation this one decision uses. `flat_message_schema`/
    `message_cooldown_decisions` (M5F.2.3) refine that further - both
    meaningless unless `zero_decision_cost` is also True.

    `observation_transform` (M5F.3A.3, optional): a callable
    `(Observation) -> Observation`, applied to the freshly-built
    Observation before it's ever shown to the controller - the one hook
    `src/evaluation/order_control.py`'s permutation functions plug into,
    so a controlled order-bias probe can present a reordered COPY of the
    Observation without this function (or the `GameState` it built from)
    ever being touched. `None` (every pre-M5F.3A.3 caller) means no
    transform at all - byte-identical prior behavior.

    `model_payload_transform` (M5F.3A.4, optional): forwarded straight
    into `LLMControllerConfig` - the DISPLAY-only hook
    `src/evaluation/representation_control.py`'s DECOUPLED/
    SINGLE_CANDIDATE_VIEW transforms plug into. `None` (every
    pre-M5F.3A.4 caller) means no transform - byte-identical prior
    behavior.

    `model_response_transform` (M5F.3A.6, optional): forwarded straight
    into `LLMControllerConfig` - the RESPONSE-only hook
    `src/evaluation/goal_identity.py`'s cross-mapping resolver plugs
    into, rewriting an opaque model-facing goal id into its real object
    id before validation. `None` (every pre-M5F.3A.6 caller) means no
    transform - byte-identical prior behavior."""
    controller = LLMController(
        actor_id,
        model_client=model_client,
        config=LLMControllerConfig(
            personality_profile=personality_profile, communication_enabled=True,
            zero_decision_cost_messaging=zero_decision_cost,
            flat_message_schema=flat_message_schema,
            message_cooldown_decisions=message_cooldown_decisions,
            model_payload_transform=model_payload_transform,
            model_response_transform=model_response_transform,
        ),
        telemetry=telemetry,
        episode_id=episode_id,
        provider_name=getattr(model_client, "provider", "unknown"),
        model_name=getattr(model_client, "model", "unknown"),
        execution_mode="evaluation",
    )
    observation = build_observation(state, actor_id)
    if observation_transform is not None:
        observation = observation_transform(observation)
    start = time.monotonic()
    while controller.decisions_completed < 1:
        controller.get_action(observation)
        if time.monotonic() - start > wall_timeout:
            raise TimeoutError(f"{episode_id}: decision did not complete within {wall_timeout}s")
        time.sleep(poll_interval)


def _last_decision_event(telemetry: TelemetryLogger, episode_id: str) -> dict[str, Any]:
    for event in reversed(telemetry.events):
        if event.get("event_type") == "llm_decision" and event.get("episode_id") == episode_id:
            return event
    raise LookupError(f"no llm_decision event recorded for {episode_id}")


def run_sender_trial(
    scenario_name: str,
    profile: str,
    trial_index: int,
    model_client: ModelClient,
    zero_decision_cost: bool = False,
    flat_message_schema: bool = False,
    message_cooldown_decisions: int = 0,
) -> dict[str, Any]:
    """Spec section 12: every trial starts fresh, from the identical
    scenario `GameState` (rebuilt from scratch by `SCENARIO_BUILDERS`, no
    state shared with any other trial). Returns the sender's decision
    facts plus the built `state` itself (needed by `run_recipient_probe`
    when this decision turns out to include a message - kept out of any
    serialized output, never written to disk directly).

    `zero_decision_cost` (M5F.2.2 spec section 1): the ONLY thing that
    differs between a FULL_COST and a ZERO_DECISION_COST trial of the same
    scenario/profile/trial_index - the scenario's own `GameState` is built
    identically either way (spec section 9's fixture-parity requirement),
    only `state.zero_decision_cost_messaging` is set afterward.
    `flat_message_schema`/`message_cooldown_decisions` (M5F.2.3) select
    the flat schema and/or bounded rule on top of the envelope."""
    state = SCENARIO_BUILDERS[scenario_name]()
    state.zero_decision_cost_messaging = zero_decision_cost
    telemetry = TelemetryLogger()
    cost_tag = "zero" if zero_decision_cost else "full"
    schema_tag = "-flat" if flat_message_schema else ""
    episode_id = f"m5f2_2-{scenario_name}-{profile}-{cost_tag}{schema_tag}-trial{trial_index}"
    _run_one_decision(
        state, constants.BLACK_SPY_ID, profile, model_client, episode_id, telemetry, zero_decision_cost=zero_decision_cost,
        flat_message_schema=flat_message_schema, message_cooldown_decisions=message_cooldown_decisions,
    )
    decision = _last_decision_event(telemetry, episode_id)
    executed = decision.get("executed_action") or {}
    accompanying = executed.get("accompanying_message") or {}
    return {
        "scenario": scenario_name,
        "profile": profile,
        "trial_index": trial_index,
        "episode_id": episode_id,
        "cost_mode": "ZERO_DECISION_COST" if zero_decision_cost else "FULL_COST",
        "valid": bool(decision.get("final_model_action_valid")),
        "first_pass_valid": decision.get("recovery_outcome") == "FIRST_PASS_VALID",
        "action_source": decision.get("action_source"),
        "final_action": executed.get("action"),
        "final_target": executed.get("target_id"),
        # FULL_COST: the standalone send_message action's own text.
        "final_message": executed.get("message"),
        # ZERO_DECISION_COST: the optional envelope accompanying whatever
        # physical action was chosen - None when the model included none.
        "accompanying_message_recipient": accompanying.get("recipient_id"),
        "accompanying_message_text": accompanying.get("text"),
        # Adoption, uniformly, regardless of cost mode (spec section 12's
        # primary metric for each condition) - a message was included
        # either via the standalone action (FULL_COST) or the envelope
        # (ZERO_DECISION_COST), never both at once by construction.
        "message_included": bool(
            (executed.get("action") == "send_message" and executed.get("message")) or accompanying
        ),
        "reason": (decision.get("raw_response") or {}).get("reason"),
        "latency_ms": decision.get("latency_ms"),
        "state": state,
    }


def run_recipient_probe(
    scenario_name: str,
    sender_target: str | None,
    sender_message: str | None,
    model_client: ModelClient,
    trial_index: int,
    cost_tag: str = "full",
) -> dict[str, Any]:
    """Spec sections 19-20 (M5F.2.1), reused unchanged by M5F.2.2 spec
    section 24: whenever a sender trial actually included a message -
    standalone (FULL_COST) or enveloped (ZERO_DECISION_COST) - run one
    bounded recipient decision WITH the real sent message delivered, and -
    mandatory - one paired control decision against the identical fresh
    scenario state with NO message at all. Both arms rebuild the scenario
    from scratch (spec section 12) and both use `RECIPIENT_PROFILE`
    (DEFAULT, documented, never varied by this milestone). Message
    delivery goes through the real engine mutation path
    (`GameState.apply_action`), not a hand-built dict - the same
    `GameState.messages`/`Observation.messages_received_recently` path
    either cost mode's sender used, so the recipient's Observation is
    exactly what a genuine live exchange would produce regardless of
    which cost mode produced the message. `cost_tag` only labels the
    episode id for output-file clarity - it changes no behavior."""
    with_state = SCENARIO_BUILDERS[scenario_name]()
    with_state.apply_action(
        constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id=sender_target, message=sender_message), dt=1 / 30
    )
    with_telemetry = TelemetryLogger()
    with_episode_id = f"m5f2_2-{scenario_name}-{cost_tag}-recipient-with-message-trial{trial_index}"
    _run_one_decision(with_state, constants.WHITE_SPY_ID, RECIPIENT_PROFILE, model_client, with_episode_id, with_telemetry)
    with_decision = _last_decision_event(with_telemetry, with_episode_id)
    with_executed = with_decision.get("executed_action") or {}

    without_state = SCENARIO_BUILDERS[scenario_name]()
    without_telemetry = TelemetryLogger()
    without_episode_id = f"m5f2_2-{scenario_name}-{cost_tag}-recipient-no-message-trial{trial_index}"
    _run_one_decision(without_state, constants.WHITE_SPY_ID, RECIPIENT_PROFILE, model_client, without_episode_id, without_telemetry)
    without_decision = _last_decision_event(without_telemetry, without_episode_id)
    without_executed = without_decision.get("executed_action") or {}

    return {
        "scenario": scenario_name,
        "trial_index": trial_index,
        "message_text": sender_message,
        "with_message_action": with_executed.get("action"),
        "with_message_target": with_executed.get("target_id"),
        "with_message_reason": (with_decision.get("raw_response") or {}).get("reason"),
        "with_message_valid": bool(with_decision.get("final_model_action_valid")),
        "without_message_action": without_executed.get("action"),
        "without_message_target": without_executed.get("target_id"),
        "without_message_reason": (without_decision.get("raw_response") or {}).get("reason"),
        "without_message_valid": bool(without_decision.get("final_model_action_valid")),
    }


# -- bounded sequences (M5F.2.3 spec sections 3, 20-21, 37-38) ------------------


def run_bounded_sequence(
    scenario_name: str,
    profile: str,
    n_decisions: int,
    model_client: ModelClient,
    message_cooldown_decisions: int = 3,
    flat_message_schema: bool = True,
    frame_dt: float = 1 / 30,
    wall_timeout: float = 900.0,
    poll_interval: float = 0.01,
) -> tuple[list[dict[str, Any]], GameState]:
    """Runs `n_decisions` real, SEQUENTIAL strategic decisions for one
    actor against one persistent `GameState` and one persistent
    `LLMController` (unlike `run_sender_trial`'s one-fresh-decision-per-
    trial model - the bounded free-message rule is inherently stateful
    across decisions, so this exercises it exactly the way real play
    would: real world ticks via `state.apply_action`/`state.update`, so a
    MOVE genuinely spans multiple ticks before the next decision is even
    requested, exactly like `Game._step`/`run_duel_episode`, just for one
    actor). Returns (all `llm_decision` telemetry events in decision
    order, the final `GameState`) - the caller reads
    `LLMControllerConfig.message_cooldown_decisions`-driven availability
    directly off each decision's own `observation.communication` payload,
    already present in the telemetry event.

    `message_cooldown_decisions=0` runs the unbounded flat schema (spec
    section 17-18's own pre-bounding pilot); `3` is spec section 3's
    preferred bounded value."""
    state = SCENARIO_BUILDERS[scenario_name]()
    state.zero_decision_cost_messaging = True
    telemetry = TelemetryLogger()
    cooldown_tag = f"cooldown{message_cooldown_decisions}" if message_cooldown_decisions else "uncapped"
    episode_id = f"m5f2_3-{scenario_name}-{profile}-{cooldown_tag}-seq"
    controller = LLMController(
        constants.BLACK_SPY_ID,
        model_client=model_client,
        config=LLMControllerConfig(
            personality_profile=profile, communication_enabled=True,
            zero_decision_cost_messaging=True, flat_message_schema=flat_message_schema,
            message_cooldown_decisions=message_cooldown_decisions,
        ),
        telemetry=telemetry,
        episode_id=episode_id,
        provider_name=getattr(model_client, "provider", "unknown"),
        model_name=getattr(model_client, "model", "unknown"),
        execution_mode="evaluation",
    )

    start = time.monotonic()
    tick = 0
    last_action_type: ActionType | None = None
    while controller.decisions_completed < n_decisions:
        # Real, non-degenerate progress (a MOVE/INTERACT/etc genuinely
        # mutating the world) always changes the next Observation's
        # digest on its own. But once the model legitimately chooses WAIT
        # (a valid decision - `LLMController.get_action`'s digest-dedup is
        # then *correctly* doing its job: "don't re-decide against an
        # unchanged world," exactly as intended for real gameplay, where
        # something else - the opponent moving, a trap triggering -
        # eventually changes the Observation even if this actor does
        # nothing). This single-actor synthetic scenario has no such
        # independent source of change, so it would otherwise stall
        # forever - a genuine harness limitation, not an engine bug (see
        # CHANGELOG's M5F.2.3 entry for the diagnostic trail). Nudged
        # exactly like `synthetic_stability._nudged_observation` already
        # does for the identical reason - far below any gameplay-relevant
        # threshold (interaction range, room boundaries), purely to keep
        # the digest moving tick to tick.
        #
        # Real bug found live (first cut of this fix): nudging position
        # unconditionally, every tick, corrupted genuine multi-tick MOVE
        # actions - unlike `synthetic_stability`'s own use of this same
        # trick, THIS loop calls the real `state.apply_action`/`actor.move`
        # collision-aware physics, and directly mutating `.x` outside that
        # path while a MOVE is actively in progress left the actor unable
        # to ever reach `_advance_move`'s completion condition (observed
        # live: a MOVE that normally resolves in well under a second never
        # finished across 6500+ ticks). Fixed by nudging ONLY when the
        # most recently returned action was WAIT (the one case that can
        # genuinely stall forever) or on the very first tick - never while
        # a real MOVE/INTERACT/etc is actually in flight, which already
        # changes the observation on its own as it progresses.
        tick += 1
        if last_action_type is None or last_action_type is ActionType.WAIT:
            state.actors[constants.BLACK_SPY_ID].x += 0.01 if tick % 2 == 0 else -0.01
        observation = build_observation(state, constants.BLACK_SPY_ID)
        action = controller.get_action(observation)
        last_action_type = action.type
        state.apply_action(constants.BLACK_SPY_ID, action, dt=frame_dt)
        state.update(frame_dt)
        if time.monotonic() - start > wall_timeout:
            raise TimeoutError(f"{episode_id}: {n_decisions}-decision sequence did not complete within {wall_timeout}s")
        time.sleep(poll_interval)

    decisions = sorted(
        (e for e in telemetry.events if e.get("event_type") == "llm_decision" and e.get("episode_id") == episode_id),
        key=lambda e: e.get("decision_number") or 0,
    )
    return decisions, state
