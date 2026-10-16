"""M5F.1: personality profiles for LLM-controlled spies - a concise,
frozen strategic-priority block appended to the system prompt (spec
sections 8-9), never a capability change (spec sections 7, 11). Each
profile block is deliberately short and stable, not roleplay prose (spec
section 8).

Deliberately NOT the same concept as `src.bot_controller.AgentProfile`
(the scripted opponent's own deterministic decision-weighting config,
unrelated to LLM prompting) - the name "OPPORTUNIST" appears in both
modules because this milestone's spec independently chose it, not because
they share any code or meaning. Never import one where the other is
needed.
"""

from __future__ import annotations

DEFAULT = "default"
RACER = "racer"
CAUTIOUS = "cautious"
SABOTEUR = "saboteur"
OPPORTUNIST = "opportunist"

VALID_PERSONALITY_PROFILES = frozenset({DEFAULT, RACER, CAUTIOUS, SABOTEUR, OPPORTUNIST})


class InvalidPersonalityProfileError(ValueError):
    """Spec section 13: an unrecognized profile name must fail clearly,
    never silently fall back to DEFAULT or any other profile."""


# Spec section 8's own worked example format, frozen before any primary
# run (spec sections 43, 33) - never edited after Stage 1 begins.
_PERSONALITY_BLOCKS: dict[str, str] = {
    RACER: """PERSONALITY: RACER

Behavior priorities:
1. Prioritize mission completion and escape above all else.
2. Prefer direct, reachable mission-progress actions; minimize detours.
3. Avoid unnecessary opponent interaction or pursuit.
4. Use traps only when cheap and directly useful to your own progress.
5. Still respect known hazards - being fast does not mean being careless.""",
    CAUTIOUS: """PERSONALITY: CAUTIOUS

Behavior priorities:
1. Reduce exposure to traps and uncertainty before acting.
2. Inspect objects before risky interaction; disarm known dangers.
3. Avoid unnecessary confrontation with the opponent.
4. Prefer routes and objects you already know are safe.
5. Accept slower mission progress in exchange for lower risk.""",
    SABOTEUR: """PERSONALITY: SABOTEUR

Behavior priorities:
1. Actively look for opportunities to interfere with the opponent's progress.
2. Prefer placing traps at contested or high-value locations.
3. React to opponent sightings; consider delaying or blocking them.
4. Still pursue your own mission - sabotage is a priority, not a replacement for it.
5. Accept some mission detour in exchange for hindering the opponent.""",
    OPPORTUNIST: """PERSONALITY: OPPORTUNIST

Behavior priorities:
1. Pursue your own mission as the default priority.
2. When the opponent is nearby, look for cheap opportunities (e.g. a trap) to exploit.
3. Use last_seen opponent information when it is available and relevant.
4. Avoid expensive or prolonged pursuit of the opponent.
5. Return to your own mission once the opportunity has passed.""",
}


def validate_personality_profile(profile: str | None) -> str:
    """Normalizes and validates a profile name (spec section 13: invalid
    values fail clearly). `None` and `"default"` both normalize to
    `DEFAULT` - the explicit backward-compatible no-op profile (spec
    section 12)."""
    normalized = (profile or DEFAULT).strip().lower()
    if normalized not in VALID_PERSONALITY_PROFILES:
        raise InvalidPersonalityProfileError(
            f"Unknown personality_profile {profile!r}. Valid values: {sorted(VALID_PERSONALITY_PROFILES)}."
        )
    return normalized


def render_personality_block(profile: str | None) -> str | None:
    """The exact text appended to an agent's system prompt (spec section
    9) - `None` for `DEFAULT`, meaning the pre-M5F.1 prompt is used
    completely unmodified (spec section 12's backward-compatibility
    requirement), not an empty-but-present block."""
    normalized = validate_personality_profile(profile)
    if normalized == DEFAULT:
        return None
    return _PERSONALITY_BLOCKS[normalized]


# -- Bugfix mission time budget V2 (§24-25, §31A-C): live workshop only --------
# The personality blocks above are frozen evidence (M5F.1); these texts are a
# separate, live-only addendum (LLMControllerConfig.live_prompt_addendum /
# StrandsAgentAdapter's prompt, set by the workshop LiveGame). They state the
# mission priority and how each profile spends slack; they force no action.

MISSION_DIRECTIVE = (
    "MISSION FIRST: completing your mission and exiting within your remaining time is the primary objective. "
    "Personality guides how you use optional slack, not whether you sacrifice the mission for unnecessary actions. "
    "time_budget shows your time left, the known time still needed, the slack and the urgency; "
    "action_costs shows what each legal action costs in seconds of your mission time."
)

TIME_PRESSURE_NOTES: dict[str, str] = {
    DEFAULT: "Spend slack on whatever most helps the mission.",
    SABOTEUR: "Prefer sabotage when it creates meaningful opponent cost and the mission still has sufficient slack. "
              "Under critical time pressure, prioritize mission completion and exit.",
    CAUTIOUS: "Prefer inspection, disarming and risk reduction when justified. "
              "Do not spend scarce mission time removing irrelevant risks.",
    RACER: "Prefer direct mission progress and the exit. Optional actions require clear value.",
    OPPORTUNIST: "Exploit high-value opportunities when affordable. Abandon low-value opportunities as slack shrinks.",
}

DECISION_SUMMARY_CONTRACT = (
    "In `reason`, give a short decision summary: 1-2 sentences, at most 180 characters, stating the "
    "decision-relevant tradeoff (e.g. time left or slack, the action's cost, mission progress, risk). "
    "State a conclusion, not step-by-step thinking, and only use numbers that appear in time_budget/action_costs."
)


OWN_TRAP_RULE = (
    "Your own traps are dangerous to you too: searching an object that carries your own armed trap "
    "(trap_status OWN_TRAP_ARMED, listed in self.own_traps) sets it off. Search an object before you trap it; "
    "if you come back to one of your armed traps, disarm it before searching."
)

LIFECYCLE_RULE = (
    "Holding every required item is not the end: the mission finishes only when you leave through exit_door."
)


URGENCY_RULE = (
    "Let time_budget.urgency set how much risk you take - whatever your personality: "
    "RELAXED: inspect and set traps as your personality prefers. "
    "TIGHT (half the mission gone): inspect only an object you have a reason to suspect (the opponent was there "
    "or it is on the opponent's likely route); search the rest directly; trap only if it costs you little. "
    "CRITICAL (last quarter): no inspections, no traps, no detours - search directly and go to exit_door; "
    "with every required item, leave now."
)


ENCOUNTER_RULE = (
    "ENCOUNTERS: when encounter.alert is present the opponent is close and touching it starts a fight at once "
    "(50/50; the loser loses a life and is knocked out for a few seconds, then respawns elsewhere). Your walk "
    "has been stopped so that you decide now: approach_opponent to fight, avoid_opponent to get away, or carry "
    "on with your mission knowingly. Weigh your lives against the opponent's (encounter.opponent_lives), the "
    "time a knock-out costs, and your mission; say which you chose and why in your summary."
)
ENCOUNTER_NOTES: dict[str, str] = {
    DEFAULT: "Fight or run according to what helps the mission most.",
    SABOTEUR: "A fight is a form of sabotage: it can cost the opponent a life and time. Consider it when you have "
              "at least as many lives as the opponent.",
    CAUTIOUS: "Prefer to avoid a fight unless the opponent is on its last life and you are not.",
    RACER: "Avoid fights unless the opponent blocks the way to what you need right now.",
    OPPORTUNIST: "Fight when the odds favour you (more lives than the opponent, opponent on its last life); "
                 "otherwise get away.",
}


ITEMS_RULE = (
    "ONE COPY: there is only one of each required item. The first spy to search the object that holds it takes "
    "it; a later search finds it empty (mission.opponent_holds then lists what you know the opponent carries). "
    "You cannot leave without every required item, so an item the opponent holds can only be won back: the "
    "winner of a fight takes one required item from the loser (everything, if the loser is eliminated). A spy "
    "eliminated by a trap leaves its items on that object. Racing to the items, guarding them with traps, and "
    "choosing when to fight all matter."
)


SPLIT_RULE = (
    "SAFE CODE: the ambassador_safe code is split in two halves, revealed by searching two different objects in "
    "two different rooms (puzzle.safe_code_halves_known). With both halves you know the code and can open the safe."
)
RIDDLE_RULE = (
    "SAFE RIDDLE: the ambassador_safe code is written nowhere. Searching one object in the security room gives a "
    "clue (puzzle.clue) that refers to numbers engraved on other objects; you read an engraving when you inspect or "
    "search that object (puzzle.inscriptions_you_have_read). Some engravings are decoys - follow the clue exactly. "
    "Work the code out yourself, then, next to the safe, choose enter_safe_code with target_id ambassador_safe and "
    "code \"<digits>\". A wrong code costs time and the alarm shows the opponent where you are; a right one opens "
    "the safe."
)
# live payload diet: the exact costs, once here instead of on every request
# (puzzle.wrong_code_costs is no longer sent - src/llm/live_payload.py)
from src import constants as _constants  # noqa: E402
from src.puzzles import WRONG_CODE_SECONDS as _WRONG_CODE_SECONDS  # noqa: E402

RIDDLE_RULE = RIDDLE_RULE.replace(
    "A wrong code costs time and the alarm shows the opponent where you are",
    f"A wrong code keeps you busy {_WRONG_CODE_SECONDS:g} s and the alarm shows the opponent where you are for "
    f"{_constants.ALARM_REVEAL_DURATION:g} s")


def lives_rule(lives: int, fights_cost_a_life: bool = False) -> str:
    fights = " Losing a fight also costs one." if fights_cost_a_life else ""
    return (f"Each trap that goes off under you - an enemy one or your own - costs one of your {lives} lives "
            f"(self.lives).{fights} With none left you are out of the match and have failed.")


def scoring_rule() -> str:
    """V3 §64: the M4E score incentives, from the same constants compute_score uses."""
    from src import constants as c

    return (f"Scoring: all required items +{c.SCORE_COMPLETE_OWN_MISSION:g}; escaping before the deadline "
            f"+{c.SCORE_ESCAPE_BEFORE_DEADLINE:g}; escaping first +{c.SCORE_ESCAPE_FIRST:g}; the other spy also "
            f"escaping +{c.SCORE_OTHER_ALSO_ESCAPES:g}; +{c.SCORE_REMAINING_TIME_BONUS_PER_SECOND:g} per second left "
            f"when you escape; not escaping {c.SCORE_FAILURE_TO_ESCAPE:g}. Traps score nothing directly - sabotage "
            "only pays if it costs the opponent more time than it costs you.")


def trap_risk_rule() -> str:
    """Live balance rules, from the engine's own constants."""
    from src import constants as c

    def pct(table):
        return ", ".join(f"{t} {round(p * 100)}%" for t, p in table.items())
    return (f"INSPECT costs {c.LIVE_TIME_COST_INSPECT:g} s (a search costs {c.TIME_COST_SEARCH:g} s) and spots a hidden "
            f"trap with these chances: {pct(c.TRAP_DETECTION_PROBABILITY)}. Disarming a detected enemy trap succeeds "
            f"with these chances: {pct(c.DISARM_SUCCESS_PROBABILITY)}; when it fails, the trap goes off under you. "
            "Removing your own trap always works. Inspecting everything is safe but slow - weigh it against your time. "
            "PLACING A TRAP: you can trap an object once you have searched it - no inspection needed; place_trap is "
            "in legal_actions while you stand next to an object you searched and you still hold traps.")


def render_live_mission_block(profile: str | None, lives: int | None = None, scoring: bool = False,
                              trap_risk: bool = False, fights_cost_a_life: bool = False,
                              encounters: bool = False, single_copy_items: bool = False,
                              puzzle: str = "none") -> str:
    normalized = validate_personality_profile(profile)
    parts = [MISSION_DIRECTIVE, f"Under time pressure: {TIME_PRESSURE_NOTES[normalized]}", OWN_TRAP_RULE, LIFECYCLE_RULE]
    if trap_risk:
        parts.append(trap_risk_rule())
        parts.append(URGENCY_RULE)
    if single_copy_items:
        parts.append(ITEMS_RULE)
    if puzzle == "split":
        parts.append(SPLIT_RULE)
    elif puzzle == "riddle":
        parts.append(RIDDLE_RULE)
    if encounters:
        parts.append(ENCOUNTER_RULE)
        parts.append(f"In an encounter: {ENCOUNTER_NOTES[normalized]}")
    if lives:
        parts.append(lives_rule(lives, fights_cost_a_life))
    if scoring:
        parts.append(scoring_rule())
    parts.append(DECISION_SUMMARY_CONTRACT)
    return "\n".join(parts)
