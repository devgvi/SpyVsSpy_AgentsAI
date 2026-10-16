"""Unobtrusive agent telemetry for the live game.

`TelemetryView` turns what the running agents already expose (controller
status attributes, the in-memory telemetry events, the Game's response
barrier) into short text lines. It reads, it never writes: nothing here can
change a decision. `LiveRenderer` is the production `Renderer` with two
small corner panels and one status line drawn before the frame flips - the
game world stays the main picture.

Shown per spy: provider/model/runtime, personality, whether it is THINKING
(a model request in flight) or ACTING right now, current high-level action, last decision id, repair/fallback markers,
memory size, and (detail view) the model's own short `reason` field,
latency and the observation snapshot hash, trap inventory / opportunities /
placements and the live-async stale-answer counts. The `reason` is the
visible rationale the action contract asks for - not hidden chain-of-thought.
In the world view, short cues mark inspections, searches and trap
placements / triggers / disarms as they happen (spectator view only - no
agent's Observation changes).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pygame

from src import constants
from src.renderer import Renderer
from src.scoring import compute_score
from workshop.live.config import ACTOR_LABELS, ACTORS, LiveConfig

PANEL_WIDTH = 330
LINE_HEIGHT = 15
# V3 §35: every live overlay colour comes from these shared theme tokens.
# Dark panel + light text; each text token keeps >= 4.5:1 contrast against
# the panel even where the panel is blended over the brightest floor
# (tests/test_bugfix_live_match_flow.py checks it with the WCAG formula).
THEME: dict[str, tuple[int, int, int]] = {
    "panel_background": (12, 12, 14),
    "border": (110, 110, 120),
    "primary_text": (240, 240, 232),
    "secondary_text": (205, 205, 196),
    "muted_text": (178, 178, 170),
    "warning_text": (255, 170, 110),
    "critical_text": (255, 120, 110),
    "success_text": (140, 225, 150),
    "thinking_text": (245, 205, 95),
    "black_spy_text": (255, 185, 80),
    "white_spy_text": (160, 215, 255),
    "exterior": (38, 58, 46),
    "exterior_text": (200, 235, 205),
    "exit_sign": (70, 220, 110),
    "exit_sign_text": (10, 30, 15),
    "divider": (15, 15, 15),
}
THEME_TEXT_TOKENS = ("primary_text", "secondary_text", "muted_text", "warning_text", "critical_text",
                     "success_text", "thinking_text", "black_spy_text", "white_spy_text")
COLOR_DECIDING = THEME["thinking_text"]
COLOR_WARN = THEME["warning_text"]
COLOR_COMMIT = THEME["success_text"]
COLOR_DIM = THEME["muted_text"]
COLOR_TEXT = THEME["primary_text"]


def _luminance(rgb) -> float:
    def channel(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(v) for v in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg, bg) -> float:
    """WCAG 2.x contrast ratio between two sRGB colours."""
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def blend(top, alpha: int, under) -> tuple[int, int, int]:
    """`top` at `alpha` (0-255) over `under` - what a translucent panel looks like."""
    return tuple(round(t * alpha / 255 + u * (255 - alpha) / 255) for t, u in zip(top[:3], under[:3]))
# action_lifecycle terminal reasons, shortened for the panel
TERMINAL_REASONS = {"path_unreachable": "no path", "target_reached": "arrived", "interrupted": "interrupted", "resolved": "done"}
TRAP_SHORT = {"bomb": "bomb", "electric": "elec", "snare": "snare", "alarm": "alarm"}
# world-view cues (seconds on screen) for engine events
CUE_SECONDS = 1.6
CUE_EVENTS = {
    "OBJECT_INSPECTED": lambda e: "inspect",
    "OBJECT_SEARCHED": lambda e: "search",
    "TRAP_PLACED": lambda e: f"{e.get('trap_type')} trap set",
    "TRAP_TRIGGERED": lambda e: f"{e.get('trap_type')} trap!",
    "TRAP_DISARMED": lambda e: f"{e.get('trap_type')} disarmed",
    "TRAP_DETECTED": lambda e: "trap spotted",
    "TRAP_DISARM_FAILED": lambda e: "disarm failed!",
    "INSCRIPTION_READ": lambda e: f"engraving: {str(e.get('text', '')).split(': ')[-1]}",
    "SAFE_CODE_WRONG": lambda e: f"wrong code {e.get('entered')}!",
    "SAFE_CODE_SOLVED": lambda e: "safe cracked!",
    "KNOWLEDGE_LEARNED": lambda e: {"safe_clue": "riddle found", "safe_code_part_1": "code half 1",
                                    "safe_code_part_2": "code half 2"}.get(e.get("knowledge_id"), "code found"),
}
FIGHT_BANNER_SECONDS = 2.5
TIMER_BAR_HEIGHT = 24
HEADER_HEIGHT = 40
# Live feedback: every text panel lives in one sidebar on the right of the
# game view (the window is this much wider than the game view).
SIDEBAR_WIDTH = 400
SIDEBAR_CHARS = 52  # debug_font characters that fit on one sidebar line (about 7 px each)
# Live feedback ("too zoomed in"): camera zoom levels for the split / single
# views (Z cycles), and an overview that fits the whole embassy (O toggles).
ZOOM_LEVELS = (2.0, 1.5, 1.0)
DEFAULT_ZOOM = 1.5
VIEW_MODES = ("split", "single", "overview")
LIFECYCLE_LABELS = {"ACTIVE": "ACTIVE", "OBJECTIVES_COMPLETE": "OBJECTIVES COMPLETE", "ESCAPED": "ESCAPED",
                    "TIMED_OUT": "TIMED OUT", "ELIMINATED": "ELIMINATED", "MISSION_IMPOSSIBLE": "MISSION IMPOSSIBLE"}
CLASH_SECONDS = 0.7


def _visible_opponent_distance(state, actor_id: str) -> float | None:
    """What this spy can see (same rule as its Observation: same room)."""
    import math
    other_id = next((a for a in state.actors if a != actor_id), None)
    if other_id is None:
        return None
    me, other = state.actors[actor_id], state.actors[other_id]
    room = state.level.room_id_at(me.x, me.y)
    if room is None or room != state.level.room_id_at(other.x, other.y):
        return None
    return math.hypot(me.x - other.x, me.y - other.y)


def _mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _short(value: Any, n: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= n else text[: max(0, n - 1)] + "~"


def _action_label(action: Any) -> str:
    if action is None:
        return "-"
    kind = getattr(getattr(action, "type", None), "value", None) or str(getattr(action, "type", "?"))
    target = getattr(action, "target_id", None)
    trap = getattr(action, "trap_type", None)
    label = kind.upper()
    if target:
        label += f" -> {target}"
    if trap:
        label += f" ({trap})"
    return label


def _action_dict_label(action: dict | None) -> str:
    if not isinstance(action, dict):
        return "-"
    label = str(action.get("action") or action.get("type") or "?").upper()
    if action.get("target_id"):
        label += f" -> {action['target_id']}"
    return label


@dataclass
class ActorSnapshot:
    """What the overlay knows about one spy at one frame."""
    actor_id: str
    provider: str
    model: str
    runtime: str
    phase: str  # deciding / ready / held / acting / idle / done
    status: str
    action: str
    action_result: str | None
    decision_id: str | None
    decisions: int
    repairs: int
    fallbacks: int
    provider_failures: int
    last_repair: bool
    last_fallback: bool
    last_source: str | None
    memory_chars: int | None
    memory_decisions: int | None
    memory_failures: int | None
    latency_ms: float | None
    reason: str
    snapshot: str | None
    outcome: str | None
    # detail view only (bugfix navigation/exploration §30)
    navigation: dict | None = None  # the last MOVE's lifecycle `navigation` block
    exploration_room: str | None = None  # the room the decision was made in
    exploration: dict | None = None  # that room's exploration facts, as the model saw them
    # bugfix live flow (Parts A, C, D)
    personality: str = "default"
    timing: str | None = None
    live_async: dict | None = None  # LiveAsyncStats.as_dict() in live_async timing
    trap_inventory: dict | None = None
    trap_opportunities: int = 0  # decisions whose legal actions offered PLACE_TRAP
    trap_chosen: int = 0  # decisions whose model proposal was PLACE_TRAP
    traps_placed: int = 0
    traps_disarmed: int = 0
    # V2 Part D: encounters
    encounters_enabled: bool = False
    opponent_distance: float | None = None  # only while the opponent is visible to this spy
    fights_won: int = 0
    fights_lost: int = 0
    # mission time budget V2
    time_remaining: float | None = None  # this spy's own mission time left (live clock)
    time_limit: float | None = None
    time_used: float | None = None
    slack: float | None = None  # the last decision's slack, aged by the time used since
    urgency: str | None = None
    action_cost: float | None = None  # what the committed action costs (from the model's action_costs)
    summary: str | None = None  # the model's own short decision summary (`reason`) of the committed decision
    facts_caption: str | None = None  # neutral, system-built facts line
    summary_mismatch: bool = False  # RATIONALE_FACT_MISMATCH
    stale_redeciding: bool = False  # the last answer was discarded as stale; a new one is on its way
    lives: int | None = None  # live feedback: lives left (None = no lives rule)
    trap_hits: int = 0  # traps that went off under this spy
    # V3: explicit lifecycle + mission item progress
    lifecycle: str = "ACTIVE"
    items_have: int = 0
    items_total: int = 0
    # Stage 3: who plays this spy (llm / surrogate / scripted) and the surrogate's last decision
    controller: str = "llm"
    surrogate: dict | None = None


def _exploration_of(event: dict) -> tuple[str | None, dict | None]:
    exploration = (event.get("observation") or {}).get("exploration") or {}
    room = exploration.get("current_room")
    return room, (exploration.get("rooms") or {}).get(room) if room else None


@dataclass
class TelemetryView:
    """Polls a running Game; one instance per match."""
    config: LiveConfig
    events: list = field(default_factory=list)
    _seen: int = 0
    last_decision: dict = field(default_factory=dict)
    last_lifecycle: dict = field(default_factory=dict)
    last_move_navigation: dict = field(default_factory=dict)
    decision_counts: dict = field(default_factory=dict)
    trap_opportunities: dict = field(default_factory=dict)
    trap_chosen: dict = field(default_factory=dict)
    commits: int = 0  # M6G.3 barrier releases (evaluation timing)
    live_commits: int = 0  # live_async commits (not discarded), both spies
    last_live_commit: dict = field(default_factory=dict)
    last_commit: dict | None = None
    last_commit_at: float = 0.0
    _last_release_seen: Any = None
    _fight_seen: Any = None
    _fight_seen_at: float = 0.0

    def update(self, game) -> None:
        telemetry = getattr(game, "telemetry", None)
        if telemetry is not None:
            events = telemetry.events
            count = len(events)  # list.append is atomic; read up to a fixed length
            for event in events[self._seen:count]:
                kind = event.get("event_type")
                if kind == "llm_decision" and self._is_hybrid(event.get("actor_id")):
                    continue  # Stage 3 milestone 4: counted once, as that spy's hybrid_decision
                if kind == "llm_decision":
                    actor = event.get("actor_id")
                    self.last_decision[actor] = event
                    self.decision_counts[actor] = self.decision_counts.get(actor, 0) + 1
                    if any(a.get("action") == "place_trap" for a in event.get("legal_actions") or [] if isinstance(a, dict)):
                        self.trap_opportunities[actor] = self.trap_opportunities.get(actor, 0) + 1
                    if (event.get("initial_proposal") or {}).get("action") == "place_trap" or \
                            (event.get("executed_action") or {}).get("action") == "place_trap":
                        self.trap_chosen[actor] = self.trap_chosen.get(actor, 0) + 1
                elif kind in ("surrogate_decision", "hybrid_decision"):  # Stage 3: a surrogate / hybrid spy
                    actor = event.get("actor_id")
                    self.last_decision[actor] = event
                    self.decision_counts[actor] = self.decision_counts.get(actor, 0) + 1
                    if "place_trap" in (event.get("legal_actions") or []):
                        self.trap_opportunities[actor] = self.trap_opportunities.get(actor, 0) + 1
                    if event.get("sampled_action") == "place_trap":
                        self.trap_chosen[actor] = self.trap_chosen.get(actor, 0) + 1
                elif kind == "opponent_decision":  # the scripted bot (never in an LLM-only match)
                    actor = event.get("actor_id")
                    self.decision_counts[actor] = self.decision_counts.get(actor, 0) + 1
                elif kind == "live_async_commit":
                    self.last_live_commit[event.get("actor_id")] = event
                    if not event.get("discarded_due_to_stale"):
                        self.live_commits += 1
                elif kind == "action_lifecycle":
                    self.last_lifecycle[event.get("actor_id")] = event
                    if event.get("navigation"):
                        self.last_move_navigation[event.get("actor_id")] = event["navigation"]
            self._seen = count
        release = getattr(game, "last_barrier_release", None)
        if release is not None and release is not self._last_release_seen:
            self._last_release_seen = release
            self.commits += 1
            self.last_commit = release
            self.last_commit_at = time.monotonic()

    def _is_hybrid(self, actor_id: str | None) -> bool:
        spec = self.config.actors.get(actor_id) if actor_id else None
        return spec is not None and spec.controller == "hybrid"

    def actor(self, game, actor_id: str) -> ActorSnapshot:
        agent = game.black_llm_controller if actor_id == constants.BLACK_SPY_ID else game.bot_controller
        spec = self.config.actors[actor_id]
        event = self.last_decision.get(actor_id) or {}
        state = game.state
        outcome = None
        actor_lifecycle = state.lifecycle(actor_id) if hasattr(state, "lifecycle") else "ACTIVE"
        if state.mission_complete.get(actor_id):
            outcome = "ESCAPED"
        elif compute_score(state, actor_id).failed:
            outcome = "FAILED"
        thinking = bool(getattr(agent, "_thinking_in_progress", False))
        ready = bool(getattr(agent, "response_ready_uncommitted", False))
        held = actor_id in getattr(game, "_held_decisions", {})
        status = str(getattr(agent, "status", "?"))
        if outcome:
            phase = "done"
        elif thinking:
            phase = "thinking"
        elif held:
            phase = "held"
        elif ready:
            phase = "ready"
        elif status == "EXECUTING":
            phase = "acting"
        else:
            phase = "idle"
        # The action line is the decided action from the same telemetry event
        # (M7.1: the agent's per-tick last_action is a MOVE step without its
        # target, or WAIT after a failed MOVE), plus how that action ended.
        decided = event.get("executed_action") or event.get("validated_action")
        action = _action_dict_label(decided) if decided else "-"
        result = None
        lifecycle = self.last_lifecycle.get(actor_id)
        if decided and lifecycle and lifecycle.get("decision_id") == event.get("decision_id"):
            status_word = lifecycle.get("action_status")
            reason = TERMINAL_REASONS.get(lifecycle.get("action_terminal_reason"), lifecycle.get("action_terminal_reason"))
            result = f"{status_word}: {reason}" if status_word == "failed" else str(status_word)
        elif decided and str(decided.get("action")).lower() == "move":
            result = "moving"
        return ActorSnapshot(
            actor_id=actor_id, provider=spec.provider, model=spec.model,
            runtime=self.config.runtime if spec.is_llm else spec.controller,
            controller=spec.controller,
            surrogate=(getattr(agent, "last_decision", None) if spec.controller == "hybrid"
                       else event.get("surrogate_overlay") if event.get("event_type") == "surrogate_decision" else None),
            phase=phase, status=status, action=action, action_result=result,
            # Every decision field comes from the same telemetry event, so the
            # id, source and repair/fallback markers can never mix two decisions
            # (M7.1: the agent's own _last_decision_id is set a moment before
            # its event is emitted, which briefly paired a new id with old flags).
            decision_id=event.get("decision_id"),
            decisions=self.decision_counts.get(actor_id, 0),
            repairs=int(getattr(agent, "repair_count", 0) or 0),
            fallbacks=int(getattr(agent, "fallback_count", 0) or 0),
            provider_failures=int(getattr(agent, "provider_failure_count", 0) or 0),
            last_repair=bool(event.get("repair_attempted")), last_fallback=bool(event.get("fallback_used")),
            last_source=event.get("action_source"),
            memory_chars=event.get("memory_size_chars"), memory_decisions=event.get("recent_decisions_count"),
            memory_failures=event.get("recent_failures_count"),
            latency_ms=event.get("latency_ms"),
            reason=str(getattr(agent, "last_reason", "") or ""),
            snapshot=event.get("observation_hash"),
            outcome=outcome,
            navigation=self.last_move_navigation.get(actor_id),
            exploration_room=_exploration_of(event)[0],
            exploration=_exploration_of(event)[1],
            personality=spec.personality,
            timing=getattr(game, "llm_execution_mode", None),
            live_async=(agent.live_async_stats.as_dict()
                        if getattr(agent, "execution_mode", None) == "live_async" and getattr(agent, "live_async_stats", None)
                        else None),
            trap_inventory=dict(state.actors[actor_id].trap_inventory),
            trap_opportunities=self.trap_opportunities.get(actor_id, 0),
            trap_chosen=self.trap_chosen.get(actor_id, 0),
            traps_placed=sum(1 for e in state.event_log if e.get("type") == "TRAP_PLACED" and e.get("actor_id") == actor_id),
            traps_disarmed=sum(1 for e in state.event_log if e.get("type") == "TRAP_DISARMED" and e.get("actor_id") == actor_id),
            encounters_enabled=bool(getattr(state, "encounters_enabled", False)),
            opponent_distance=_visible_opponent_distance(state, actor_id),
            fights_won=sum(1 for f in getattr(state, "fights", []) if f["winner_id"] == actor_id),
            fights_lost=sum(1 for f in getattr(state, "fights", []) if f["loser_id"] == actor_id),
            **self._time_fields(game, actor_id, agent),
            lives=state.lives.get(actor_id) if getattr(state, "lives_enabled", False) else None,
            trap_hits=getattr(state, "trap_hits", {}).get(actor_id, 0),
            lifecycle=actor_lifecycle,
            items_have=sum(1 for i in state.items.required if state.actors[actor_id].inventory.has(i)),
            items_total=len(state.items.required),
        )

    def _time_fields(self, game, actor_id: str, agent) -> dict:
        """Mission time budget V2 (§29-31H): authoritative time now + the
        committed decision's cost, summary and facts (from its commit event)."""
        from src.mission_time import urgency_for

        state = game.state
        remaining = state.time_remaining_for(actor_id) if hasattr(state, "time_remaining_for") else state.mission_time_remaining
        if state.mission_complete.get(actor_id) and actor_id in getattr(state, "escape_snapshot", {}):
            remaining = state.escape_snapshot[actor_id]  # its time stopped when it escaped
        out = {"time_remaining": remaining, "time_limit": state.mission_time_total,
               "time_used": state.mission_time_total - remaining}
        commit = self.last_live_commit.get(actor_id)
        if not commit or commit.get("mission_time_remaining") is None:
            return out
        aged = (commit["mission_time_remaining"] - remaining)
        slack = None if commit.get("slack_s") is None else commit["slack_s"] - aged
        out.update({
            "slack": slack,
            "urgency": urgency_for(remaining, commit.get("known_required_time"), state.mission_time_total, share_floor=True),
            "action_cost": commit.get("chosen_action_time_cost"),
            "summary": commit.get("decision_summary"),
            "facts_caption": commit.get("facts_caption"),
            "summary_mismatch": bool(commit.get("rationale_fact_mismatch")),
            "stale_redeciding": bool(commit.get("discarded_due_to_stale")),
        })
        return out

    @staticmethod
    def time_line(snap: ActorSnapshot) -> tuple[str, tuple[int, int, int]] | None:
        if snap.time_remaining is None:
            return None
        def mmss(seconds: float) -> str:
            seconds = max(0, int(round(seconds)))
            return f"{seconds // 60:02d}:{seconds % 60:02d}"
        slack = "-" if snap.slack is None else ("+" if snap.slack >= 0 else "-") + mmss(abs(snap.slack))
        urgency = snap.urgency or "-"
        color = {"CRITICAL": COLOR_WARN, "TIGHT": COLOR_DECIDING}.get(urgency, COLOR_COMMIT)
        return f"TIME {mmss(snap.time_remaining)} | SLACK {slack} | {urgency}", color

    def actor_lines(self, snap: ActorSnapshot, detail: bool) -> list[tuple[str, tuple[int, int, int]]]:
        text = COLOR_TEXT
        phase_text = {
            "thinking": ("THINKING...", COLOR_DECIDING),
            "ready": ("answer ready", COLOR_DECIDING),
            "held": ("ready - barrier holds until the other spy answers", COLOR_DECIDING),
            "acting": ("ACTING", text),
            "idle": ("idle", COLOR_DIM),
            "done": (LIFECYCLE_LABELS.get(snap.lifecycle, "FINISHED"), COLOR_COMMIT if snap.outcome == "ESCAPED" else COLOR_WARN),
        }[snap.phase]
        flags = []
        if snap.last_repair:
            flags.append("REPAIR")
        if snap.last_fallback:
            flags.append("FALLBACK")
        phase_word = phase_text[0]
        if snap.phase == "done" and snap.outcome == "ESCAPED" and snap.time_remaining is not None:
            phase_word += f" - {_mmss(snap.time_remaining)} left"
        first = (_short(f"{ACTOR_LABELS[snap.actor_id].upper()} [{snap.runtime}]  {phase_word}", 46), phase_text[1])
        if snap.controller == "surrogate":  # Stage 3: its own compact block; the LLM lines below are untouched
            return [first] + self.surrogate_lines(snap, detail)
        if snap.controller == "hybrid":  # Stage 3 milestone 4
            return [first] + self.hybrid_lines(snap, detail)
        if not detail:
            # compact view (live feedback): what it is doing, and what it decided - nothing else
            return [first] + self.summary_lines(snap, detail, width=SIDEBAR_CHARS - 6)  # "AGENT " prefix
        lines = [
            first,
            (_short(f"{snap.provider} / {snap.model}", 46), COLOR_DIM),
            (_short(f"persona  {snap.personality}", 46), COLOR_DIM),
        ]
        time_line = self.time_line(snap)
        if time_line is not None:
            lines.append(time_line)
        if snap.lives is not None:
            lines.append((f"LIVES {snap.lives}/{constants.SPY_LIVES}   TRAPS HIT {snap.trap_hits}   FIGHTS LOST {snap.fights_lost}",
                          COLOR_WARN if snap.lives <= 1 else text))
        if snap.items_total:
            lines.append((f"ITEMS {snap.items_have}/{snap.items_total}", text))
        if snap.lifecycle == "OBJECTIVES_COMPLETE" and snap.phase != "done":
            lines.append(("OBJECTIVES COMPLETE - REACH THE EXIT", COLOR_DECIDING))
        if snap.phase == "done":
            outcome = {"ESCAPED": "MISSION SUCCESS - ESCAPED", "ELIMINATED": "ELIMINATED - NO LIVES LEFT",
                       "TIMED_OUT": "TIMED OUT - MISSION FAILED",
                       "MISSION_IMPOSSIBLE": "MISSION IMPOSSIBLE - THE OPPONENT LEFT WITH THE ITEMS"}.get(
                snap.lifecycle, "MISSION FAILED")
            lines.append((outcome, COLOR_COMMIT if snap.outcome == "ESCAPED" else COLOR_WARN))
            if snap.time_used is not None:
                lines.append((f"time used {snap.time_used:.0f} s, left {max(0.0, snap.time_remaining):.0f} s", COLOR_DIM))
        cost = "" if snap.action_cost is None else f"  (+{snap.action_cost:g} s)"
        lines += [
            (_short(f"action   {snap.action}", 46 - len(cost)) + cost, text),
            (_short(f"result   {snap.action_result or '-'}", 46),
             COLOR_WARN if (snap.action_result or "").startswith("failed") else COLOR_DIM),
            (_short(f"decision {(snap.decision_id or '-')[:8]}  #{snap.decisions}  src={snap.last_source or '-'}", 46)
             + ("  " + " ".join(flags) if flags else ""), COLOR_WARN if flags else text),
        ]
        lines.extend(self.summary_lines(snap, detail))
        mem = "-" if snap.memory_chars is None else f"{snap.memory_chars / 1000:.1f}k chars"
        mem_counts = "" if snap.memory_decisions is None else f"  recent: {snap.memory_decisions} dec, {snap.memory_failures or 0} fail"
        lines.append((_short(f"memory   {mem}{mem_counts}", 46), text))
        totals = f"totals   repair {snap.repairs}  fallback {snap.fallbacks}"
        if snap.provider_failures:
            totals += f"  provider errors {snap.provider_failures}"
        lines.append((totals, COLOR_WARN if snap.provider_failures else COLOR_DIM))
        if detail:
            latency = "-" if snap.latency_ms is None else f"{snap.latency_ms:.0f} ms"
            lines.append((f"latency  {latency}   snapshot {(snap.snapshot or '-')[:10]}", COLOR_DIM))
            inv = snap.trap_inventory or {}
            inv_text = " ".join(f"{TRAP_SHORT.get(t, t)}{n}" for t, n in sorted(inv.items())) or "-"
            lines.append((_short(f"traps    {inv_text}", 46), COLOR_DIM))
            lines.append((_short(f"         offered {snap.trap_opportunities}  chosen {snap.trap_chosen}"
                                 f"  placed {snap.traps_placed}  disarmed {snap.traps_disarmed}", 46), COLOR_DIM))
            if snap.encounters_enabled:
                seen = ("not visible" if snap.opponent_distance is None
                        else f"visible {snap.opponent_distance:.0f} px (fight at {constants.ENCOUNTER_DISTANCE:.0f})")
                lines.append((_short(f"opponent {seen}", 46), COLOR_DIM))
                lines.append((_short(f"fights   won {snap.fights_won}  lost {snap.fights_lost}", 46), COLOR_DIM))
            if snap.live_async is not None:
                la = snap.live_async
                lines.append((_short(f"async    fresh {la['fresh']}  stale-ok {la['stale_but_still_legal']}"
                                     f"  discarded {la['discarded_due_to_stale']}", 46),
                              COLOR_WARN if la["discarded_due_to_stale"] else COLOR_DIM))
            reason = snap.reason or "-"
            for i in range(0, min(len(reason), 3 * 44), 44):
                lines.append(((" reason  " if i == 0 else "         ") + reason[i:i + 44], COLOR_DIM))
            lines.extend(self.debug_lines(snap))
        return lines

    def hybrid_lines(self, snap: ActorSnapshot, detail: bool) -> list[tuple[str, tuple[int, int, int]]]:
        """HYBRID - model / personality, base, router, coverage, B1-B2 TVD, decision (B2 or LLM) and reason."""
        s = snap.surrogate or {}
        lines = [(_short(f"HYBRID - {s.get('model_display') or snap.model} / {s.get('personality') or snap.personality}", 46),
                  COLOR_DECIDING)]
        lines += [(_short(f"base: B2 ({s.get('emulator') or 'logistic-v1'})", 46), COLOR_DIM),
                  (_short(f"router: {s.get('router') or self.config.hybrid_router}", 46), COLOR_DIM)]
        if not s:
            return lines + [("no decision yet", COLOR_DIM)]
        if s.get("coverage"):
            lines.append((f"coverage: {s['coverage']}", COLOR_WARN if s["coverage"] in ("LOW", "OOD") else COLOR_TEXT))
        if s.get("b1_b2_tvd") is not None:
            lines.append((f"B1<->B2 TVD: {s['b1_b2_tvd']:.2f}", COLOR_TEXT))
        decision = s.get("decision") or "B2"
        word = {"LLM ESCALATION": "LLM ESCALATION", "LLM": "LLM", "B2": "B2"}.get(decision, decision)
        lines.append((_short(f"decision: {word}", 46), COLOR_WARN if decision.startswith("LLM") else COLOR_TEXT))
        reasons = s.get("router_reasons") or []
        if reasons:
            short = {"repetition_loop": "repetition", "strong_b1_b2_disagreement": "disagreement",
                     "coverage_ood": "OOD", "low_ood_critical": "critical", "low_high_confidence": "LOW+confident"}
            lines.append((_short("reason: " + " + ".join(short.get(r, r) for r in reasons), 46), COLOR_DIM))
        if s.get("budget_exhausted"):
            lines.append(("LLM budget exhausted - B2", COLOR_WARN))
        if detail:
            lines.append((_short(f"action   {snap.action}", 46), COLOR_TEXT))
            if s.get("confidence") is not None:
                lines.append((f"B2 confidence {s.get('confidence', 0):.2f}", COLOR_DIM))
            for action, p in s.get("top") or []:
                lines.append((_short(f"  {str(action).upper():<18} {p:.2f}", 46), COLOR_DIM))
            for reason in s.get("coverage_reasons") or []:
                lines.append((_short(f" cov     {reason}", 46), COLOR_DIM))
        return lines

    def surrogate_lines(self, snap: ActorSnapshot, detail: bool) -> list[tuple[str, tuple[int, int, int]]]:
        """SURROGATE / policy / emulator / confidence / choice; the detail view adds the
        probabilities and the table bucket, backoff level and training samples."""
        s = snap.surrogate
        lines = [("SURROGATE", COLOR_DECIDING)]
        if not s:
            return lines + [(_short(f"policy: {snap.model}", 46), COLOR_TEXT), ("no decision yet", COLOR_DIM)]
        lines += [(_short(f"policy: {s.get('model_display')} / {s.get('personality')}", 46), COLOR_TEXT),
                  (_short(f"emulator: {s.get('emulator')}", 46), COLOR_DIM),
                  (_short(f"choice: {s.get('choice')}", 46), COLOR_TEXT),
                  (f"confidence: {s.get('confidence', 0):.2f}", COLOR_TEXT)]
        if s.get("coverage"):
            lines.append((f"coverage: {s['coverage']}", COLOR_WARN if s["coverage"] in ("LOW", "OOD") else COLOR_TEXT))
        if detail:
            lines.append((_short(f"action   {snap.action}", 46), COLOR_TEXT))
            lines.append((_short(f"result   {snap.action_result or '-'}", 46), COLOR_DIM))
            for action, p in s.get("top") or []:
                lines.append((_short(f"  {str(action).upper():<18} {p:.2f}", 46), COLOR_DIM))
            for reason in s.get("coverage_reasons") or []:
                lines.append((_short(f" cov     {reason}", 46), COLOR_DIM))
            explanation = s.get("explanation") or {}
            if "path" in explanation:  # B3: leaf, path, leaf samples
                lines.append((f"leaf {explanation.get('leaf')}  leaf_samples {explanation.get('leaf_samples')}", COLOR_DIM))
                for step in (explanation.get("path") or [])[:6]:
                    lines.append((_short(f"  {step}", 46), COLOR_DIM))
            elif "top_features" in explanation:  # B2: the columns that moved the choice most
                for column, weight in (explanation.get("top_features") or [])[:4]:
                    lines.append((_short(f"  {column} {weight:+.2f}", 46), COLOR_DIM))
            else:  # B0 / B1: bucket, backoff level, bucket samples
                bucket = str(s.get("bucket") or "*")
                for i in range(0, min(len(bucket), 3 * 44), 44):
                    lines.append(((" bucket  " if i == 0 else "         ") + bucket[i:i + 44], COLOR_DIM))
                lines.append((f"backoff_level {s.get('backoff_level')}  training_samples {s.get('training_samples')}",
                              COLOR_DIM))
            time_line = self.time_line(snap)
            if time_line is not None:
                lines.append(time_line)
            if snap.lives is not None:
                lines.append((f"LIVES {snap.lives}/{constants.SPY_LIVES}   TRAPS HIT {snap.trap_hits}", COLOR_DIM))
        return lines

    @staticmethod
    def summary_lines(snap: ActorSnapshot, detail: bool, width: int = 44) -> list[tuple[str, tuple[int, int, int]]]:
        """The committed decision's own summary (quoted, the agent's words) and,
        separately, the authoritative facts - never merged (§31D-H). The
        compact view shows the whole summary (at most 180 characters)."""
        if snap.stale_redeciding:  # the discarded decision's summary is not the current intent
            return [("STALE - REDECIDING", COLOR_WARN)]
        lines = []
        if snap.summary:
            quote = f'"{snap.summary}"'
            limit = len(quote)
            chunks = [quote[i:i + width] for i in range(0, min(len(quote), limit), width)]
            if not detail and len(quote) > limit:
                chunks[-1] = chunks[-1][:-1] + "~"
            for i, chunk in enumerate(chunks):
                lines.append((("AGENT " if i == 0 else "      ") + chunk, COLOR_DIM))
            if snap.summary_mismatch:
                lines.append(("       (summary numbers differ from the facts)", COLOR_WARN))
        if snap.facts_caption and (detail or not snap.summary):
            lines.append((_short(f"FACTS {snap.facts_caption}", 46), COLOR_DIM))
        return lines

    @staticmethod
    def debug_lines(snap: ActorSnapshot) -> list[tuple[str, tuple[int, int, int]]]:
        """Engine vs agent choice (bugfix §30): did the last MOVE get the spy
        where the intended interaction is legal, and what exploration facts did
        the model have when it decided."""
        lines = []
        nav = snap.navigation
        if nav:
            legal = nav.get("interaction_legal_after_move")
            dist = nav.get("distance_to_target_after_move")
            where = "reached" if legal else (nav.get("failure_detail") or "not in range").replace("_", " ")
            lines.append((_short(f" move    {nav.get('move_target_id')}: {where}", 46), COLOR_DIM if legal else COLOR_WARN))
            lines.append((_short(f"         dist {'-' if dist is None else f'{dist:.0f}'}/{nav.get('interaction_range')} px"
                                 f"  interact {'legal' if legal else 'not legal'}", 46), COLOR_DIM))
        room = snap.exploration
        if room is not None:
            lines.append((_short(f" explore {snap.exploration_room} {room.get('status', '?')}", 46),
                          COLOR_DIM if room.get("status") == "COMPLETE" else COLOR_DECIDING))
            for label, key in (("unsearched", "unsearched"), ("uninspect", "uninspected"), ("exits left", "untraversed_exits")):
                ids = room.get(key) or []
                if ids:
                    lines.append((_short(f"   {label:<10} {', '.join(ids)}", 46), COLOR_DIM))
        return lines

    def _recent_fight(self, game) -> dict | None:
        """The last fight, for FIGHT_BANNER_SECONDS of wall time after it was first drawn."""
        fights = getattr(getattr(game, "state", None), "fights", None) or []
        if not fights:
            return None
        last = fights[-1]
        if last is not self._fight_seen:
            self._fight_seen, self._fight_seen_at = last, time.monotonic()
        return last if time.monotonic() - self._fight_seen_at < FIGHT_BANNER_SECONDS else None

    def status_line(self, game, *, paused: bool, speed: float, match_over: bool) -> tuple[str, tuple[int, int, int]]:
        parts = []
        if match_over:
            parts.append("MATCH OVER - R restart, Esc quit")
        elif paused:
            parts.append("PAUSED - Space resume")
        if speed != 1.0:
            parts.append(f"speed x{speed:g} (game time per wall second - not for comparisons)")
        policy = getattr(game, "commit_policy", "?")
        if self.last_commit:
            order = " then ".join(ACTOR_LABELS.get(a, a).split()[0] for a in self.last_commit.get("commit_order", []))
            parts.append(f"commit #{self.last_commit.get('decision_boundary_id')}: {order}")
        fight = self._recent_fight(game)
        if fight is not None:
            winner = ACTOR_LABELS.get(fight["winner_id"], fight["winner_id"]).split()[0]
            parts.insert(0, f"FIGHT Black vs White - Winner: {winner}")
        if getattr(game, "llm_execution_mode", None) == "live_async":
            parts.append("timing live_async (game runs while models think)")
        else:
            parts.append(f"policy {policy}")
        renderer = getattr(game, "renderer", None)
        if renderer is not None and hasattr(renderer, "view_mode"):
            zoom = "" if renderer.overview else f" x{renderer.zoom_level:g}"
            parts.append(f"view {renderer.view_mode}{zoom}")
        parts.append("Space pause  R restart  +/- speed  T telemetry  V detail  P split  O overview  Z zoom  "
                     "F3 debug  Esc quit")
        fresh = self.last_commit is not None and time.monotonic() - self.last_commit_at < 0.8
        return ("   |   ".join(parts), COLOR_COMMIT if fresh else COLOR_DIM)


# readable text colours for each spy on the dark panels / cue backings
# (live feedback: Black's own colour, (20, 20, 20), is black-on-dark)
SPY_TEXT_COLORS = {constants.BLACK_SPY_ID: THEME["black_spy_text"], constants.WHITE_SPY_ID: THEME["white_spy_text"]}
PANEL_ALPHA = 225  # was 175: the floor showed through and text read as grey-on-grey
EXIT_SIGN = THEME["exit_sign"]


class LiveRenderer(Renderer):
    """The production renderer plus the telemetry panels, drawn before flip.

    Live feedback: by default the window is split in two, each half
    following one spy (P toggles a single view following Black). The
    human-player HUD (Black's trap keys, inventory, prompts) is replaced
    by a short live header per spy. An escaped spy has left the building
    and is no longer drawn; the exit is marked."""

    def __init__(self, screen: pygame.Surface) -> None:
        super().__init__(screen)
        self.show_telemetry = True
        self.show_detail = False
        self.view: TelemetryView | None = None
        self.game = None
        self.paused = False
        self.speed = 1.0
        self.match_over = False
        self.split = True
        self.overview = False
        self.zoom_level = DEFAULT_ZOOM
        self._surfaces: dict[tuple[int, int], pygame.Surface] = {}
        self._half_world = pygame.Surface((round(self.view_width / 2), round(self.view_height)))
        self._cue_seen = 0
        self._clashes: list[tuple[float, tuple[float, float]]] = []
        self._cues: list[tuple[float, str, tuple[float, float], tuple[int, int, int]]] = []
        self._escaped_at: dict[str, float] = {}
        self.match_result: dict | None = None  # set by MatchRunner when every spy is terminal
        self._exterior: tuple | None = None  # (key, surface) - the non-room area, drawn as outside
        self.pane_rects: dict[str, pygame.Rect] = {}  # last frame's pane per spy (tests / layout checks)
        self.panel_rects: dict[str, pygame.Rect] = {}  # last frame's telemetry panel per spy
        self.header_rects: dict[str, pygame.Rect] = {}

    # -- frame -----------------------------------------------------------------
    def render(self, state, dt, selected_trap_type=None, actions=None, bot_controller=None) -> None:
        self.last_actions = actions  # read by the --debug recorder only
        self._collect_cues(state)
        self._update_effects(state, dt)
        game = self.game_rect()
        self.pane_rects = self.panes(game.width, game.height)
        self.screen.fill(THEME["panel_background"])
        if self.overview:
            self._render_overview(state, dt, actions, game)
        elif self.split:
            half = game.width // 2
            for i, actor_id in enumerate((constants.BLACK_SPY_ID, constants.WHITE_SPY_ID)):
                pane = self.pane_rects[actor_id]
                surface = self._world_surface(pane.size, self.zoom_level)
                camera = self._camera_for(state, actor_id, surface.get_width(), surface.get_height())
                # animations advance once per frame: only in the first view
                self._render_view(state, dt if i == 0 else 0.0, actions, surface, camera, pane)
            pygame.draw.line(self.screen, THEME["divider"], (half, 0), (half, game.height), 4)
        else:
            surface = self._world_surface(game.size, self.zoom_level)
            camera = self._camera_for(state, constants.BLACK_SPY_ID, surface.get_width(), surface.get_height())
            self._render_view(state, dt, actions, surface, camera, game)
        self.draw_overlay()
        if self.match_result is not None:
            self._draw_final_ranking(self.match_result)
        pygame.display.flip()

    def game_rect(self) -> pygame.Rect:
        width, height = self.screen.get_size()
        return pygame.Rect(0, 0, width - self.sidebar_rect().width, height)

    def sidebar_rect(self) -> pygame.Rect:
        width, height = self.screen.get_size()
        side = min(SIDEBAR_WIDTH, width // 3)
        return pygame.Rect(width - side, 0, side, height)

    def panes(self, width: int, height: int) -> dict[str, pygame.Rect]:
        """Game-view area per spy (the window minus the sidebar): the two
        halves in split view; in single view both name the one camera."""
        half = width // 2
        return {constants.BLACK_SPY_ID: pygame.Rect(0, 0, half, height),
                constants.WHITE_SPY_ID: pygame.Rect(half, 0, width - half, height)}

    @property
    def view_mode(self) -> str:
        return "overview" if self.overview else ("split" if self.split else "single")

    def cycle_zoom(self) -> float:
        i = ZOOM_LEVELS.index(self.zoom_level) if self.zoom_level in ZOOM_LEVELS else 0
        self.zoom_level = ZOOM_LEVELS[(i + 1) % len(ZOOM_LEVELS)]
        return self.zoom_level

    def _world_surface(self, size: tuple[int, int], zoom: float) -> pygame.Surface:
        """The world area a pane of `size` screen px shows at `zoom` (cached)."""
        key = (max(1, round(size[0] / zoom)), max(1, round(size[1] / zoom)))
        if key not in self._surfaces:
            self._surfaces[key] = pygame.Surface(key)
        return self._surfaces[key]

    def overview_rect(self, state) -> tuple[pygame.Rect, float]:
        """Where the whole map lands in the game view, and its scale."""
        game = self.game_rect()
        map_w, map_h = state.level.size
        scale = min(game.width / map_w, game.height / map_h)
        rect = pygame.Rect(0, 0, round(map_w * scale), round(map_h * scale))
        rect.center = game.center
        return rect, scale

    def _render_overview(self, state, dt, actions, game: pygame.Rect) -> None:
        """The whole embassy in the game view (O). Below 1x the map is scaled
        smoothly - nearest neighbour would drop whole rows of pixel art - and
        the text cues are drawn afterwards at screen size so they stay legible."""
        dest, scale = self.overview_rect(state)
        surface = self._world_surface(state.level.size, 1.0)
        self._render_view(state, dt, actions, surface, (0, 0), dest, cues=False)
        self._draw_cues(target=self.screen, to_screen=lambda x, y: (round(dest.x + x * scale), round(dest.y + y * scale)))
        for actor_id in (constants.BLACK_SPY_ID, constants.WHITE_SPY_ID):  # who is who, at a glance
            actor = state.actors[actor_id]
            if getattr(state, "escaped_leave_world", False) and state.mission_complete.get(actor_id) \
                    and actor_id not in getattr(state, "exit_walking", {}):
                continue
            sx, sy = round(dest.x + actor.x * scale), round(dest.y + (actor.y - 60) * scale)
            name = ACTOR_LABELS[actor_id].split()[0].upper()
            surf = self.debug_font.render(name, True, SPY_TEXT_COLORS[actor_id])
            rect = surf.get_rect(midbottom=(sx, sy))
            self._draw_panel_backing(rect.inflate(6, 2))
            self.screen.blit(surf, rect.topleft)

    def _render_view(self, state, dt, actions, surface, camera, dest: pygame.Rect, cues: bool = True) -> None:
        saved = self.world
        self.world, self.camera = surface, camera
        try:
            self._draw_floor(state)
            self._draw_exterior(state)
            self._draw_decorations(state)
            self._draw_y_sorted(state, dt, actions)
            self._draw_exit_marker(state)
            self._draw_effects(state)
            if cues:
                self._draw_cues()
            if self.debug_enabled:
                self._draw_debug(state)
            whole = surface.get_size()
            # integer zoom: crisp nearest neighbour; any other scale: smooth
            exact = dest.width % whole[0] == 0 and dest.width // whole[0] == dest.height // max(1, whole[1])
            scaled = (pygame.transform.scale(surface, dest.size) if exact
                      else pygame.transform.smoothscale(surface, dest.size))
            self.screen.blit(scaled, dest.topleft)
        finally:
            self.world = saved

    def _camera_for(self, state, actor_id: str, view_w: float, view_h: float) -> tuple[int, int]:
        focus = state.actors[actor_id]
        map_w, map_h = state.level.size
        cam_x = min(max(0, focus.x - view_w / 2), max(0, map_w - view_w))
        cam_y = min(max(0, focus.y - view_h / 2), max(0, map_h - view_h))
        return int(cam_x), int(cam_y)

    def _draw_actor(self, state, actor_id: str, dt: float, actions) -> None:
        if getattr(state, "escaped_leave_world", False) and state.mission_complete.get(actor_id) \
                and actor_id not in getattr(state, "exit_walking", {}):
            if not self.animators[actor_id].active("celebrate"):
                return  # it walked out of the building (drawn while crossing, then a short celebration)
        super()._draw_actor(state, actor_id, dt, actions)
        actor = state.actors[actor_id]
        ko_left = state.ko_seconds_left(actor_id) if hasattr(state, "ko_seconds_left") else None
        if ko_left is not None:
            # KO countdown over the spy that lost the fight (spectator view)
            import math as _math
            text = f"KO - RESPAWN {max(1, _math.ceil(ko_left))}"
            surf = self.debug_font.render(text, True, THEME["critical_text"])
            sx, sy = self._screen_pos(actor.x, actor.y - 70)
            rect = surf.get_rect(midbottom=(sx, sy))
            backing = pygame.Surface((rect.width + 6, rect.height + 2), pygame.SRCALPHA)
            backing.fill((*THEME["panel_background"], PANEL_ALPHA))
            self.world.blit(backing, (rect.x - 3, rect.y - 1))
            self.world.blit(surf, rect.topleft)
        if self.animators[actor_id].active("respawn"):
            # respawn effect: an expanding ring at the feet
            from src.animation import RESPAWN_SECONDS
            t = 1.0 - self.animators[actor_id]._timers["respawn"] / RESPAWN_SECONDS
            sx, sy = self._screen_pos(actor.x, actor.y - 30)
            pygame.draw.circle(self.world, THEME["thinking_text"], (sx, sy), int(10 + 30 * t), width=2)

    def _draw_exterior(self, state) -> None:
        """V3 Part D: everything that is neither a room nor a wall is outside
        the building - drawn as a distinct dark-green ground, with an OUTSIDE
        label next to the exit, so walking out of the exit reads as leaving."""
        level = state.level
        key = (id(level), len(level.walls), tuple(level.size))
        if self._exterior is None or self._exterior[0] != key:
            surface = pygame.Surface(level.size, pygame.SRCALPHA)
            surface.fill((*THEME["exterior"], 255))
            for room in level.rooms:
                surface.fill((0, 0, 0, 0), room.rect)
            for wall in level.walls:
                surface.fill((0, 0, 0, 0), wall)
            for obj in level.objects:
                if obj.type == "door":
                    surface.fill((0, 0, 0, 0), obj.collision_rect)
            self._exterior = (key, surface)
        cam_x, cam_y = self.camera
        self.world.blit(self._exterior[1], (-cam_x, -cam_y))
        from src.exit_geometry import exterior_point
        point = exterior_point(level)
        if point is not None:
            surf = self.debug_font.render("OUTSIDE", True, THEME["exterior_text"])
            sx, sy = self._screen_pos(point[0], point[1] - 60)
            self.world.blit(surf, surf.get_rect(center=(sx, sy)))

    def _draw_exit_marker(self, state) -> None:
        """Live feedback: the exit read as part of the wall. A green frame and
        an EXIT sign make it findable at a glance."""
        exit_obj = state.level.get_object("exit_door")
        if exit_obj is None:
            return
        rect = self._screen_rect(exit_obj.collision_rect)
        pygame.draw.rect(self.world, EXIT_SIGN, rect, width=3)
        surf = self.debug_font.render("EXIT", True, THEME["exit_sign_text"])
        sign = pygame.Rect(0, 0, surf.get_width() + 8, surf.get_height() + 4)
        sign.midbottom = (rect.centerx, rect.top - 2)
        pygame.draw.rect(self.world, EXIT_SIGN, sign)
        self.world.blit(surf, (sign.x + 4, sign.y + 2))

    # -- cues ------------------------------------------------------------------
    def _collect_cues(self, state) -> None:
        """Short labels for engine events, collected once per frame (drawn in
        each view). Presentation only: reads the event log, changes nothing."""
        now = time.monotonic()
        log = state.event_log
        if len(log) < self._cue_seen:  # a restarted match
            self._cue_seen = 0
            self._cues, self._clashes, self._escaped_at = [], [], {}
        for event in log[self._cue_seen:]:
            kind = event.get("type")
            actor_id = event.get("actor_id")
            color = SPY_TEXT_COLORS.get(actor_id, COLOR_TEXT)
            if kind == "SPY_FIGHT_RESOLVED":
                b, w = state.actors[constants.BLACK_SPY_ID], state.actors[constants.WHITE_SPY_ID]
                mid = ((b.x + w.x) / 2, (b.y + w.y) / 2)
                winner = ACTOR_LABELS.get(event["winner_id"], event["winner_id"]).split()[0]
                self._cues.append((now + FIGHT_BANNER_SECONDS, f"FIGHT! {winner} wins", (mid[0], mid[1] - 80),
                                   SPY_TEXT_COLORS.get(event["winner_id"], COLOR_TEXT)))
                self._clashes.append((now, mid))
                continue
            if kind == "MISSION_COMPLETED":
                exit_obj = state.level.get_object("exit_door")
                if exit_obj is not None:
                    name = ACTOR_LABELS.get(actor_id, actor_id).split()[0]
                    self._cues.append((now + 3.0, f"{name} ESCAPED", exit_obj.collision_rect.midtop, color))
                continue
            if kind == "ITEM_STOLEN":
                a = state.actors.get(actor_id)
                if a is not None:
                    name = ACTOR_LABELS.get(actor_id, actor_id).split()[0]
                    item = state.items.items[event["item_id"]].name if event.get("item_id") in state.items.items else event.get("item_id")
                    self._cues.append((now + 3.0, f"{name} TOOK THE {str(item).upper()}!", (a.x, a.y - 110), color))
                continue
            if kind == "ITEM_ALREADY_TAKEN":
                obj = state.level.get_object(event.get("object_id"))
                if obj is not None:
                    self._cues.append((now + CUE_SECONDS, "empty - already taken", obj.collision_rect.midtop, color))
                continue
            if kind in ("ACTOR_KNOCKED_OUT", "ACTOR_RESPAWNED"):
                a = state.actors.get(actor_id)
                if a is not None:
                    name = ACTOR_LABELS.get(actor_id, actor_id).split()[0]
                    text = f"{name} KO!" if kind == "ACTOR_KNOCKED_OUT" else f"{name} respawned"
                    self._cues.append((now + 2.0, text, (a.x, a.y - 95), color))
                continue
            if kind == "LIFE_LOST":
                a = state.actors.get(actor_id)
                if a is not None:
                    cause = str(event.get("cause", ""))
                    head = "LOST THE FIGHT!" if cause == "fight" else f"TRAPPED! ({cause.replace('_trap', '')})"
                    self._cues.append((now + 2.5, f"{head} Lives: {event.get('lives_left')}/{constants.SPY_LIVES}",
                                       (a.x, a.y - 90), THEME["critical_text"]))
                continue
            label = CUE_EVENTS.get(kind)
            obj = state.level.get_object(event.get("object_id")) if event.get("object_id") else None
            if label is None or obj is None:
                continue
            self._cues.append((now + CUE_SECONDS, label(event), obj.collision_rect.midtop, color))
        self._cue_seen = len(log)
        self._cues = [c for c in self._cues if c[0] > now]
        self._clashes = [c for c in self._clashes if now - c[0] < CLASH_SECONDS]

    def _draw_event_cues(self, state) -> None:  # kept for callers/tests: collect + draw in the current view
        self._collect_cues(state)
        self._draw_cues()

    def _draw_cues(self, target=None, to_screen=None) -> None:
        """Cues in the current world view, or (overview) straight on `target`
        with `to_screen` mapping world -> screen."""
        import math
        target = target if target is not None else self.world
        to_screen = to_screen or self._screen_pos
        now = time.monotonic()
        for started, (x, y) in self._clashes:  # a quick expanding clash burst
            t = (now - started) / CLASH_SECONDS
            sx, sy = to_screen(x, y - 30)
            radius = int(8 + 30 * t)
            for i in range(8):
                angle = i * math.pi / 4
                inner = (sx + math.cos(angle) * radius * 0.4, sy + math.sin(angle) * radius * 0.4)
                outer = (sx + math.cos(angle) * radius, sy + math.sin(angle) * radius)
                pygame.draw.line(target, (255, 230, 90), inner, outer, 3)
            pygame.draw.circle(target, (255, 255, 255), (sx, sy), max(2, int(6 * (1 - t))))
        for i, (_, text, (x, y), color) in enumerate(self._cues[-6:]):
            surf = self.debug_font.render(text, True, color)
            sx, sy = to_screen(x, y)
            rect = surf.get_rect(midbottom=(sx, sy - 4 - 12 * (i % 2)))
            backing = pygame.Surface((rect.width + 6, rect.height + 2), pygame.SRCALPHA)
            backing.fill((*THEME["panel_background"], PANEL_ALPHA))
            target.blit(backing, (rect.x - 3, rect.y - 1))
            target.blit(surf, rect.topleft)

    # -- live HUD (replaces the human-player HUD) --------------------------------
    def _draw_live_hud(self, state) -> None:
        """Kept for callers: the per-spy info is in the sidebar (draw_overlay)."""
        self.draw_overlay()

    def _snapshot(self, actor_id: str):
        if self.view is None or self.game is None:
            return None
        try:
            return self.view.actor(self.game, actor_id)
        except Exception:  # the overlay never breaks the frame
            return None

    def _draw_spy_header(self, state, actor_id: str, rect: pygame.Rect) -> int:
        """Sidebar header for one spy: name - personality, hearts (filled = left,
        hollow = lost), traps hit, fights lost, items. Returns the height used."""
        name = ACTOR_LABELS[actor_id].upper().replace(" SPY", "")
        persona = ""
        if self.view is not None:
            persona = f" - {self.view.config.actors[actor_id].personality.upper()}"
        color = SPY_TEXT_COLORS[actor_id]
        line1 = self.font.render(f"{name}{persona}", True, color)
        self.screen.blit(line1, (rect.x + 8, rect.y + 2))
        x, y = rect.x + 8, rect.y + 6 + line1.get_height()
        if getattr(state, "lives_enabled", False):
            lives = state.lives.get(actor_id, 0)
            for i in range(constants.SPY_LIVES):
                self._draw_heart((x + 7 + i * 17, y + 7), filled=i < lives, color=THEME["critical_text"])
            x += 17 * constants.SPY_LIVES + 6
        parts = []
        if getattr(state, "lives_enabled", False):
            parts.append(f"TRAPS {state.trap_hits.get(actor_id, 0)}")
            parts.append(f"FIGHTS LOST {sum(1 for f in getattr(state, 'fights', []) if f['loser_id'] == actor_id)}")
        required = state.items.required
        have = sum(1 for i in required if state.actors[actor_id].inventory.has(i))
        parts.append(f"ITEMS {have}/{len(required)}")
        if state.mission_clock_mode == "action_budget":  # per-spy clocks
            parts.append(f"TIME {_mmss(state.time_remaining_for(actor_id))}")
        text = self.label_font.render("  ".join(parts), True, THEME["primary_text"])
        self.screen.blit(text, (x, y))
        return (y - rect.y) + text.get_height() + 6

    @staticmethod
    def _phase_word(state, actor_id: str, snap) -> tuple[str, tuple[int, int, int]]:
        lifecycle = state.lifecycle(actor_id) if hasattr(state, "lifecycle") else "ACTIVE"
        if lifecycle in constants.TERMINAL_LIFECYCLES:
            return LIFECYCLE_LABELS[lifecycle], COLOR_COMMIT if lifecycle == "ESCAPED" else COLOR_WARN
        phase = getattr(snap, "phase", "idle")
        word = {"thinking": "THINKING...", "ready": "ANSWER READY", "held": "HELD", "acting": "ACTING",
                "idle": "IDLE"}.get(phase, phase.upper())
        return word, COLOR_DECIDING if phase in ("thinking", "ready", "held") else THEME["secondary_text"]

    def _draw_heart(self, center: tuple[int, int], *, filled: bool, color) -> None:
        """A drawn heart (fonts may lack the glyph). A lost life is an
        outline only - the count never relies on colour alone."""
        cx, cy = center
        points = [(cx - 7, cy - 2), (cx - 4, cy - 6), (cx, cy - 3), (cx + 4, cy - 6), (cx + 7, cy - 2), (cx, cy + 6)]
        if filled:
            pygame.draw.polygon(self.screen, color, points)
        else:
            pygame.draw.polygon(self.screen, THEME["muted_text"], points, width=1)

    def _draw_spy_outcome(self, state, actor_id: str, area: pygame.Rect) -> None:
        """Terminal banner (§32): BLACK - ESCAPED - 01:12 remaining /
        WHITE - ELIMINATED / TIMED OUT. Holding every item while still inside
        is OBJECTIVES COMPLETE - REACH THE EXIT (never "mission complete")."""
        name = ACTOR_LABELS[actor_id].upper().replace(" SPY", "")
        lifecycle = state.lifecycle(actor_id) if hasattr(state, "lifecycle") else None
        font = self.title_font
        if state.mission_complete.get(actor_id):
            left = state.escape_snapshot.get(actor_id, state.time_remaining_for(actor_id))
            text, color = f"{name} - ESCAPED - {_mmss(left)} remaining", COLOR_COMMIT
        elif getattr(state, "eliminated", {}).get(actor_id):
            text, color = f"{name} - ELIMINATED", THEME["critical_text"]
        elif lifecycle == constants.LIFECYCLE_MISSION_IMPOSSIBLE:
            text, color = f"{name} - MISSION IMPOSSIBLE", COLOR_WARN
        elif lifecycle == constants.LIFECYCLE_TIMED_OUT or compute_score(state, actor_id).failed:
            text, color = f"{name} - TIMED OUT", COLOR_WARN
        elif lifecycle == constants.LIFECYCLE_OBJECTIVES_COMPLETE:
            text, color, font = "OBJECTIVES COMPLETE - REACH THE EXIT", COLOR_DECIDING, self.label_font
        else:
            return
        surf = font.render(text, True, color)
        box = pygame.Rect(0, 0, min(area.width - 8, surf.get_width() + 24), surf.get_height() + 14)
        if self.split:
            box.center = (area.centerx, TIMER_BAR_HEIGHT + HEADER_HEIGHT + 30)
        else:
            box.center = (area.centerx, TIMER_BAR_HEIGHT + HEADER_HEIGHT + 30)
        box.clamp_ip(area)
        self._draw_panel_backing(box)
        self.screen.blit(surf, (box.x + 12, box.y + 7), area=pygame.Rect(0, 0, box.width - 24, surf.get_height()))

    def _draw_match_timer(self, rect: pygame.Rect, state) -> int:
        """One authoritative MATCH timer, top of the sidebar. Returns the height used."""
        remaining = state.mission_time_remaining
        text = f"MATCH {_mmss(remaining)}" + ("   PAUSED" if self.paused else "")
        color = THEME["critical_text"] if remaining <= 30 else THEME["primary_text"]
        surf = self.title_font.render(text, True, color)
        self.screen.blit(surf, surf.get_rect(midtop=(rect.centerx, rect.y + 6)))
        return surf.get_height() + 14

    def ranking_lines(self, match_result: dict) -> list[tuple[str, tuple[int, int, int]]]:
        """FINAL RANKING (§60/§65) - the same facts as the match_result telemetry."""
        lines = [("FINAL RANKING", THEME["primary_text"])]
        for r in match_result.get("ranking", []):
            name = ACTOR_LABELS.get(r["actor_id"], r["actor_id"])
            state_word = "SUCCESS" if r["mission_success"] else LIFECYCLE_LABELS.get(r["lifecycle"], r["lifecycle"])
            order = {1: "escaped first", 2: "escaped second"}.get(r["escape_order"], "did not escape")
            lives = "" if r.get("lives") is None else f" - {r['lives']}/{constants.SPY_LIVES} lives"
            tie = " (TIE)" if r.get("tie") else ""
            lines.append((f"{r['rank']}. {name}{tie} - {r['score_total']:g} pts - {state_word} - {order}"
                          f" - {r['time_remaining_s']:.0f} s left{lives} - traps hit {r['trap_hits']}",
                          COLOR_COMMIT if r["mission_success"] else COLOR_WARN))
            lines.append((f"     mission {r['score_mission']:g} + escape {r['score_escape']:g} + first {r['score_first_escape']:g}"
                          f" + other escaped {r['score_other_survives']:g} + time {r['score_time_bonus']:g}"
                          f" + failure {r['score_failure']:g}", THEME["secondary_text"]))
        lines.append(("One match - not a measure of model quality.   R restart   Esc quit", THEME["muted_text"]))
        return lines

    def _draw_final_ranking(self, match_result: dict) -> None:
        lines = self.ranking_lines(match_result)
        rendered = [(self.title_font if i == 0 else self.label_font).render(t, True, c) for i, (t, c) in enumerate(lines)]
        game = self.game_rect()
        width = min(game.width - 40, max(s.get_width() for s in rendered) + 40)
        height = sum(s.get_height() + 6 for s in rendered) + 24
        box = pygame.Rect(0, 0, width, height)
        box.center = game.center
        self._draw_panel_backing(box)
        pygame.draw.rect(self.screen, THEME["border"], box, width=2)
        y = box.y + 12
        for surf in rendered:
            self.screen.blit(surf, (box.x + 20, y), area=pygame.Rect(0, 0, box.width - 40, surf.get_height()))
            y += surf.get_height() + 6

    def _draw_white_spy_panel(self, state, bot_controller) -> None:
        # The telemetry panels below already show White Spy (and Black);
        # with them hidden, keep the production White panel.
        if not (self.show_telemetry and self.view is not None):
            super()._draw_white_spy_panel(state, bot_controller)

    def _draw_panel_backing(self, rect: pygame.Rect) -> None:
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((*THEME["panel_background"], PANEL_ALPHA))
        self.screen.blit(panel, rect.topleft)

    def draw_overlay(self) -> None:
        """The sidebar: MATCH timer, one section per spy (header + panel:
        compact = phase line and the agent's summary, V = everything), and
        the status / controls lines at the bottom."""
        bar = self.sidebar_rect()
        self.screen.fill(THEME["panel_background"], bar)
        pygame.draw.line(self.screen, THEME["border"], bar.topleft, bar.bottomleft, 2)
        if self.game is None:
            return
        state = self.game.state
        if self.view is not None:
            self.view.update(self.game)
        y = bar.y + self._draw_match_timer(bar, state)
        status_lines = self._status_lines(bar.width)
        bottom = bar.bottom - LINE_HEIGHT * len(status_lines) - 8
        section_h = (bottom - y) // 2
        for actor_id in ACTORS:
            section = pygame.Rect(bar.x + 6, y, bar.width - 12, section_h - 6)
            pygame.draw.rect(self.screen, THEME["border"], section, width=1)
            used = self._draw_spy_header(state, actor_id, section)
            self.header_rects[actor_id] = pygame.Rect(section.x, section.y, section.width, used)
            panel = pygame.Rect(section.x, section.y + used, section.width, section.height - used)
            self.panel_rects[actor_id] = panel
            if self.show_telemetry and self.view is not None:
                lines = self.view.actor_lines(self.view.actor(self.game, actor_id), self.show_detail)
                ly = panel.y + 2
                for text, color in lines:
                    if ly + LINE_HEIGHT > panel.bottom:
                        break  # detail view: what does not fit is cut, never drawn over the next spy
                    self.screen.blit(self.debug_font.render(text, True, color), (panel.x + 6, ly))
                    ly += LINE_HEIGHT
            y += section_h
        ly = bottom + 4
        for text, color in status_lines:
            self.screen.blit(self.debug_font.render(text, True, color), (bar.x + 8, ly))
            ly += LINE_HEIGHT

    def _status_lines(self, width_px: int) -> list[tuple[str, tuple[int, int, int]]]:
        if self.view is None or self.game is None:
            return []
        text, color = self.view.status_line(self.game, paused=self.paused, speed=self.speed, match_over=self.match_over)
        chars = max(20, (width_px - 16) // 7)
        lines, current = [], ""
        for part in text.split("   |   "):
            # break between commands ("Space pause", "P split" ...), not inside one
            sep = "  " if "  " in part else " "
            for word in part.split(sep):
                word = word.strip()
                if current and len(current) + len(word) + len(sep) > chars:
                    lines.append(current)
                    current = word
                else:
                    current = f"{current}{sep}{word}" if current else word
            lines.append(current)
            current = ""
        return [(line, color if i == 0 else COLOR_DIM) for i, line in enumerate(l for l in lines if l)]

