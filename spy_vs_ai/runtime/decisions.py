"""Structured decision records, telemetry totals and the match summary (FINAL-2 spec 17, 18, 23-25, 40).

Every field comes from the controllers' own telemetry events (surrogate_decision, hybrid_decision, llm_decision,
opponent_decision, provider_call) or from the actor-scoped observation the agent received. Nothing is inferred:
a field the telemetry does not carry is simply absent, and no hidden reasoning is requested or shown. The only
model text displayed is the `reason` field of the structured action the model returned.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from spy_vs_ai.providers.base import FALLBACK_USED, user_error

ACTORS = ("black_spy", "white_spy")
LABEL = {"black_spy": "BLACK SPY", "white_spy": "WHITE SPY"}
DECISION_EVENTS = ("surrogate_decision", "hybrid_decision", "llm_decision", "opponent_decision")


def action_text(action: Any) -> str:
    """'INSPECT safe' from a GameAction / dict / string."""
    if action is None:
        return "-"
    if isinstance(action, str):
        return action.upper()
    if isinstance(action, dict):
        name = str(action.get("action") or action.get("type") or "?")
        target = action.get("target_id") or action.get("target")
        extra = f" ({action['trap_type']})" if action.get("trap_type") else ""
        code = " code ****" if action.get("code") else ""
        return f"{name.upper()} {target}{extra}{code}".strip() if target else f"{name.upper()}{extra}{code}"
    kind = getattr(getattr(action, "type", None), "value", None)
    if kind is not None:
        return action_text({"action": kind, "target_id": getattr(action, "target_id", None),
                            "trap_type": getattr(action, "trap_type", None), "code": getattr(action, "code", None)})
    return str(action)


def _top(probs: Any, n: int = 3) -> list[tuple[str, float]]:
    if not isinstance(probs, dict):
        return []
    return [(str(k), round(float(v), 3)) for k, v in sorted(probs.items(), key=lambda kv: -float(kv[1]))[:n]]


def summarize_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """What the spy saw (its own obs-v1 payload; never world truth)."""
    if not isinstance(payload, dict):
        return {}
    room = payload.get("room") or {}
    me = payload.get("self") or {}
    mission = payload.get("mission") or {}
    opponent = payload.get("opponent") or {}
    time_ = payload.get("time") or {}
    objects = []
    for o in (room.get("objects") or [])[:8]:
        if isinstance(o, dict):
            flags = [f for f in ("inspected", "searched", "locked") if o.get(f)]
            objects.append(f"{o.get('id')}" + (f" ({', '.join(flags)})" if flags else ""))
    last_seen = opponent.get("last_seen")
    if isinstance(last_seen, dict):
        last_seen = last_seen.get("room") or last_seen.get("room_id") or last_seen
    return {"room": room.get("id") or me.get("room"), "near": me.get("near_object_id"),
            "inventory": list(me.get("inventory") or mission.get("have") or []),
            "required": list(mission.get("required_items") or []), "ready_to_exit": mission.get("ready_to_exit"),
            "lives": me.get("lives"), "opponent_visible": opponent.get("visible"), "opponent_last_seen": last_seen,
            "time_remaining": round(float(time_["remaining"]), 1) if isinstance(time_.get("remaining"), int | float) else None,
            "room_objects": objects}


def legal_text(legal: Any, limit: int = 12) -> list[str]:
    out = []
    for entry in legal or []:
        if isinstance(entry, dict):
            out.append(action_text(entry))
        elif isinstance(entry, str):
            out.append(entry.upper())
    return out[:limit] + ([f"... {len(out) - limit} more"] if len(out) > limit else [])


class DecisionLog:
    """Incremental: feed it the growing telemetry event list each frame (bounded history, spec 40)."""

    def __init__(self, agents: dict[str, dict[str, Any]], *, history: int = 400) -> None:
        self.agents = agents  # actor_id -> {"policy", "label", "kind", "personality", "provider", "model"}
        self.records: deque[dict[str, Any]] = deque(maxlen=history)
        self.seen = 0
        self.llm_by_id: dict[str, dict[str, Any]] = {}
        self.calls_by_decision: dict[str, int] = {}  # decision_id -> physical provider calls (provider_call events)
        self.observations: dict[str, Any] = {}  # actor -> latest observation payload the agent was given
        self.totals = {a: {"decisions": 0, "provider_calls": 0, "provider_call_failures": 0, "escalations": 0,
                           "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "latency_ms_sum": 0.0,
                           "latency_n": 0, "fallbacks": 0, "repairs": 0, "budget_exhausted": 0, "errors": {}}
                       for a in ACTORS}
        self.count = 0

    def note_observation(self, actor_id: str, payload: dict[str, Any] | None) -> None:
        self.observations[actor_id] = payload

    def update(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        new = []
        while self.seen < len(events):
            event = events[self.seen]
            self.seen += 1
            record = self._consume(event)
            if record is not None:
                self.records.append(record)
                new.append(record)
        return new

    def _llm_part(self, e: dict[str, Any]) -> dict[str, Any]:
        validated = e.get("validated_action") if isinstance(e.get("validated_action"), dict) else None
        category = e.get("error_type")
        return {"action": action_text(validated) if validated else None,
                "stated_reason": (validated or {}).get("reason") if validated else None,
                "latency_ms": e.get("latency_ms"), "input_tokens": e.get("input_tokens"),
                "output_tokens": e.get("output_tokens"), "repair": bool(e.get("repair_attempted")),
                "fallback": bool(e.get("fallback_used")), "recovery": e.get("recovery_outcome"),
                "provider_calls": self.calls_by_decision.get(e.get("decision_id")) if e.get("decision_id") else None,
                "error": user_error(category, e.get("error_message")) if category else None}

    def _consume(self, e: dict[str, Any]) -> dict[str, Any] | None:
        kind = e.get("event_type")
        actor = e.get("actor_id")
        if actor not in self.totals:
            return None
        t = self.totals[actor]
        if kind == "provider_call":
            t["provider_calls"] += 1
            if e.get("decision_id"):
                self.calls_by_decision[e["decision_id"]] = self.calls_by_decision.get(e["decision_id"], 0) + 1
            if e.get("success") is False:
                t["provider_call_failures"] += 1
                cat = e.get("provider_error_type") or "provider_error"
                t["errors"][cat] = t["errors"].get(cat, 0) + 1
            return None
        if kind not in DECISION_EVENTS:
            return None
        agent = self.agents.get(actor, {})
        if kind == "llm_decision":
            self._count_llm(t, e)
            if agent.get("kind") == "hybrid":
                if e.get("decision_id"):
                    self.llm_by_id[e["decision_id"]] = e
                return None
        self.count += 1
        t["decisions"] += 1
        rec: dict[str, Any] = {"seq": self.count, "id": e.get("decision_id"), "spy": actor, "event": kind,
                               "number": e.get("decision_number", e.get("decision_index")),
                               "policy": agent.get("label"), "personality": agent.get("personality"),
                               "game_time": e.get("mission_elapsed", e.get("game_time")),
                               "executed": action_text(e.get("executed_action") or e.get("selected_action")
                                                       or e.get("sampled_action"))}
        if kind == "llm_decision":
            rec.update({"provider": e.get("provider"), "model": e.get("model"), "llm": self._llm_part(e),
                        "observation": summarize_payload(e.get("observation")), "legal": legal_text(e.get("legal_actions")),
                        "fallback": bool(e.get("fallback_used"))})
            return rec
        if kind == "opponent_decision":
            rec["note"] = "scripted bot"
            rec["observation"] = summarize_payload(self.observations.get(actor))
            return rec
        # surrogate / hybrid
        rec["observation"] = summarize_payload(self.observations.get(actor))
        rec["legal"] = legal_text([{"action": a} for a in e.get("legal_actions") or []] if e.get("legal_actions")
                                  and isinstance((e.get("legal_actions") or [None])[0], str) else e.get("legal_actions"))
        proposal = {"action": action_text({"action": e.get("sampled_action") or e.get("b2_proposed_action"),
                                           "target_id": e.get("target_id")}),
                    "confidence": e.get("confidence") if e.get("confidence") is not None else e.get("b2_confidence"),
                    "probs": _top(e.get("probs")), "coverage": e.get("coverage_label")}
        base = agent.get("base")
        rec["surrogate"] = {**proposal, "name": {"b1": "B1", "b2_v1": "B2-v1", "b2_v2": "B2-v2"}.get(base, "surrogate")}
        if kind == "hybrid_decision":
            escalated = e.get("decision_source") == "llm"
            if escalated:
                t["escalations"] += 1
                counter = e.get("b2_counterfactual") or {}
                rec["surrogate"]["action"] = action_text(counter) if counter else action_text(e.get("b2_proposed_action"))
            if e.get("router_budget_exhausted"):
                t["budget_exhausted"] += 1
            rec["b1"] = {"action": action_text(e.get("b1_top_action")), "probs": _top(e.get("b1_probs")),
                         "tvd_to_b2": e.get("b1_b2_tvd")}
            rec["router"] = {"id": e.get("router_id"), "escalate": bool(e.get("router_escalated")),
                             "reasons": list(e.get("router_reasons") or []),
                             "budget_exhausted": bool(e.get("router_budget_exhausted")), "escalated": escalated}
            rec["provider"], rec["model"] = e.get("provider"), e.get("model")
            if escalated:
                llm = self.llm_by_id.get(e.get("llm_decision_id")) or {}
                usage = e.get("llm_usage") or {}
                part = self._llm_part(llm) if llm else {}
                part.update({k: usage.get(k, part.get(k)) for k in ("latency_ms", "input_tokens", "output_tokens")})
                if usage.get("provider_calls") is not None:
                    part["provider_calls"] = usage.get("provider_calls")
                part["fallback"] = bool(e.get("fallback_used") or part.get("fallback"))
                part["outcome"] = e.get("llm_outcome")
                rec["llm"] = part
                if llm.get("observation"):
                    rec["observation"] = summarize_payload(llm.get("observation"))
                rec["fallback"] = part["fallback"]
        return rec

    @staticmethod
    def _count_llm(t: dict[str, Any], e: dict[str, Any]) -> None:
        for k in ("input_tokens", "output_tokens", "cached_input_tokens"):
            if isinstance(e.get(k), int | float):
                t[k] += int(e[k])
        if isinstance(e.get("latency_ms"), int | float):
            t["latency_ms_sum"] += float(e["latency_ms"])
            t["latency_n"] += 1
        if e.get("fallback_used"):
            t["fallbacks"] += 1
        if e.get("repair_attempted"):
            t["repairs"] += 1

    def telemetry(self) -> dict[str, dict[str, Any]]:
        out = {}
        for a, t in self.totals.items():
            out[a] = {k: v for k, v in t.items() if k not in ("latency_ms_sum", "latency_n")}
            out[a]["avg_latency_ms"] = round(t["latency_ms_sum"] / t["latency_n"], 1) if t["latency_n"] else None
        return out

    def recent(self, n: int = 50) -> list[dict[str, Any]]:
        return list(self.records)[-n:]


def inspector_lines(rec: dict[str, Any] | None) -> list[str]:
    """The decision inspector (spec 18): only fields that exist in the record."""
    if not rec:
        return ["no decision yet"]
    lines = [f"{LABEL.get(rec['spy'], rec['spy'])} - decision {rec.get('number', rec['seq'])}"
             + (f"  (t={rec['game_time']:.1f} s)" if isinstance(rec.get("game_time"), int | float) else ""),
             f"Policy: {rec.get('policy')}  |  personality {rec.get('personality')}"]
    if rec.get("id"):
        lines.append(f"Decision id: {rec['id']}")
    obs = rec.get("observation") or {}
    if obs:
        lines.append("Observation (what the spy saw):")
        lines.append(f"  room {obs.get('room')}  near {obs.get('near') or '-'}  lives {obs.get('lives')}")
        lines.append(f"  have {', '.join(obs.get('inventory') or []) or 'nothing'} / need "
                     f"{', '.join(obs.get('required') or []) or '?'}" + ("  READY TO EXIT" if obs.get("ready_to_exit") else ""))
        seen = "visible" if obs.get("opponent_visible") else f"last seen {obs.get('opponent_last_seen') or 'never'}"
        lines.append(f"  opponent {seen}  time left {obs.get('time_remaining')} s")
        if obs.get("room_objects"):
            lines.append("  objects: " + ", ".join(obs["room_objects"][:5]))
    if rec.get("legal"):
        lines.append("Legal: " + ", ".join(rec["legal"][:8]))
    s = rec.get("surrogate")
    if s:
        conf = f"  confidence {s['confidence']:.2f}" if isinstance(s.get("confidence"), int | float) else ""
        lines.append(f"{s.get('name')}: {s.get('action')}{conf}" + (f"  coverage {s['coverage']}" if s.get("coverage") else ""))
        if s.get("probs"):
            lines.append("  p: " + ", ".join(f"{a} {p:.2f}" for a, p in s["probs"]))
    b1 = rec.get("b1")
    if b1:
        tvd = f"  (B1/B2 TVD {b1['tvd_to_b2']:.2f})" if isinstance(b1.get("tvd_to_b2"), int | float) else ""
        lines.append(f"B1: {b1.get('action')}{tvd}")
    r = rec.get("router")
    if r:
        lines.append(f"{r.get('id')}: {'ESCALATE' if r.get('escalate') else 'no escalation'}"
                     + (f"  reason: {', '.join(r['reasons'])}" if r.get("reasons") else "")
                     + ("  (budget exhausted)" if r.get("budget_exhausted") else ""))
    llm = rec.get("llm")
    if llm:
        if rec.get("provider"):
            lines.append(f"Provider: {rec.get('provider')}  model {rec.get('model')}")
        lines.append(f"LLM action: {llm.get('action') or '-'}"
                     + (f"  ({llm['outcome']})" if llm.get("outcome") and llm.get("outcome") != "committed" else ""))
        if llm.get("stated_reason"):
            lines.append(f"  stated reason (returned field): {str(llm['stated_reason'])[:90]}")
        meta = []
        if isinstance(llm.get("latency_ms"), int | float):
            meta.append(f"latency {llm['latency_ms'] / 1000:.2f} s")
        if llm.get("input_tokens") is not None or llm.get("output_tokens") is not None:
            meta.append(f"tokens {llm.get('input_tokens') or 0} in / {llm.get('output_tokens') or 0} out")
        if llm.get("provider_calls") is not None:
            meta.append(f"{llm['provider_calls']} provider call(s)")
        if meta:
            lines.append("  " + "  ".join(meta))
        if llm.get("error"):
            lines.append(f"  {llm['error']}")
        if llm.get("repair"):
            lines.append("  retry failed" if llm.get("fallback") else "  repair attempted")
        if llm.get("fallback"):
            lines.append(f"  {FALLBACK_USED}")
    elif rec.get("note"):
        lines.append(rec["note"])
    lines.append(f"Executed: {rec.get('executed')}")
    return lines


def history_line(rec: dict[str, Any]) -> str:
    who = "B" if rec["spy"] == "black_spy" else "W"
    s = (rec.get("surrogate") or {}).get("action")
    esc = (rec.get("router") or {}).get("escalated")
    lat = (rec.get("llm") or {}).get("latency_ms")
    parts = [f"{rec['seq']:>4} {who}"]
    if s and s != rec.get("executed"):
        parts.append(f"{s} ->")
    parts.append(rec.get("executed") or "-")
    if esc:
        parts.append("[LLM]")
    if isinstance(lat, int | float):
        parts.append(f"{lat / 1000:.1f}s")
    if rec.get("fallback"):
        parts.append("FALLBACK")
    return " ".join(parts)


def match_summary(game, log: DecisionLog, *, wall_seconds: float | None = None) -> dict[str, Any]:
    """Spec 25: outcome per spy and in total, from the engine state and the decision telemetry."""
    from src.scoring import compute_score
    state = game.state
    result = getattr(game, "match_result", None) or {}
    ranking = {r["actor_id"]: r for r in result.get("ranking", [])}
    tel = log.telemetry()
    spies = {}
    for a in ACTORS:
        failed = compute_score(state, a).failed
        outcome = ("escaped" if state.mission_complete.get(a) else "eliminated" if state.eliminated.get(a)
                   else "mission impossible" if a in state.mission_impossible else "timed out" if failed else "in progress")
        r = ranking.get(a, {})
        spies[a] = {"outcome": outcome, "rank": r.get("rank"), "score": r.get("score_total"),
                    "mission_completed": bool(state.mission_complete.get(a)), "escaped": outcome == "escaped",
                    "timed_out": outcome == "timed out", "eliminated": outcome == "eliminated",
                    "lives": state.lives.get(a) if state.lives_enabled else None, **tel[a]}
    total = {k: sum(spies[a][k] or 0 for a in ACTORS) for k in ("decisions", "provider_calls", "escalations", "input_tokens",
                                                                 "output_tokens", "fallbacks", "repairs")}
    winners = [a for a, r in ranking.items() if r.get("rank") == 1]
    tie = len(winners) > 1 or any(r.get("tie") for r in ranking.values())
    winner = "tie" if tie else (winners[0] if winners else None)
    return {"winner": winner, "escape_order": result.get("escape_order"), "spies": spies, "total": total,
            "game_time_s": round(state.elapsed_time, 1), "wall_seconds": None if wall_seconds is None else round(wall_seconds, 1),
            "complete": bool(result)}


def summary_lines(summary: dict[str, Any]) -> list[str]:
    w = summary.get("winner")
    lines = ["MATCH SUMMARY",
             f"Winner: {'tie' if w == 'tie' else LABEL.get(w, 'none (no spy ranked first)') if w else 'none'}"
             f"   game time {summary.get('game_time_s')} s" + (f"   wall {summary['wall_seconds']} s" if summary.get("wall_seconds") else "")]
    for a in ACTORS:
        s = summary["spies"][a]
        lines.append(f"{LABEL[a]}: {s['outcome'].upper()}  rank {s.get('rank')}  score {s.get('score')}"
                     f"  lives {s.get('lives')}")
        lines.append(f"   decisions {s['decisions']}  provider calls {s['provider_calls']}  escalations {s['escalations']}"
                     f"  tokens {s['input_tokens']} in / {s['output_tokens']} out  fallbacks {s['fallbacks']}"
                     + (f"  avg latency {s['avg_latency_ms'] / 1000:.2f} s" if s.get("avg_latency_ms") else ""))
    t = summary["total"]
    lines.append(f"TOTAL: decisions {t['decisions']}  provider calls {t['provider_calls']}  escalations {t['escalations']}"
                 f"  tokens {t['input_tokens']} in / {t['output_tokens']} out  fallbacks {t['fallbacks']}")
    return lines


__all__ = ["DecisionLog", "inspector_lines", "history_line", "match_summary", "summary_lines", "summarize_payload",
           "action_text"]
