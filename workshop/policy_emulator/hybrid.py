"""Stage 3 milestone 4: HybridController v1 - B2 by default, the real LLM on selected decisions.

The game polls `get_action(observation) -> GameAction` every frame, as for any
controller. Between decisions the hybrid behaves exactly like the B2
SurrogateController (it is one: the same gate, features, executor and memory).
At each decision point:

    observation -> policy features -> B2 masked distribution + sampled proposal
      -> coverage-v1 -> B1 masked distribution on the same executable actions (diagnostic only)
      -> router (hybrid-router-v1, R0..R3)
           stay      -> the B2 proposal and its target-v2.1 target are committed (SurrogateController._commit)
           escalate  -> the B2 proposal is dropped; this one decision is delegated to the real
                        LLMController of the same identity, which serializes, calls the provider,
                        validates / repairs / falls back and emits its own llm_decision telemetry

Exactly one source owns a decision. The delegate shares the hybrid's
ActionExecutor and memory tracker (so its strategy memory lists the surrogate's
decisions too) and is driven for exactly one request per escalation: its digest
and throttle are opened just before the call and closed (+inf) right after the
launch, so neither its idle re-decide nor its action-boundary fall-through can
start a second request. In evaluation timing the hybrid reports "thinking"
from launch to hand-over, so the engine freezes the world and the mission clock
exactly as for an LLM spy (provider latency spends no mission time). An answer
that is ready when the spy has become terminal is discarded, never handed over;
no request is ever launched after a terminal state.

Budget guards (per spy, per match, total; counted in provider calls = requests
+ repairs) are checked before each escalation; when a cap is reached the B2
proposal is committed and the decision logged with router_budget_exhausted.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

from src import constants
from src.actions import ActionType, GameAction
from workshop.policy_emulator import router as rt
from workshop.policy_emulator import schema
from workshop.policy_emulator.artifacts import ArtifactProvenanceError, LoadedPolicy
from workshop.policy_emulator.surrogate import SurrogateController

HYBRID_RUNTIME = "hybrid"
HYBRID_RUNTIME_VERSION = "hybrid-v1"
DISCARDED_TERMINAL = "DISCARDED_TERMINAL"
INF = math.inf


@dataclass
class HybridBudget:
    """Provider-call caps (None = no cap). One object per MatchRunner: `total` spans its matches.
    With a `ledger` (milestone 4C, workshop.runners.call_ledger.CallLedger shared by every match process of an
    experiment) the total cap is the ledger's: an escalation reserves its first call there atomically before
    the delegate starts, and its repair calls are added when it ends.
    Milestone 4D: when the delegate's client is a workshop.runners.gated_client.LedgerGatedClient, the launch
    reservation is handed to it and every later call (a repair) is reserved by the gate before it is made;
    nothing is added after the fact."""
    max_llm_calls_per_spy: int | None = None
    max_llm_calls_per_match: int | None = None
    max_llm_calls_total: int | None = None
    spent_in_previous_matches: int = 0
    controllers: list = field(default_factory=list)
    exhausted_events: int = 0
    ledger: Any = None

    @staticmethod
    def _gate(controller: "HybridController"):
        client = getattr(controller.__dict__.get("_ctl"), "model_client", None)
        return client if getattr(client, "gated", False) else None

    def reserve(self, controller: "HybridController") -> tuple[bool, str | None]:
        if self.ledger is None:
            return True, None
        ok, cap = self.ledger.reserve(controller.actor_id, controller.model_name)
        if not ok:
            self.exhausted_events += 1
        elif (gate := self._gate(controller)) is not None:
            gate.prereserve()  # the delegate's initial call uses this reservation
        return ok, cap

    def settle(self, controller: "HybridController", calls: int) -> None:
        if self.ledger is None:
            return
        gate = self._gate(controller)
        if gate is not None:  # repairs were reserved before they were made; return an unused launch reservation
            if gate.unreserve():
                self.ledger.release(controller.actor_id, controller.model_name)
        elif calls > 1:
            self.ledger.add(controller.actor_id, controller.model_name, calls - 1)

    def release(self, controller: "HybridController") -> None:
        if self.ledger is not None:
            gate = self._gate(controller)
            if gate is not None:
                gate.unreserve()
            self.ledger.release(controller.actor_id, controller.model_name)

    def new_match(self) -> None:
        self.spent_in_previous_matches += self.match_calls()
        self.controllers = []

    def register(self, controller: "HybridController") -> None:
        self.controllers.append(controller)

    def match_calls(self) -> int:
        return sum(c.llm_calls for c in self.controllers)

    def available(self, controller: "HybridController") -> tuple[bool, str | None]:
        checks = [("max_llm_calls_per_spy", controller.llm_calls), ("max_llm_calls_per_match", self.match_calls())]
        if self.ledger is None:  # with a ledger the total is checked atomically in reserve()
            checks.append(("max_llm_calls_total", self.spent_in_previous_matches + self.match_calls()))
        for name, used in checks:
            cap = getattr(self, name)
            if cap is not None and used >= cap:
                self.exhausted_events += 1
                return False, name
        return True, None

    def as_dict(self) -> dict[str, Any]:
        return {"max_llm_calls_per_spy": self.max_llm_calls_per_spy, "max_llm_calls_per_match": self.max_llm_calls_per_match,
                "max_llm_calls_total": self.max_llm_calls_total, "spent_in_previous_matches": self.spent_in_previous_matches,
                "spent_this_match": self.match_calls(), "exhausted_events": self.exhausted_events,
                **({"ledger": str(self.ledger.path), "ledger_match": self.ledger.match_id} if self.ledger is not None else {})}


def _terminal(observation) -> bool:
    return bool(observation.mission_complete or getattr(observation, "time_expired", False)
                or getattr(observation, "lifecycle", "ACTIVE") in constants.TERMINAL_LIFECYCLES)


def _counter(name: str):
    """A count that adds the delegate's own counter (the base class keeps assigning its part)."""
    own = f"_own_{name}"

    def get(self) -> int:
        ctl = self.__dict__.get("_ctl")
        return self.__dict__.get(own, 0) + (int(getattr(ctl, name, 0) or 0) if ctl is not None else 0)

    def set_(self, value: int) -> None:
        ctl = self.__dict__.get("_ctl")
        self.__dict__[own] = value - (int(getattr(ctl, name, 0) or 0) if ctl is not None else 0)
    return property(get, set_)


class HybridController(SurrogateController):
    controller_kind = "hybrid"
    decision_event_type = "hybrid_decision"
    runtime_name = HYBRID_RUNTIME
    runtime_version = HYBRID_RUNTIME_VERSION
    agent_runtime = HYBRID_RUNTIME
    agent_runtime_version = HYBRID_RUNTIME_VERSION

    fallback_count = _counter("fallback_count")
    repair_count = _counter("repair_count")
    provider_failure_count = _counter("provider_failure_count")

    def __init__(self, actor_id: str, base: LoadedPolicy, b1: LoadedPolicy, llm_agent: Any, *, router_id: str,
                 protocol: dict[str, Any], budget: HybridBudget | None = None, telemetry: Any = None,
                 episode_id: str = "", seed: int | str | None = None, dataset: str = "phase-a-v2",
                 environment_seed: int | None = None, provider_name: str | None = None, model_name: str | None = None,
                 pricing: dict[str, Any] | None = None) -> None:
        _check_frozen(base, b1, router_id, protocol)
        self._llm = llm_agent
        self._ctl = getattr(llm_agent, "controller", llm_agent)
        super().__init__(actor_id, base, telemetry=telemetry, episode_id=episode_id, seed=seed, dataset=dataset,
                         environment_seed=environment_seed)
        self._b1 = b1
        self.router_id = router_id
        self.router_hash = protocol["router_hash"]
        self.router_version = protocol.get("router_version", rt.ROUTER_VERSION)
        self.budget = budget
        self.pricing = pricing or {}
        self.provider_name = provider_name or getattr(self._ctl, "provider_name", None)
        self.model_name = model_name or getattr(self._ctl, "model_name", None)
        self.execution_mode = getattr(self._ctl, "execution_mode", "evaluation")
        # one executor, one memory: the delegate continues the hybrid's own action lifecycle and history
        self._ctl._executor = self._executor
        self._ctl._memory_tracker = self._memory_tracker
        self._close_delegate()
        self._escalation: dict[str, Any] | None = None
        self.escalations = 0
        self.escalation_reasons: dict[str, int] = {}
        self.escalations_committed = 0
        self.escalations_discarded = 0
        self.escalation_launch_failures = 0
        self.budget_exhausted_decisions = 0
        self.llm_wall_seconds = 0.0
        self.last_call_committed_ready_response = False
        self._ready_at: float | None = None
        if budget is not None:
            budget.register(self)

    # -- the attribute surface the engine and the live layer read ---------------------------------

    @property
    def _thinking_in_progress(self) -> bool:
        ctl = self.__dict__.get("_ctl")
        return self.__dict__.get("_escalation") is not None and bool(ctl._thinking_in_progress or ctl._response_ready_uncommitted)

    @_thinking_in_progress.setter
    def _thinking_in_progress(self, value: bool) -> None:  # the base class initialises it; the delegate owns it
        pass

    @property
    def response_ready_uncommitted(self) -> bool:
        return self.__dict__.get("_escalation") is not None and bool(self._ctl.response_ready_uncommitted)

    @response_ready_uncommitted.setter
    def response_ready_uncommitted(self, value: bool) -> None:  # initialised by the base class; the delegate owns it
        pass

    @property
    def response_ready_at(self) -> float | None:
        if self.__dict__.get("_escalation") is not None:
            return getattr(self._ctl, "response_ready_at", None)
        return self.__dict__.get("_ready_at")

    @property
    def config(self):
        return self._ctl.config

    @property
    def llm_calls(self) -> int:
        """Provider calls of the delegate: one per request, one more per repair."""
        return int(self._ctl.decisions_requested) + int(getattr(self._ctl, "repair_count", 0) or 0)

    @property
    def escalated(self) -> bool:
        return self._escalation is not None

    # -- the controller contract ----------------------------------------------------------------

    def get_action(self, observation) -> GameAction:
        self.last_call_committed_ready_response = False
        if self._escalation is not None:
            return self._continue_escalation(observation)
        return super().get_action(observation)

    def _decide(self, observation, payload: dict[str, Any]) -> GameAction:
        d = self._propose(payload)
        executable = [a for a in d["legal"] if a not in d["unsupported"]] or ["wait"]
        b1 = self._b1.action_policy.predict(d["features"], executable)
        inputs = rt.RouterInputs(features=d["features"], b2_probs=d["prediction"].probs, b2_proposed=d["sampled"],
                                 b2_confidence=d["prediction"].confidence, coverage=d["coverage"] or {},
                                 b1_probs=b1.probs, history=tuple(self._recent))
        available, cap = self.budget.available(self) if self.budget is not None else (True, None)
        result = rt.route(self.router_id, inputs, budget_available=available)
        if result.escalate and self.budget is not None:
            reserved, ledger_cap = self.budget.reserve(self)  # the shared experiment total, atomically
            if not reserved:
                cap = ledger_cap
                result = rt.route(self.router_id, inputs, budget_available=False)
        launch_failed = False
        if result.escalate:
            action = self._start_escalation(observation, d, result, b1.probs)
            if action is not None:
                return action
            launch_failed = True
            self.escalation_launch_failures += 1
            if self.budget is not None:
                self.budget.release(self)
        if result.budget_exhausted:
            self.budget_exhausted_decisions += 1
        extra = self._router_fields(result, b1.probs, source="surrogate")
        extra.update({"router_budget_exhausted": result.budget_exhausted, "budget_cap_reached": cap,
                      "escalation_launch_failed": launch_failed, "provider_call": False})
        extra["overlay"] = {"decision": "B2", "router": self.router_id, "router_reasons": result.reasons,
                            "b1_b2_tvd": result.b1_b2_tvd, "budget_exhausted": result.budget_exhausted}
        action = self._commit(observation, payload, d, extra=extra)
        # an instant answer: under the response barrier (a hybrid against an LLM or another hybrid) it is
        # held like any handed-over decision until no other spy's model is thinking; alone it commits now
        self.last_call_committed_ready_response = True
        self._ready_at = time.monotonic()
        return action

    def _router_fields(self, result: rt.RouterResult, b1_probs: dict[str, float], *, source: str) -> dict[str, Any]:
        return {"controller_mode": self.controller_kind, "decision_source": source,
                "router_id": self.router_id, "router_version": self.router_version, "router_hash": self.router_hash,
                "router_escalated": source == "llm", "router_reasons": result.reasons, "router_signals": result.signals,
                "b2_top_action": result.b2_top, "b2_confidence": result.b2_confidence, "b2_proposed_action": result.b2_proposed,
                "b1_probs": {a: round(p, 4) for a, p in b1_probs.items()}, "b1_top_action": result.b1_top,
                "b1_b2_tvd": result.b1_b2_tvd, "b1_b2_top_differ": result.top_actions_differ,
                "critical_state": result.critical_state, "repetition_count": result.repetition_count,
                "model": self.model_name, "provider": self.provider_name,
                "personality": self.policy.metadata.get("personality")}

    # -- escalation ----------------------------------------------------------------------------

    def _open_delegate(self) -> None:
        ctl = self._ctl
        ctl._last_observation_digest = None
        ctl._next_decision_allowed_at = 0.0
        ctl._next_game_decision_allowed_at = -INF
        ctl._live_idle_since = None
        ctl.last_action = GameAction(ActionType.WAIT)  # never re-issue an older delegated action while it thinks

    def _close_delegate(self) -> None:
        self._ctl._next_decision_allowed_at = INF
        self._ctl._next_game_decision_allowed_at = INF

    def _start_escalation(self, observation, d: dict[str, Any], result: rt.RouterResult,
                          b1_probs: dict[str, float]) -> GameAction | None:
        ctl = self._ctl
        if ctl._thinking_in_progress or ctl._response_ready_uncommitted or self._executor.in_progress:
            return None
        before = {"requests": ctl.decisions_requested, "repairs": ctl.repair_count, "fallbacks": ctl.fallback_count,
                  "failures": ctl.provider_failure_count, "calls": self.llm_calls}
        self._open_delegate()
        ctl._update_memory_from_observation = lambda _obs: None  # the hybrid gate already updated memory this frame
        try:
            action = self._llm.get_action(observation)
        finally:
            ctl.__dict__.pop("_update_memory_from_observation", None)
            self._close_delegate()
        if ctl.decisions_requested != before["requests"] + 1:
            return None
        self.escalations += 1
        for reason in result.reasons:
            self.escalation_reasons[reason] = self.escalation_reasons.get(reason, 0) + 1
        self._escalation = {"result": result, "proposal": d, "b1_probs": b1_probs, "before": before,
                            "decision_index": self.decisions_requested, "committed": False, "llm_decision_id": None,
                            "launched_wall": time.monotonic(), "launch_mission_time": observation.mission_time_remaining}
        self.status = "THINKING"
        self.last_reason = f"{self.router_id} escalated: {', '.join(result.reasons)}"
        self.last_decision = {**(self.last_decision or {}), "decision": "LLM ESCALATION", "router": self.router_id,
                              "router_reasons": result.reasons, "b1_b2_tvd": result.b1_b2_tvd,
                              "coverage": result.coverage, "emulator": self.policy.emulator,
                              "model_display": self.policy.metadata.get("model_display"),
                              "personality": self.policy.metadata.get("personality")}
        return self._after_delegate(observation, action)

    def _continue_escalation(self, observation) -> GameAction:
        ctl = self._ctl
        if _terminal(observation):
            if ctl._thinking_in_progress:
                self.status = "IDLE"
                return GameAction(ActionType.WAIT)  # the answer is discarded when it arrives
            if ctl._response_ready_uncommitted:
                ctl._cancel_ready_response(DISCARDED_TERMINAL)
                self._end_escalation(observation, "discarded_terminal")
            else:
                self._end_escalation(observation, "finished_at_terminal")
            self.status = "IDLE"
            return GameAction(ActionType.WAIT)
        return self._after_delegate(observation, self._llm.get_action(observation))

    def _after_delegate(self, observation, action: GameAction) -> GameAction:
        ctl, esc = self._ctl, self._escalation
        if ctl.last_call_committed_ready_response and not esc["committed"]:
            esc["committed"] = True
            self.last_call_committed_ready_response = True
            self._ready_at = getattr(ctl, "response_ready_at", None)
            self._commit_escalation(observation)
        if esc["committed"] and not self._executor.in_progress and not ctl._thinking_in_progress \
                and not ctl._response_ready_uncommitted:
            self._escalation = None
            self.status = "IDLE"
        else:
            self.status = "THINKING" if not esc["committed"] else "EXECUTING"
        self.last_action = action
        return action

    def _llm_event(self, decision_id: str | None) -> dict[str, Any]:
        events = getattr(self.telemetry, "events", None) or []
        for e in reversed(events):
            if e.get("event_type") == "llm_decision" and e.get("decision_id") == decision_id:
                return e
        return {}

    def _commit_escalation(self, observation) -> None:
        esc, ctl = self._escalation, self._ctl
        decision_id = ctl._last_decision_id
        esc["llm_decision_id"] = decision_id
        self._last_decision_id = decision_id
        event = self._llm_event(decision_id)
        executed = event.get("executed_action") or event.get("validated_action") or {}
        committed = str(executed.get("action") or "wait").lower() if event else str(ctl.last_action.type.value).lower()
        self._recent.append(committed)
        self.escalations_committed += 1
        self._emit_llm_decision(observation, esc, event, executed, committed, outcome="committed")

    def _end_escalation(self, observation, outcome: str) -> None:
        esc = self._escalation
        if esc is not None and not esc["committed"]:
            self.escalations_discarded += 1
            self._emit_llm_decision(observation, esc, self._llm_event(self._ctl._last_decision_id), {}, None, outcome=outcome)
        self._escalation = None

    def _emit_llm_decision(self, observation, esc: dict[str, Any], event: dict[str, Any], executed: dict[str, Any],
                           committed: str | None, *, outcome: str) -> None:
        ctl, b = self._ctl, esc["before"]
        wall = time.monotonic() - esc["launched_wall"]
        self.llm_wall_seconds += wall
        d, result = esc["proposal"], esc["result"]
        usage = {"input_tokens": event.get("input_tokens"), "cached_input_tokens": event.get("cached_input_tokens"),
                 "output_tokens": event.get("output_tokens"), "latency_ms": event.get("latency_ms"),
                 "provider_calls": self.llm_calls - b["calls"], "repairs": ctl.repair_count - b["repairs"]}
        usage["estimated_cost_usd"] = estimated_cost(self.pricing, self.model_name, usage)
        if self.budget is not None:  # the first call was reserved at launch; count the repair calls it made
            self.budget.settle(self, usage["provider_calls"])
        overlay = {**(self.last_decision or {}), "decision": "LLM", "router": self.router_id,
                   "router_reasons": result.reasons, "b1_b2_tvd": result.b1_b2_tvd, "coverage": result.coverage,
                   "action": committed, "target": executed.get("target_id"), "llm_outcome": outcome}
        self.last_decision = overlay
        if self.telemetry is None:
            return
        self.telemetry.emit_event({
            "event_type": self.decision_event_type, "episode_id": self.episode_id, "actor_id": self.actor_id,
            "decision_id": esc["llm_decision_id"] or event.get("decision_id"), "decision_index": esc["decision_index"],
            "decision_number": esc["decision_index"],
            "mission_time_remaining": round(observation.mission_time_remaining, 3),
            "mission_seconds_while_thinking": round(esc["launch_mission_time"] - observation.mission_time_remaining, 3),
            "policy": self.policy.policy_key, "emulator": self.policy.emulator, "target_emulator": self.policy.target_version,
            "policy_rng_seed": self.policy_rng_seed, "environment_seed": self.environment_seed,
            **self._router_fields(result, esc["b1_probs"], source="llm"),
            "probs": {a: round(p, 4) for a, p in d["prediction"].probs.items()}, "confidence": round(d["prediction"].confidence, 4),
            "coverage": d["coverage"] or None, "coverage_label": (d["coverage"] or {}).get("coverage"),
            "coverage_score": (d["coverage"] or {}).get("score"), "coverage_reasons": (d["coverage"] or {}).get("reasons") or [],
            "features": d["features"], "legal_actions": d["legal"],
            "intended_action": committed, "intended_target": executed.get("target_id"),
            "sampled_action": committed, "target_id": executed.get("target_id"), "executed_action": executed or None,
            "b2_counterfactual": {"action": d["sampled"], "target": d["sampled_target"]},
            "llm_decision_id": esc["llm_decision_id"], "llm_outcome": outcome,
            "action_source": event.get("action_source"), "fallback_used": bool(event.get("fallback_used")),
            "repair_attempted": bool(event.get("repair_attempted")),
            "provider_failure": ctl.provider_failure_count > b["failures"], "provider_call": True,
            "valid": bool(event.get("valid", True)) and outcome == "committed",
            "llm_usage": usage, "decision_wall_seconds": round(wall, 3),
            "budget": self.budget.as_dict() if self.budget is not None else None,
            "surrogate_overlay": overlay,
        })

    # -- reporting ------------------------------------------------------------------------------

    def provenance(self) -> dict[str, Any]:
        return {**super().provenance(), "router_id": self.router_id, "router_version": self.router_version,
                "router_hash": self.router_hash, "b1_artifact_hash": self._b1.metadata.get("content_hash"),
                "llm_provider": self.provider_name, "llm_model": self.model_name,
                "llm_runtime": getattr(self._llm, "runtime_name", None)}

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "router_id": self.router_id, "escalations": self.escalations,
                "escalation_reasons": dict(self.escalation_reasons), "escalations_committed": self.escalations_committed,
                "escalations_discarded": self.escalations_discarded,
                "escalation_launch_failures": self.escalation_launch_failures,
                "budget_exhausted_decisions": self.budget_exhausted_decisions,
                "llm_requests": int(self._ctl.decisions_requested), "llm_provider_calls": self.llm_calls,
                "llm_repairs": int(self._ctl.repair_count), "llm_fallbacks": int(self._ctl.fallback_count),
                "llm_provider_failures": int(self._ctl.provider_failure_count),
                "llm_wall_seconds": round(self.llm_wall_seconds, 3),
                "budget": self.budget.as_dict() if self.budget is not None else None}


def estimated_cost(pricing: dict[str, Any], model: str | None, usage: dict[str, Any]) -> float | None:
    """USD from a {per_million_tokens: {model: {input, cached_input, output}}} block; None when unpriced."""
    price = ((pricing or {}).get("per_million_tokens") or {}).get(model or "")
    if not price or usage.get("input_tokens") is None:
        return None
    return round(((usage.get("input_tokens") or 0) * price["input"] + (usage.get("cached_input_tokens") or 0)
                  * price.get("cached_input", price["input"]) + (usage.get("output_tokens") or 0) * price["output"]) / 1e6, 6)


def lab_pricing() -> dict[str, Any]:
    """The pricing block of the cross-model lab (read only; Sonnet 5.5 has no price there)."""
    from workshop.runners.common import PROJECT_ROOT
    path = PROJECT_ROOT / "workshop/labs/cross_model_full_match/config.json"
    try:
        return json.loads(path.read_text()).get("pricing") or {}
    except (OSError, ValueError):
        return {}


def _check_frozen(base: LoadedPolicy, b1: LoadedPolicy, router_id: str, protocol: dict[str, Any]) -> None:
    problems = []
    if base.emulator != rt.BASE_EMULATOR:
        problems.append(f"base emulator {base.emulator}, the hybrid base is {rt.BASE_EMULATOR}")
    if base.target_version != rt.TARGET_POLICY:
        problems.append(f"target policy {base.target_version}, the hybrid uses {rt.TARGET_POLICY}")
    if base.coverage is None:
        problems.append("the base policy has no coverage estimator")
    if b1.emulator != rt.DIAGNOSTIC_EMULATOR:
        problems.append(f"diagnostic emulator {b1.emulator}, expected {rt.DIAGNOSTIC_EMULATOR}")
    if base.policy_key != b1.policy_key:
        problems.append(f"B1 {b1.policy_key} and B2 {base.policy_key} are different identities")
    if router_id not in (protocol.get("routers") or {}):
        problems.append(f"unknown router {router_id!r} for {protocol.get('router_version')}")
    ident = (protocol.get("identities") or {}).get(base.policy_key) or {}
    expected = {"base_emulator_hash": base.metadata.get("content_hash"),
                "target_policy_hash": (base.target_metadata or {}).get("content_hash"),
                "coverage_hash": base.coverage_hash, "b1_hash": b1.metadata.get("content_hash")}
    problems += [f"router protocol {k} {ident.get(k)} != artifact {v}" for k, v in expected.items() if ident.get(k) != v]
    if problems:
        raise ArtifactProvenanceError("hybrid controller: " + "; ".join(problems))


def model_id_of(policy_key: str) -> str:
    model, _ = schema.parse_policy_key(policy_key)
    return schema.MODEL_BY_NAME[model]["model_id"]
