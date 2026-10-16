"""Historical replay adapter (FINAL-2 spec 28): watch a frozen research match again, where its telemetry permits.

The research telemetry has no engine-input log, so a historical match is rebuilt by re-simulation, then checked:

1. the match's own frozen configuration (match_spec.json) runs again through the frozen research match process
   (workshop.scripts.confirmatory_trial's runner: fixed 1/60 s simulated step, evaluation timing, the same seeds, the
   Stage-5B / 5C arm bindings), with an engine recorder attached and no provider:
   - call-free arms (R0: "B2 loop pathology" in Stage 4C, the Stage-5B / 5C B2-v1 and B2-v2 arms) need nothing else;
   - provider arms (REAL, Hybrid-v1, Hybrid-v2) get a recorded-answer client that hands each spy, in order, the
     structured action its model returned in the historical match (validated_action; the executed fallback where the
     historical call failed). No model, no network, no credential;
2. the regenerated engine events (every engine_event, in order, with its game time) and the final ranking are compared
   with the historical telemetry. Only an exact match is shown (REPRODUCED); otherwise the adapter reports DIVERGED
   with the first difference and shows nothing - a re-simulation that differs is not that match.

The historical files are only read. The re-simulation writes into a temporary directory. The viewer then shows the
historical decision telemetry (latency, tokens, provider calls, router reasons) over the reproduced frames.
Supported: matches run by the confirmatory match process (Stages 4C, 5B, 5C: run.json kind confirmatory_*_match).
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spy_vs_ai.runtime import config as rc
from spy_vs_ai.runtime.recorder import EngineRecorder, bundle as replay_bundle

DECISION_TYPES = ("surrogate_decision", "hybrid_decision", "llm_decision", "opponent_decision")
SUPPORTED_KIND_PREFIX = "confirmatory_"
ANSWER_SECONDS = 0.05
EXAMPLES = {  # spec 28's examples, one frozen match each (Stage 4C level 2, seed 9009, cautious, orientation A)
    "REAL": "workshop_runs/stage4c-confirmatory/matches/real/level_02-9009-cautious-A",
    "B2 loop pathology (B2-v1 through Hybrid R0)": "workshop_runs/stage4c-confirmatory/matches/r0/level_02-9009-cautious-A",
    "Hybrid-v1": "workshop_runs/stage5c-confirmatory/matches/hybrid_v1/level_02-9200-cautious-A",
    "Hybrid-v2": "workshop_runs/stage5c-confirmatory/matches/hybrid_v2/level_02-9200-cautious-A",
}


class HistoricalError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def policy_of(entry: dict[str, Any]) -> str:
    arm, cand = entry.get("arm"), entry.get("candidate")
    if arm == "real":
        return "real_llm"
    if arm == "r0":
        return "b2_v2" if cand == "b2v2" else "b2_v1"
    if arm == "hybrid":
        return "hybrid_v2" if cand == "b2v2" else "hybrid_v1"
    raise HistoricalError(f"unknown research arm '{arm}'")


class RecordedAnswerClient:
    """Hands back, in order, the structured actions one spy's model returned in a historical match.

    Each answer takes ANSWER_SECONDS of wall time, so that - as with the historical provider latency - the request is
    still in flight when the frame checks it and the world stays frozen until the barrier commits (evaluation timing:
    how long a model thinks never changes game time, only whether the frame is frozen)."""
    fake = True

    def __init__(self, spec, answers: list[dict[str, Any]]) -> None:
        self.model_name, self.provider = spec.model, spec.provider
        self.answers = deque(answers)
        self.calls = 0
        self.exhausted = 0

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        from src.llm.clients.base import ModelResponse
        self.calls += 1
        time.sleep(ANSWER_SECONDS)
        if self.answers:
            pick = self.answers.popleft()
        else:
            self.exhausted += 1
            pick = {"action": "wait", "reason": "no recorded answer left"}
        return ModelResponse(payload=dict(pick), raw_text=json.dumps(pick), provider=self.provider, model=self.model_name,
                             latency_ms=0.0, input_tokens=0, output_tokens=0)


def recorded_answers(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        if e.get("event_type") != "llm_decision":
            continue
        action = e.get("executed_action") if e.get("fallback_used") else e.get("validated_action")
        if not isinstance(action, dict):
            action = e.get("validated_action") or e.get("executed_action") or {"action": "wait"}
        answer = {k: v for k, v in action.items() if v is not None}
        answer.setdefault("reason", "recorded answer")
        out[e.get("actor_id")].append(answer)
    return out


def _engine_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in e.items() if k != "episode_id"} for e in events if e.get("event_type") == "engine_event"]


def _ranking(events: list[dict[str, Any]]) -> Any:
    result = next((e for e in reversed(events) if e.get("event_type") == "match_result"), None)
    return None if result is None else {"ranking": result.get("ranking"), "escape_order": result.get("escape_order")}


def compare(historical: list[dict[str, Any]], regenerated: list[dict[str, Any]]) -> dict[str, Any]:
    h, r = _engine_events(historical), _engine_events(regenerated)
    first = next((i for i in range(min(len(h), len(r))) if json.dumps(h[i], sort_keys=True) != json.dumps(r[i], sort_keys=True)),
                 None if len(h) == len(r) else min(len(h), len(r)))
    ranking_same = json.dumps(_ranking(historical), sort_keys=True) == json.dumps(_ranking(regenerated), sort_keys=True)
    out = {"historical_engine_events": len(h), "regenerated_engine_events": len(r), "first_difference": first,
           "engine_events_identical": first is None, "ranking_identical": ranking_same}
    if first is not None:
        out["historical_at_difference"] = h[first] if first < len(h) else None
        out["regenerated_at_difference"] = r[first] if first < len(r) else None
    return out


def merge_telemetry(regenerated: list[dict[str, Any]], historical: list[dict[str, Any]], *, provider_arm: bool,
                    keep: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[int]]:
    """The viewer's telemetry: the regenerated event order, with every decision event replaced by the historical one
    (same type, spy and ordinal - the decisions are the same once the engine events match) and, for provider arms,
    the historical provider_call events (real latency / tokens / errors) in place of the recorded-answer ones."""
    hist_by: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    calls_by: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for e in historical:
        t = e.get("event_type")
        if t in DECISION_TYPES:
            hist_by[(t, e.get("actor_id"))].append(e)
        elif t == "provider_call":
            calls_by[e.get("decision_id")].append(e)
    seen: Counter = Counter()
    out: list[dict[str, Any]] = []
    index: list[int] = []
    for i, e in enumerate(regenerated):
        t = e.get("event_type")
        if t not in keep or (t == "provider_call" and provider_arm):
            continue
        if t in DECISION_TYPES:
            key = (t, e.get("actor_id"))
            k = seen[key]
            seen[key] += 1
            if k < len(hist_by[key]):
                e = hist_by[key][k]
                if t == "llm_decision" and provider_arm:
                    for call in calls_by.get(e.get("decision_id"), []):
                        out.append(call)
                        index.append(i)
        out.append(e)
        index.append(i)
    return out, index


def runtime_config_for(entry: dict[str, Any], live) -> rc.GameRuntimeConfig:
    policy = policy_of(entry)
    agents = {}
    for side, actor in (("black", live.black), ("white", live.white)):
        identity = (actor.policy or "").split(":")[0] or None
        if policy == "real_llm":
            agents[side] = rc.AgentRuntimeConfig(policy, actor.personality, provider=actor.provider, model=actor.model)
        elif policy.startswith("hybrid"):
            agents[side] = rc.AgentRuntimeConfig(policy, actor.personality, provider=actor.provider, model=actor.model,
                                                 identity=identity)
        else:
            agents[side] = rc.AgentRuntimeConfig(policy, actor.personality, identity=identity)
    return rc.validate(rc.GameRuntimeConfig(black=agents["black"], white=agents["white"], level=entry["level"],
                                            seed=live.seed, mode="evaluation", view="research",
                                            time_limit=float(live.time_limit), save_replay=False,
                                            enhanced_gameplay=False))  # the research ran without them


def resimulate(match_dir: str | Path, work_dir: str | Path, *, verbose: bool = True) -> dict[str, Any]:
    """Re-runs one frozen research match (no provider) and returns {"verdict", "comparison", "bundle", ...}."""
    match_dir, work_dir = Path(match_dir).resolve(), Path(work_dir)
    for name in ("match_spec.json", "run.json", "telemetry_match01.jsonl"):
        if not (match_dir / name).is_file():
            raise HistoricalError(f"{match_dir} is not a research match directory (no {name}). Historical re-simulation "
                                  "needs the research archive; examples/replays/ holds re-simulated examples")
    try:  # the frozen research match process: present in the research archive only
        from workshop.scripts import confirmatory_trial as ct
    except ImportError as exc:
        raise HistoricalError(f"historical re-simulation needs the research archive ({exc.name} is not installed "
                              "here); examples/replays/ holds re-simulated examples") from exc
    from workshop.live import session as live_session
    from workshop.live.config import resolve_config
    from workshop.policy_emulator import stage4_level_select as ls
    from spy_vs_ai.runtime.decisions import DecisionLog, match_summary
    from spy_vs_ai.runtime.session import ACTOR_OF, engine_identity, spies_identity
    spec = json.loads((match_dir / "match_spec.json").read_text())
    run = json.loads((match_dir / "run.json").read_text())
    if not str(run.get("kind", "")).startswith(SUPPORTED_KIND_PREFIX):
        raise HistoricalError(f"{match_dir}: kind '{run.get('kind')}' - only matches of the confirmatory match process "
                              "(Stages 4C, 5B, 5C) can be re-simulated")
    entry = spec["entry"]
    historical = _jsonl(match_dir / "telemetry_match01.jsonl")
    arm, cand = entry["arm"], entry.get("candidate")
    provider_arm = arm != "r0"
    ls.select(entry["level"])
    if cand:
        from workshop.policy_emulator import stage5b_protocol as s5b
        if arm == "r0":
            s5b.install(cand)  # the frozen Stage-5B / 5C R0 binding (arm base + shadow R2.1), this process only
        else:
            live_session.LiveGame._build_hybrid_agent = s5b.make_builder(cand)  # as stage5c_trial.provider_arm_match
    config = resolve_config(spec["config"])
    answers = recorded_answers(historical)
    clients: dict[str, Any] = {}

    def factory(actor_spec, cfg):
        if provider_arm:
            clients[actor_spec.actor_id] = RecordedAnswerClient(actor_spec, answers.get(actor_spec.actor_id, []))
        else:
            from workshop.scripts.hybrid_router_replay import FakeLegalClient
            clients[actor_spec.actor_id] = FakeLegalClient(actor_spec)
        return clients[actor_spec.actor_id]

    base = ct._trial_runner_class()

    class RecordingRunner(base):
        def new_match(self) -> None:
            super().new_match()
            game = self.game
            self.recorder = EngineRecorder()
            self.recorder.install(game.state)
            inner = game._step

            def step(dt, _inner=inner, _rec=self.recorder, _game=game):
                _rec.begin()
                _inner(dt)
                _rec.end(len(_game.telemetry.events) if _game.telemetry is not None else 0)
            game._step = step  # instance attribute: the frozen class is untouched

    work_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    runner = RecordingRunner(config, work_dir, headless=True, max_decisions=ct.MAX_DECISIONS, max_wall_seconds=7200.0,
                             client_factory=factory)
    if arm == "hybrid":
        from workshop.policy_emulator.hybrid import HybridBudget
        runner.hybrid_budget = HybridBudget(config.max_llm_calls_per_spy, config.max_llm_calls_per_match, None)
    summary = runner.run()
    game = runner.game
    regenerated = list(game.telemetry.events)
    comparison = compare(historical, regenerated)
    comparison["stop_reason"] = summary.get("stop_reason")
    comparison["recorded_answers_left"] = {a: len(c.answers) for a, c in clients.items() if hasattr(c, "answers")}
    comparison["recorded_answers_missing"] = {a: c.exhausted for a, c in clients.items() if hasattr(c, "exhausted")}
    ok = (comparison["engine_events_identical"] and comparison["ranking_identical"]
          and not any(comparison["recorded_answers_missing"].values()) and not any(comparison["recorded_answers_left"].values()))
    verdict = "REPRODUCED" if ok else "DIVERGED"
    out: dict[str, Any] = {"verdict": verdict, "match_dir": str(match_dir), "arm": arm, "arm_label": entry.get("arm_label"),
                           "experiment": entry.get("experiment"), "match_id": entry.get("match_id"),
                           "comparison": comparison, "provider_calls": 0,
                           "fake_client_calls": {a: c.calls for a, c in clients.items()}}
    if not ok:
        return out
    cfg = runtime_config_for(entry, config)
    keep = ("surrogate_decision", "hybrid_decision", "llm_decision", "opponent_decision", "provider_call", "barrier_commit",
            "live_async_commit", "engine_event", "spy_fight", "match_result", "action_lifecycle")
    merged, index = merge_telemetry(regenerated, historical, provider_arm=provider_arm, keep=keep)
    agents = {}
    for side in rc.ACTORS:
        a = cfg.agent(side)
        agents[ACTOR_OF[side]] = {"policy": a.policy, "label": a.descriptor.label, "kind": a.descriptor.kind,
                                  "base": a.descriptor.base, "personality": a.personality, "provider": a.provider,
                                  "model": a.model}
    log = DecisionLog(agents)
    log.update(merged)
    match = match_summary(game, log)
    manifest = {"engine": engine_identity(), "execution": rc.execution(cfg), "spies": spies_identity(cfg, {}),
                "level_file": {"level": cfg.level, "path": ls.level_path(cfg.level), "sha256": ls.file_sha256(cfg.level)}}
    body = replay_bundle(runtime_config=cfg.public(), live_config=config.to_public_dict(), manifest=manifest,
                         recorder=runner.recorder, events=regenerated, decisions=[], summary=match, game=game, complete=True,
                         started_utc=started, finished_utc=_now())
    body["telemetry"], body["telemetry_index"] = merged, index
    body["kind"] = "historical research match, re-simulated without a provider (not a new experiment)"
    body["replay_label"] = f"historical {entry.get('experiment')} {entry.get('arm_label') or arm} {entry.get('match_id')}"
    body["historical"] = {k: out[k] for k in ("match_dir", "arm", "arm_label", "experiment", "match_id", "comparison")}
    out["bundle"] = body
    return out


__all__ = ["resimulate", "compare", "merge_telemetry", "RecordedAnswerClient", "recorded_answers", "policy_of", "EXAMPLES",
           "HistoricalError"]
