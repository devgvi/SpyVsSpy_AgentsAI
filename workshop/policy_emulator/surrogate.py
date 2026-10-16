"""3F - SurrogateController: a trained policy artifact playing one spy, no provider call.

The existing controller contract, unchanged: the game calls
`get_action(observation) -> GameAction` every frame, exactly as it calls the
LLM controller and the scripted bot. Inside:

    Observation
      -> the live payload (serialize_observation + annotate_live_payload: what an LLM would see)
      -> PolicyFeatureExtractorV1
      -> the loaded policy artifact (B0 frequency-v1, B1 table-v1, B2 logistic-v1 or B3 tree-v1)
      -> legal-action masking (+ actions the surrogate cannot carry out)
      -> seeded action sample -> target policy -> trap type / safe code
      -> the shared ActionExecutor (the LLM controller's multi-frame MOVE / approach /
         avoid semantics, lifecycle telemetry and memory bookkeeping)
      -> GameAction

It decides only when idle, like the LLM controller: after its last action
ended and either the observation changed (time excluded) or it has been idle
for IDLE_REDECIDE_SECONDS of game time; never more often than every
DECISION_INTERVAL_SECONDS of game time. It never "thinks" (no background
thread), so it never freezes the clock or holds the evaluation barrier.

Randomness (milestone 3): `policy_rng_seed` (default: the environment seed)
controls every stochastic choice. Decision k of an actor draws from its own
generator, seeded "surrogate:<policy_rng_seed>:<actor>:<policy>:<k>", so two
emulators playing the same identity with the same policy RNG seed share their
random numbers decision by decision (common random numbers); the emulator is
not part of the seed. `environment_seed` (the game seed) is recorded beside it.

Every decision is one `surrogate_decision` telemetry event: probabilities,
masking, the sampled action and target, the model's explanation (B1 bucket /
backoff / samples, B2 top features, B3 leaf / path / leaf samples), its
confidence (the largest masked probability) and, separately, the coverage
of the state in the policy's training data (coverage-v1).
"""
from __future__ import annotations

import hashlib
import json
import random
import uuid
from typing import Any

from src import constants
from src.actions import ActionType, GameAction
from src.agents.action_lifecycle import ActionExecutor
from src.agents.live_async import IDLE_REDECIDE_SECONDS
from src.llm.agent_memory import AgentMemory, AgentMemoryTracker
from src.llm_controller import serialize_observation
from src.mission_time import annotate_live_payload
from workshop.policy_emulator import schema
from workshop.policy_emulator.features import PolicyFeatureExtractorV1
from workshop.policy_emulator.artifacts import LoadedPolicy
from workshop.policy_emulator.coverage import COVERAGE_VERSION
from workshop.policy_emulator.policies import legal_action_set

DECISION_INTERVAL_SECONDS = 0.5  # LLMController._decision_interval_seconds
SURROGATE_EXECUTION_MODE = "surrogate"  # not "evaluation": the response barrier is between two thinking models


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in payload.items() if k != "time"}, sort_keys=True,
                                     default=str).encode()).hexdigest()


def _sample(probs: dict[str, float], rng: random.Random, order: list) -> Any:
    """Inverse-CDF sample in a fixed key order (reproducible for a fixed seed)."""
    keys = [k for k in order if probs.get(k, 0) > 0]
    if not keys:
        return None
    u, acc = rng.random() * sum(probs[k] for k in keys), 0.0
    for k in keys:
        acc += probs[k]
        if u < acc:
            return k
    return keys[-1]


class SurrogateController:
    controller_kind = "surrogate"
    runtime_name = schema.SURROGATE_RUNTIME
    runtime_version = schema.SURROGATE_RUNTIME_VERSION
    agent_runtime = schema.SURROGATE_RUNTIME
    agent_runtime_version = schema.SURROGATE_RUNTIME_VERSION
    execution_mode = SURROGATE_EXECUTION_MODE
    decision_event_type = "surrogate_decision"

    def __init__(self, actor_id: str, policy: LoadedPolicy, *, telemetry: Any = None, episode_id: str = "",
                 seed: int | str | None = None, dataset: str = "phase-a-v2", environment_seed: int | None = None) -> None:
        self.actor_id = actor_id
        self.policy = policy
        self.telemetry = telemetry
        self.episode_id = episode_id
        self.dataset = dataset
        self.policy_rng_seed = seed if seed is not None else environment_seed
        self.environment_seed = environment_seed
        self.rng_seed = f"surrogate:{self.policy_rng_seed}:{actor_id}:{policy.policy_key}"  # + ":<decision index>"
        self._rng = random.Random(f"{self.rng_seed}:0")
        self._executor = ActionExecutor()
        self._memory_tracker = AgentMemoryTracker(actor_id)
        self._extractor = PolicyFeatureExtractorV1()
        self._recent: list[str] = []
        self._last_digest: str | None = None
        self._last_decision_clock: float | None = None  # game clock (-mission time remaining) of the last decision
        self._last_decision_id: str | None = None
        self.provider_name = SURROGATE_EXECUTION_MODE
        self.model_name = policy.policy_key
        self.status = "IDLE"
        self.last_action = GameAction(ActionType.WAIT)
        self.last_reason = "No surrogate decision yet."
        self.last_decision: dict[str, Any] | None = None
        self.decisions_requested = 0
        self.decisions_completed = 0
        self.fallback_count = 0
        self.repair_count = 0
        self.provider_failure_count = 0
        self.masking_changed_top = 0
        self.target_fallbacks = 0
        self.unsupported_masked = 0
        self._thinking_in_progress = False
        self.response_ready_uncommitted = False
        self.last_call_committed_ready_response = False

    @property
    def memory(self) -> AgentMemory:
        return self._memory_tracker.memory

    # -- the controller contract ---------------------------------------------------------------

    def get_action(self, observation) -> GameAction:
        if (observation.mission_complete or observation.stunned or observation.busy
                or getattr(observation, "time_expired", False)
                or getattr(observation, "lifecycle", "ACTIVE") in constants.TERMINAL_LIFECYCLES):
            self.status = "IDLE"
            if self._executor.in_progress:
                _, boundary = self._executor.continue_(observation)
                if boundary is not None:
                    self._finish(observation, boundary)
            self.last_action = GameAction(ActionType.WAIT)
            return self.last_action
        self._memory_tracker.update_from_observation(observation)
        if self._executor.in_progress:
            action, boundary = self._executor.continue_(observation)
            self.last_action = action
            if boundary is None:
                self.status = "EXECUTING"
                return action
            self._finish(observation, boundary)
        payload = serialize_observation(observation, live_state=True, live_workshop=True)
        digest = _digest(payload)
        clock = -observation.mission_time_remaining
        if self._last_decision_clock is not None:
            since = clock - self._last_decision_clock
            if since < DECISION_INTERVAL_SECONDS or (digest == self._last_digest and since < IDLE_REDECIDE_SECONDS):
                self.status = "IDLE"
                self.last_action = GameAction(ActionType.WAIT)
                return self.last_action
        self._last_digest, self._last_decision_clock = digest, clock
        return self._decide(observation, annotate_live_payload(observation, payload))

    # -- one decision --------------------------------------------------------------------------

    def decision_rng(self, index: int) -> random.Random:
        """The generator of decision `index` (1-based): independent of earlier decisions and of the emulator."""
        return random.Random(f"{self.rng_seed}:{index}")

    def decide_payload(self, payload: dict[str, Any], rng: random.Random | None = None) -> dict[str, Any]:
        """The surrogate's choice for one annotated payload, without executing it
        (also used by tests and by logged-state replay of the live path)."""
        rng = rng or self._rng
        features = self._extractor.extract(payload, list(reversed(self._recent))[:3])
        legal = legal_action_set(payload)
        code = schema.code_from_visible_puzzle(payload)
        unsupported = ["enter_safe_code"] if "enter_safe_code" in legal and code is None else []
        executable = [a for a in legal if a not in unsupported] or ["wait"]
        prediction = self.policy.action_policy.predict(features, executable)
        action = _sample(prediction.probs, rng, list(schema.ACTIONS)) or "wait"
        choice = self.policy.target_policy.choose(action, payload, features)
        target_order = [t for t in schema.legal_targets(payload).get(action, [])]
        target = _sample(choice.distribution, rng, target_order) if choice.distribution else None
        trap_type = self.policy.target_policy.trap_type(payload, rng) if action == "place_trap" else None
        legal_pairs = {(a.get("action"), a.get("target_id")) for a in payload.get("legal_actions") or []}
        valid = (action, target) in legal_pairs and not (action == "place_trap" and trap_type is None)
        coverage = self.policy.coverage.assess(features, legal) if self.policy.coverage is not None else None
        return {"features": features, "prediction": prediction, "sampled": action, "sampled_target": target,
                "action": action if valid else "wait",
                "target": target if valid else None, "trap_type": trap_type if valid else None,
                "code": code if action == "enter_safe_code" and valid else None, "choice": choice,
                "unsupported": unsupported, "valid": valid, "legal": legal, "coverage": coverage}

    def _decide(self, observation, payload: dict[str, Any]) -> GameAction:
        return self._commit(observation, payload, self._propose(payload))

    def _propose(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One decision point: the masked distribution and the sampled proposal (nothing committed yet)."""
        self.decisions_requested += 1
        return self.decide_payload(payload, self.decision_rng(self.decisions_requested))

    def _commit(self, observation, payload: dict[str, Any], d: dict[str, Any],
                extra: dict[str, Any] | None = None) -> GameAction:
        """Start the proposal on the executor and record it (the surrogate path; the hybrid
        controller's non-escalated decisions go through exactly this code)."""
        prediction, choice = d["prediction"], d["choice"]
        decision_id = str(uuid.uuid4())
        self._last_decision_id = decision_id
        if not d["valid"]:
            self.fallback_count += 1
        self.masking_changed_top += int(prediction.masking_changed_top)
        self.target_fallbacks += int(choice.fallback)
        self.unsupported_masked += int(bool(d["unsupported"]))
        game_time = observation.mission_time_remaining
        mission_elapsed = round(float(getattr(observation, "mission_time_limit", 0.0) or 0.0) - game_time, 3)
        action_dict = {"action": d["action"], "target_id": d["target"], "trap_type": d["trap_type"]}
        game_action, boundary, _ = self._executor.start(observation, decision_id, d["action"], d["target"], d["trap_type"],
                                                        game_time, code=d["code"])
        source = "surrogate" if d["valid"] else "fallback"
        self.memory.record_decision(game_time=game_time, decision_id=decision_id, selected_action=d["action"],
                                    target_id=d["target"], valid=d["valid"], action_source=source)
        self._memory_tracker.set_pending_decision(decision_id, d["action"], d["target"], observation.nearby_object_id)
        if boundary is not None:
            self._finish(observation, boundary)
        self._recent.append(d["action"])
        self.decisions_completed += 1
        confidence = prediction.probs.get(d["action"], 0.0)
        coverage = d["coverage"] or {}
        self.last_reason = (f"{self.policy.emulator}: {d['action']} p={confidence:.2f}, model confidence "
                            f"{prediction.confidence:.2f}, coverage {coverage.get('coverage', '-')}")
        self.status = "EXECUTING" if d["action"] != "wait" else "IDLE"
        self.last_action = game_action
        overlay = {"policy": self.policy.policy_key, "model_display": self.policy.metadata.get("model_display"),
                   "personality": self.policy.metadata.get("personality"), "emulator": self.policy.emulator,
                   "confidence": round(prediction.confidence, 3), "chosen_p": round(confidence, 3),
                   "coverage": coverage.get("coverage"), "coverage_score": coverage.get("score"),
                   "coverage_reasons": (coverage.get("reasons") or [])[:3],
                   "target_policy": self.policy.target_version, "explanation": prediction.explanation,
                   "choice": schema.category(d["action"], d["target"], payload),
                   "action": d["action"], "target": d["target"],
                   "top": sorted(((a, round(p, 3)) for a, p in prediction.probs.items()), key=lambda x: -x[1])[:4],
                   "bucket": prediction.matched_bucket, "backoff_level": prediction.backoff_level,
                   "training_samples": prediction.sample_count}
        overlay.update((extra or {}).get("overlay") or {})
        self.last_decision = overlay
        if self.telemetry is not None:
            self.telemetry.emit_event({
                "event_type": self.decision_event_type, "episode_id": self.episode_id, "actor_id": self.actor_id,
                "controller_mode": self.controller_kind,
                "decision_id": decision_id, "decision_number": self.decisions_requested,
                "decision_index": self.decisions_requested,
                "mission_time_remaining": round(game_time, 3), "mission_elapsed": mission_elapsed,
                # the engine telemetry sink redacts any top-level string field whose name contains
                # "key" (or token / secret / password / authorization), so the identity is "policy"
                "policy": self.policy.policy_key, "policy_id": self.policy.metadata.get("policy_id"),
                "policy_rng_seed": self.policy_rng_seed, "environment_seed": self.environment_seed,
                "emulator": self.policy.emulator, "target_emulator": self.policy.target_version,
                "training_dataset": self.dataset, "training_dataset_hash": self.policy.metadata.get("training_dataset_hash"),
                "feature_schema_version": self._extractor.version, "runtime": self.runtime_name,
                "runtime_version": self.runtime_version, "rng_seed": self.rng_seed,
                "probs": {a: round(p, 4) for a, p in prediction.probs.items()},
                "raw_top_action": prediction.raw_top_action, "top_action": prediction.top_action,
                "masking_changed_top": prediction.masking_changed_top, "matched_bucket": prediction.matched_bucket,
                "backoff_level": prediction.backoff_level, "sample_count": prediction.sample_count,
                "unsupported_masked": d["unsupported"], "legal_actions": d["legal"],
                "intended_action": d["sampled"], "intended_target": d["sampled_target"],
                "sampled_action": d["action"], "confidence": round(prediction.confidence, 4),
                "chosen_probability": round(confidence, 4), "coverage": coverage or None,
                "coverage_label": coverage.get("coverage"), "coverage_score": coverage.get("score"),
                "coverage_reasons": coverage.get("reasons") or [],
                "coverage_version": COVERAGE_VERSION if self.policy.coverage is not None else None,
                "explanation": prediction.explanation, "target_policy": self.policy.target_version,
                "target_id": d["target"], "target_source": choice.source, "target_fallback": choice.fallback,
                "target_class_probs": {c: round(p, 4) for c, p in choice.target_class_probs.items()},
                "trap_type": d["trap_type"], "code_from_visible_puzzle": d["code"] is not None,
                "valid": d["valid"], "action_source": source,
                "executed_action": action_dict, "category": overlay["choice"],
                "features": d["features"], "surrogate_overlay": overlay,
                **{k: v for k, v in (extra or {}).items() if k != "overlay"},
            })
        return game_action

    def _finish(self, observation, boundary: str) -> None:
        self._executor.finish(memory=self.memory, telemetry=self.telemetry, episode_id=self.episode_id,
                              actor_id=self.actor_id, agent_runtime=self.runtime_name,
                              agent_runtime_version=self.runtime_version, observation=observation, boundary=boundary)

    def provenance(self) -> dict[str, Any]:
        """What reproduces this surrogate (milestone 3, spec 32)."""
        return {"policy": self.policy.policy_key, "emulator": self.policy.emulator,
                "emulator_artifact_hash": self.policy.metadata.get("content_hash"),
                "target_emulator": self.policy.target_version,
                "target_artifact_hash": (self.policy.target_metadata or {}).get("content_hash"),
                "coverage_version": COVERAGE_VERSION if self.policy.coverage is not None else None,
                "coverage_artifact_hash": self.policy.coverage_hash,
                "feature_schema_version": self._extractor.version,
                "training_dataset_hash": self.policy.metadata.get("training_dataset_hash"),
                "policy_rng_seed": self.policy_rng_seed, "environment_seed": self.environment_seed,
                "rng_seed": self.rng_seed, "runtime_version": self.runtime_version}

    def stats(self) -> dict[str, Any]:
        return {"policy_key": self.policy.policy_key, "emulator": self.policy.emulator, "provenance": self.provenance(),
                "decisions": self.decisions_completed, "fallbacks": self.fallback_count,
                "masking_changed_top": self.masking_changed_top, "target_fallbacks": self.target_fallbacks,
                "decisions_with_unsupported_action_masked": self.unsupported_masked,
                "actions_started": self._executor.actions_started, "actions_completed": self._executor.actions_completed,
                "actions_failed": self._executor.actions_failed, "actions_interrupted": self._executor.actions_interrupted,
                "training_dataset_hash": self.policy.metadata.get("training_dataset_hash"), "rng_seed": self.rng_seed}
