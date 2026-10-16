from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any


class TelemetryLogger:
    """Minimal JSONL telemetry logger for M5A decisions and episode summaries."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._handle = None
        if path:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8")

    def emit_event(self, event: dict[str, Any]) -> dict[str, Any]:
        sanitized = self._sanitize(event)
        self.events.append(sanitized)
        if self._handle is not None:
            self._handle.write(json.dumps(sanitized, sort_keys=True) + "\n")
            self._handle.flush()
        return sanitized

    def _sanitize(self, event: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in event.items():
            if isinstance(value, str) and any(token in key.lower() for token in ("key", "secret", "token", "authorization", "password")):
                result[key] = "[redacted]"
            else:
                result[key] = value
        return result

    def emit_episode_start(
        self,
        *,
        episode_id: str,
        seed: int,
        white_controller: str,
        black_controller: str,
        provider: str,
        model: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        event = {
            "event_type": "episode_start",
            "episode_id": episode_id,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "white_controller": white_controller,
            "black_controller": black_controller,
            "provider": provider,
            "model": model,
            "prompt_version": kwargs.get("prompt_version", "m5a-agent-v1"),
            "observation_schema_version": kwargs.get("observation_schema_version", "obs-v1"),
            "action_schema_version": kwargs.get("action_schema_version", "action-v1"),
            "game_version": kwargs.get("game_version", "spy_vs_ai"),
            "map_id": kwargs.get("map_id", "level_01"),
            "generation_config": kwargs.get("generation_config", {}),
        }
        return self.emit_event(event)

    def emit_llm_decision(
        self,
        *,
        episode_id: str,
        actor_id: str,
        tick: int,
        game_time: float,
        provider: str,
        model: str,
        prompt_version: str,
        observation_schema_version: str,
        action_schema_version: str,
        observation_hash: str,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        raw_response: dict[str, Any],
        validated_action: dict[str, Any],
        valid: bool,
        repair_attempted: bool,
        fallback_used: bool,
        action_source: str = "llm",
        decision_id: str | None = None,
        latency_ms: float | None = None,
        time_to_first_token_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        finish_reason: str | None = None,
        provider_request_id: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        observation_size_chars: int | None = None,
        model_payload_size_chars: int | None = None,
        legal_actions_count: int | None = None,
        memory_schema_version: str | None = None,
        memory_size_chars: int | None = None,
        visited_rooms_count: int | None = None,
        known_requirements_count: int | None = None,
        known_empty_objects_count: int | None = None,
        recent_events_count: int | None = None,
        recent_failures_count: int | None = None,
        recent_decisions_count: int | None = None,
        recent_repetitions_count: int | None = None,
        agent_runtime: str | None = None,
        agent_runtime_version: str | None = None,
        # M5B.1: strict-validity attribution, kept distinct from the
        # existing `valid`/`repair_attempted`/`fallback_used`/`action_source`
        # fields above rather than replacing them - see
        # M5B.1_STRICT_MODEL_OUTPUT_VALIDATION.md sections 8, 11, 14.
        initial_model_output_valid: bool | None = None,
        initial_error_type: str | None = None,
        repair_raw_response: dict[str, Any] | None = None,
        repair_valid: bool | None = None,
        repair_error_type: str | None = None,
        final_model_action_valid: bool | None = None,
        fallback_action: dict[str, Any] | None = None,
        executed_action: dict[str, Any] | None = None,
        executed_action_valid: bool | None = None,
        # M5C.1: the SDK backing the runtime, distinct from agent_runtime
        # (our own adapter name/version) and provider/model (the backend the
        # SDK talks to) - never label a runtime simply as its provider.
        agent_sdk: str | None = None,
        agent_sdk_version: str | None = None,
        # M5C.1: raw, runtime-specific facts that don't fit the common
        # schema (e.g. Strands tool_name/tool_arguments/cycle_count) - never
        # invented metrics, just what that runtime actually exposes. None
        # for runtimes (like Native) that have nothing extra to add.
        runtime_metadata: dict[str, Any] | None = None,
        # M5D.5: recovery-specific fields (spec section 18) - kept distinct
        # from the M5B.1 validity fields above rather than overloading
        # them, since these describe the *repair contract itself* (what it
        # was given, what it produced), not just pass/fail.
        initial_proposal: dict[str, Any] | None = None,
        repair_context_version: str | None = None,
        repair_legal_action_count: int | None = None,
        repair_near_object_id: str | None = None,
        recovery_outcome: str | None = None,
        # M5D.7: which action-representation variant produced this
        # decision (spec section 12) - "native_current-v1" /
        # "native_tool_like-v1" / "strands_tool-v1" - distinct from
        # agent_runtime (native/strands, the orchestration layer) since
        # M5D.7 introduces two representations under the same Native runtime.
        representation_variant: str | None = None,
        # M5E.1: decision-level diagnostics (spec sections 5-6) - kept
        # optional/None-default so every pre-M5E.1 call site (every
        # existing test and live run) is unaffected.
        decision_number: int | None = None,
        condition: str | None = None,
        serialized_legal_actions_chars: int | None = None,
        repair_context_chars: int | None = None,
        tool_schema_chars: int | None = None,
        total_tokens: int | None = None,
        # M5F.1 spec section 15: which personality profile (if any)
        # produced this decision - a purely diagnostic tag, never read by
        # any decision/validation logic, following the exact same pattern
        # `condition` (M5E.1) already established.
        personality_profile: str | None = None,
        # M5F.2.3 spec section 10: the exact availability facts shown to
        # the model this decision (only meaningful under the bounded
        # rule) - explicit named params, not left to `**kwargs` (M5F.1's
        # own discovery: unnamed kwargs are silently dropped from the
        # emitted event, never re-derivable after the fact otherwise,
        # since the compact model-facing payload itself isn't stored here
        # - only the raw pre-injection `observation` is).
        communication_available: bool | None = None,
        communication_next_available_in_decisions: int | None = None,
        # M6B.2 spec section 11: a running per-episode count of how many
        # times this exact (action, target, trap_type) invalid-proposal
        # signature has now been seen - diagnostic only (spec section 11
        # explicitly: "do not automatically suppress the action unless
        # existing policy supports it"), never behavior-shaping. `None`
        # for a valid decision (the field only means something for an
        # invalid final proposal).
        same_invalid_action_signature_count: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        event = {
            "event_type": "llm_decision",
            # M5C: which agent runtime produced this decision - a dimension
            # separate from provider/model (see src/agents/base.py).
            "agent_runtime": agent_runtime,
            "agent_runtime_version": agent_runtime_version,
            "agent_sdk": agent_sdk,
            "agent_sdk_version": agent_sdk_version,
            "runtime_metadata": runtime_metadata,
            # M5B: use the caller's decision_id (the same id recorded in
            # AgentMemory) so telemetry and memory correlate on one id,
            # instead of minting an unrelated random one here.
            "decision_id": decision_id or str(uuid.uuid4()),
            "episode_id": episode_id,
            "actor_id": actor_id,
            "tick": tick,
            "game_time": game_time,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "observation_schema_version": observation_schema_version,
            "action_schema_version": action_schema_version,
            "observation_hash": observation_hash,
            "observation": observation,
            "legal_actions": legal_actions,
            "raw_response": raw_response,
            "validated_action": validated_action,
            "valid": valid,
            "repair_attempted": repair_attempted,
            "fallback_used": fallback_used,
            "action_source": action_source,
            "latency_ms": latency_ms,
            "time_to_first_token_ms": time_to_first_token_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "finish_reason": finish_reason,
            "provider_request_id": provider_request_id,
            "error_type": error_type,
            "error_message": error_message,
            "observation_size_chars": observation_size_chars,
            "model_payload_size_chars": model_payload_size_chars,
            "legal_actions_count": legal_actions_count,
            # M5B: memory telemetry (see AgentMemory.to_telemetry_dict).
            "memory_schema_version": memory_schema_version,
            "memory_size_chars": memory_size_chars,
            "visited_rooms_count": visited_rooms_count,
            "known_requirements_count": known_requirements_count,
            "known_empty_objects_count": known_empty_objects_count,
            "recent_events_count": recent_events_count,
            "recent_failures_count": recent_failures_count,
            "recent_decisions_count": recent_decisions_count,
            "recent_repetitions_count": recent_repetitions_count,
            # M5B.1: strict validity attribution - never derive model
            # validity from the fact that the engine eventually received a
            # valid (possibly fallback) action.
            "initial_model_output_valid": initial_model_output_valid,
            "initial_error_type": initial_error_type,
            "repair_raw_response": repair_raw_response,
            "repair_valid": repair_valid,
            "repair_error_type": repair_error_type,
            "final_model_action_valid": final_model_action_valid,
            "fallback_action": fallback_action,
            "executed_action": executed_action,
            "executed_action_valid": executed_action_valid,
            # M5D.5: recovery telemetry (spec section 18).
            "initial_proposal": initial_proposal,
            "repair_context_version": repair_context_version,
            "repair_legal_action_count": repair_legal_action_count,
            "repair_near_object_id": repair_near_object_id,
            "recovery_outcome": recovery_outcome,
            "representation_variant": representation_variant,
            # M5E.1 sections 5-6: decision-number/condition tagging plus the
            # context-size fields not already covered by
            # observation_size_chars/model_payload_size_chars above.
            "decision_number": decision_number,
            "condition": condition,
            "personality_profile": personality_profile,
            "communication_available": communication_available,
            "communication_next_available_in_decisions": communication_next_available_in_decisions,
            "serialized_legal_actions_chars": serialized_legal_actions_chars,
            "repair_context_chars": repair_context_chars,
            "tool_schema_chars": tool_schema_chars,
            "total_tokens": total_tokens,
            "same_invalid_action_signature_count": same_invalid_action_signature_count,
        }
        return self.emit_event(event)

    def emit_provider_call(
        self,
        *,
        episode_id: str,
        actor_id: str,
        decision_id: str,
        decision_number: int,
        call_index: int,
        call_type: str,
        runtime: str,
        provider: str,
        model: str,
        start_time: str,
        end_time: str,
        latency_ms: float,
        success: bool,
        condition: str | None = None,
        provider_error_type: str | None = None,
        provider_error_message_normalized: str | None = None,
        timeout: bool = False,
        http_status: int | None = None,
        retry_count: int = 0,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """M5E.1 spec section 9: one record per real model call - initial or
        repair - separate from `emit_llm_decision`'s one-per-decision
        granularity, so per-call latency/error/timeout facts are never
        summed/blended together the way `emit_llm_decision`'s aggregate
        `latency_ms`/`error_type` already are (kept there unchanged for
        backward compatibility; this is additive, not a replacement)."""
        event = {
            "event_type": "provider_call",
            "episode_id": episode_id,
            "actor_id": actor_id,
            "decision_id": decision_id,
            "decision_number": decision_number,
            "call_index": call_index,
            "call_type": call_type,
            "runtime": runtime,
            "provider": provider,
            "model": model,
            "condition": condition,
            "start_time": start_time,
            "end_time": end_time,
            "latency_ms": latency_ms,
            "success": success,
            "provider_error_type": provider_error_type,
            "provider_error_message_normalized": provider_error_message_normalized,
            "timeout": timeout,
            "http_status": http_status,
            "retry_count": retry_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        return self.emit_event(event)

    def emit_opponent_decision(
        self,
        *,
        episode_id: str,
        actor_id: str,
        decision_id: str,
        controller: str,
        observation_hash: str,
        selected_action: str,
        target_id: str | None,
        reason_code: str,
        game_time: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """M5E spec section 27: lightweight, actor-tagged decision facts for
        a deterministic scripted opponent - no LLM-style token/latency
        fields (there's no model call to measure), and no free-form
        `reason` text - `reason_code` is one of a small closed vocabulary
        (spec section 28) naming which policy branch fired, so opponent
        behavior stays auditable without re-deriving it from raw actions.
        """
        event = {
            "event_type": "opponent_decision",
            "episode_id": episode_id,
            "actor_id": actor_id,
            "decision_id": decision_id,
            "controller": controller,
            "observation_hash": observation_hash,
            "selected_action": selected_action,
            "target_id": target_id,
            "reason_code": reason_code,
            "game_time": game_time,
        }
        return self.emit_event(event)

    def emit_action_lifecycle(
        self,
        *,
        episode_id: str,
        actor_id: str,
        decision_id: str,
        action_id: str,
        action: str,
        target_id: str | None,
        action_status: str,
        action_terminal_reason: str,
        action_started_game_time: float,
        action_completed_game_time: float,
        action_simulation_ticks: int,
        agent_runtime: str | None = None,
        agent_runtime_version: str | None = None,
        navigation: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """M5D.2: one event per accepted action reaching a lifecycle
        boundary (spec sections 4-5, 19) - a *different* granularity from
        emit_llm_decision (one per strategic decision). A single decision's
        action may complete many ticks - and, for MOVE, many seconds -
        after its own decision event was already emitted, so this is
        always a separate event, correlated by decision_id and action_id
        (spec section 12), never folded into the original decision event.
        """
        event = {
            "event_type": "action_lifecycle",
            "episode_id": episode_id,
            "actor_id": actor_id,
            "decision_id": decision_id,
            "action_id": action_id,
            "action": action,
            "target_id": target_id,
            "action_status": action_status,
            "action_terminal_reason": action_terminal_reason,
            "action_started_game_time": action_started_game_time,
            "action_completed_game_time": action_completed_game_time,
            "action_duration_game_time": action_completed_game_time - action_started_game_time,
            "action_simulation_ticks": action_simulation_ticks,
            "agent_runtime": agent_runtime,
            "agent_runtime_version": agent_runtime_version,
        }
        if navigation is not None:
            # Bugfix (navigation/exploration §29): MOVE instrumentation only -
            # absent for other actions, so older consumers see the same keys.
            event["navigation"] = navigation
        return self.emit_event(event)

    def emit_episode_end(
        self,
        *,
        episode_id: str,
        escaped: bool,
        success: bool,
        llm_calls: int,
        valid_actions: int,
        invalid_actions: int,
        repair_attempts: int,
        fallbacks: int,
        mean_latency_ms: float,
        min_latency_ms: float | None = None,
        max_latency_ms: float | None = None,
        valid_action_rate: float | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        event = {
            "event_type": "episode_end",
            "episode_id": episode_id,
            "escaped": escaped,
            "success": success,
            "llm_calls": llm_calls,
            "valid_actions": valid_actions,
            "invalid_actions": invalid_actions,
            "repair_attempts": repair_attempts,
            "fallbacks": fallbacks,
            "mean_latency_ms": mean_latency_ms,
            "min_latency_ms": min_latency_ms,
            "max_latency_ms": max_latency_ms,
            "valid_action_rate": valid_action_rate,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "seed": kwargs.get("seed"),
        }
        return self.emit_event(event)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
