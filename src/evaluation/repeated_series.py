"""M5F.3C: runs a repeated-episode series between the same two stable
actor identities (spec sections 13-14), extracting factual social events
per episode and - under MEMORY_ON - persisting them into each actor's own
`CrossEpisodeSocialMemory`, injected into the NEXT episode via the
existing `model_payload_transform` hook. `GameState` resets fully between
episodes (spec section 3); only explicitly permitted facts carry forward.

Epistemic scope note (documented, not hidden): follow-through facts
("did the opponent move toward the target they mentioned") are computed
from full Engine-recorded telemetry (both actors' own actions), matching
this project's own established `cooperation_follow_through`/
`threat_follow_through` functions already used for within-episode
reporting - a real, bounded simplification. A fully strict implementation
would additionally gate these on the remembering actor's own in-game
visibility of the opponent's movements; this milestone's evaluation-
harness-level memory does not yet model that. Spec section 4's own
example list sanctions "opponent followed through on claimed target" as
an allowed persistent fact without requiring visibility-gating language,
so this is treated as within scope, not a leak of hidden state (the
forbidden list in spec section 5 is about undetected traps/hidden
inventory/private rationale/unobserved routes, none of which this
touches).
"""

from __future__ import annotations

import json
import os
from typing import Any

from src import constants
from src.evaluation.cross_episode_memory import (
    CrossEpisodeSocialMemory,
    SocialFact,
    build_memory_from_episode_facts,
    model_payload_transform_for,
)
from src.evaluation.decision_trace import build_progress_snapshot
from src.evaluation.duel_runner import run_duel_episode
from src.evaluation.free_social import (
    COORDINATION_PROPOSAL,
    FACTUAL_CLAIM,
    SAFETY_WARNING,
    classify_warning_origin,
    mentions_danger_claim,
    mentions_safe_claim,
    resolve_claim_outcome,
)
from src.evaluation.free_social_matchup import analyze_social_on_episode, build_initial_traps

MESSAGE_COOLDOWN = 3
MAX_DECISIONS_PER_ACTOR = 40


def _load_decisions_and_lifecycle(telemetry_path: str, episode_id: str) -> tuple[list[dict], list[dict]]:
    decisions, lifecycle = [], []
    if not os.path.exists(telemetry_path):
        return decisions, lifecycle
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("episode_id") != episode_id:
                continue
            if e.get("event_type") == "llm_decision":
                decisions.append(e)
            elif e.get("event_type") == "action_lifecycle":
                lifecycle.append(e)
    return decisions, lifecycle


def _trap_status_history_for(decisions: list[dict], actor_id: str, target_id: str, after_game_time: float | None) -> list[str | None]:
    """The remembering actor's own real, chronological `trap_status`
    readings for `target_id`, strictly after `after_game_time` - spec
    section 28's resolution input. `None`/"unknown" entries (never
    verified) are included as-is; `resolve_claim_outcome` itself filters
    to only the resolving values."""
    rows = sorted(
        [d for d in decisions if d.get("actor_id") == actor_id and d.get("game_time") is not None and (after_game_time is None or d["game_time"] > after_game_time)],
        key=lambda d: d["game_time"],
    )
    statuses = []
    for d in rows:
        room_objects = ((d.get("observation") or {}).get("room") or {}).get("objects") or []
        match = next((o for o in room_objects if o.get("id") == target_id), None)
        statuses.append(match.get("trap_status") if match else None)
    return statuses


def extract_episode_social_facts(
    *, episode_number: int, decisions: list[dict], lifecycle: list[dict], episode_id: str, seed: int,
    trap_owner_snapshots: dict[int, dict[str, str | None]], black_id: str, white_id: str,
) -> dict[str, list[SocialFact]]:
    """Returns `{actor_id: [SocialFact, ...]}` - one list per actor,
    containing only facts THAT actor could legitimately have learned
    (spec section 4): every message it received, plus outcome/consistency
    facts derived from ITS OWN subsequent observations, never the
    opponent's hidden state."""
    analysis = analyze_social_on_episode(decisions, lifecycle, episode_id, seed)
    facts: dict[str, list[SocialFact]] = {black_id: [], white_id: []}

    for m in analysis["messages"]:
        sender, recipient = m["sender_actor_id"], m["recipient_actor_id"]
        category, text, target = m["category"], m["message_text"], m["referenced_target_id"]
        decision_number = None  # messages dict doesn't carry decision_number; game_time is the join key
        game_time = m["game_time"]

        # Every message the recipient actually received is always a
        # legitimate fact for the recipient to remember (spec section 4:
        # "opponent sent message X").
        label = {
            COORDINATION_PROPOSAL: "proposed",
            SAFETY_WARNING: "sent a warning/disclosure",
            FACTUAL_CLAIM: "made a claim",
        }.get(category, "sent a message")
        facts[recipient].append(SocialFact(
            episode_number=episode_number, fact_type="MESSAGE_SENT",
            text=f'{sender} {label}: "{text}"',
            detail={"sender": sender, "recipient": recipient, "category": category, "text": text, "target": target},
        ))

        if category == SAFETY_WARNING and target is not None:
            owner = (trap_owner_snapshots.get(_nearest_snapshot_key(trap_owner_snapshots, game_time)) or {}).get(target)
            origin = classify_warning_origin(trap_owner_id=owner, sender_id=sender, recipient_id=recipient)
            facts[recipient].append(SocialFact(
                episode_number=episode_number, fact_type="WARNING_ORIGIN",
                text=f"{sender}'s warning about {target} was later classified as {origin} (owner at the time: {owner or 'none recorded'}).",
                detail={"sender": sender, "target": target, "origin": origin, "trap_owner": owner},
            ))

        if category in (FACTUAL_CLAIM, SAFETY_WARNING) and target is not None:
            claims_safe = mentions_safe_claim(text)
            claims_danger = mentions_danger_claim(text)
            later_statuses = _trap_status_history_for(decisions, recipient, target, game_time)
            outcome = resolve_claim_outcome(message_category=category, message_claims_safe=claims_safe, message_claims_danger=claims_danger, later_trap_statuses=later_statuses)
            facts[recipient].append(SocialFact(
                episode_number=episode_number, fact_type="CLAIM_OUTCOME",
                text=f"{sender}'s claim about {target} was {outcome.replace('_', ' ').lower()}.",
                detail={"sender": sender, "target": target, "outcome": outcome},
            ))

    for c in analysis["cooperation_events"]:
        sender, recipient = c["sender_actor_id"], c["recipient_actor_id"]
        if c.get("sender_pursued_claimed_target") is not None:
            verb = "moved toward" if c["sender_pursued_claimed_target"] else "did not move toward"
            facts[recipient].append(SocialFact(
                episode_number=episode_number, fact_type="FOLLOW_THROUGH",
                text=f"{sender} then {verb} the target it proposed for itself.",
                detail={"sender": sender, "sender_pursued_claimed_target": c["sender_pursued_claimed_target"]},
            ))

    # Episode-outcome facts are appended by the caller (`run_series`),
    # which has direct access to `DuelResult.black_game_facts`/
    # `white_game_facts` - `analyze_social_on_episode` doesn't compute
    # those itself.
    return facts


def _nearest_snapshot_key(snapshots: dict[int, Any], game_time: float | None) -> int:
    """Trap-ownership snapshots are keyed by decision_number, but message
    facts only carry game_time - since decision_number is a per-actor
    counter (see `free_social_matchup._window_after`'s own comment), the
    nearest available snapshot key by insertion order is used as a
    reasonable single global proxy (ownership rarely changes multiple
    times within a few decisions)."""
    if not snapshots:
        return 0
    return max(snapshots.keys())


def episode_outcome_facts(*, episode_number: int, remembering_actor_id: str, completed_mission: bool, escaped: bool) -> SocialFact:
    status = "completed your mission and escaped" if escaped else ("made progress but did not escape in time" if completed_mission else "did not complete your mission in time")
    return SocialFact(
        episode_number=episode_number, fact_type="EPISODE_OUTCOME",
        text=f"In that episode, you {status}.",
        detail={"completed_mission": completed_mission, "escaped": escaped},
    )


def run_series(
    *, black_personality: str, white_personality: str, seeds: list[int], memory_on: bool,
    provider: str, model: str, region: str | None, agent_runtime: str, telemetry_path: str,
    episode_prefix: str = "m5f3c", llm_timeout: float = 60.0,
    # M5F.3D spec sections 6-8: the new causal intervention - only
    # meaningful when `memory_on=True`. `3` means "clear both actors'
    # accumulated memory immediately before building episode 4's
    # transform" (spec section 7's own example numbering: 1-indexed
    # episode number after which the clear happens). `None` (every
    # pre-M5F.3D caller's implicit value, including M5F.3C's own) means
    # no reset ever happens - byte-identical MEMORY_ON behavior.
    reset_after_episode: int | None = None,
) -> dict[str, Any]:
    """Runs one repeated series (spec sections 13, 36) - `len(seeds)`
    episodes, same stable actor identities throughout. Under MEMORY_ON,
    each actor's own `CrossEpisodeSocialMemory` accumulates across
    episodes and is injected into every subsequent episode via
    `model_payload_transform`; under MEMORY_OFF, no transform is ever
    passed (byte-identical to a bare `run_duel_episode` call). Under
    MEMORY_RESET (`memory_on=True` + `reset_after_episode` set), memory
    accumulates normally through `reset_after_episode`, is then cleared
    (never disclosed to the model - spec section 7's own instruction),
    and rebuilds from scratch starting the next episode."""
    black_memory: CrossEpisodeSocialMemory | None = None
    white_memory: CrossEpisodeSocialMemory | None = None
    episodes: list[dict[str, Any]] = []
    initial_traps = build_initial_traps()
    condition_label = "MEMORY_OFF" if not memory_on else ("MEMORY_RESET" if reset_after_episode is not None else "MEMORY_ON")

    for i, seed in enumerate(seeds, start=1):
        if memory_on and reset_after_episode is not None and i == reset_after_episode + 1:
            black_memory = None
            white_memory = None

        episode_id = f"{episode_prefix}-{black_personality}_vs_{white_personality}-{condition_label}-ep{i}-seed{seed}"

        trap_owner_snapshots: dict[int, dict[str, str | None]] = {}
        last_black_n = last_white_n = 0

        def _on_decision(black_n: int, white_n: int, state) -> None:
            nonlocal last_black_n, last_white_n
            if (black_n > last_black_n or white_n > last_white_n) or (black_n == 0 and white_n == 0):
                snap = {}
                for obj in state.level.objects:
                    entry = state.object_states.get(obj.id)
                    trap = entry.get("trap") if entry else None
                    snap[obj.id] = trap.get("owner_id") if trap else None
                trap_owner_snapshots[max(black_n, white_n)] = snap
                last_black_n, last_white_n = black_n, white_n

        # M5F.3D spec sections 34-35: explicit regression protection for
        # exactly the bug class M5F.3C's own analysis found and fixed live
        # (comparing an episode's messages against its own just-built
        # memory) - asserted here at the SOURCE, not just checked later in
        # analysis, so a violation fails loudly the moment it would matter.
        for mem in (black_memory, white_memory):
            if mem is not None:
                assert all(fact.episode_number < i for fact in mem.facts), (
                    f"memory integrity violation: episode {i}'s own injected memory contains a fact "
                    f"from episode >= {i} ({[f.episode_number for f in mem.facts if f.episode_number >= i]})"
                )

        # Captured BEFORE this episode's own facts are folded in below -
        # this is the memory actually INJECTED as input to episode i,
        # never the post-episode result (spec section 52's memory
        # snapshot trace wants exactly this "included prior episode IDs"
        # fact).
        included_prior_episode_ids_black = sorted({f.episode_number for f in black_memory.facts}) if black_memory else []
        included_prior_episode_ids_white = sorted({f.episode_number for f in white_memory.facts}) if white_memory else []
        is_reset_episode = bool(memory_on and reset_after_episode is not None and i == reset_after_episode + 1)

        black_transform = model_payload_transform_for(black_memory) if memory_on else None
        white_transform = model_payload_transform_for(white_memory) if memory_on else None

        result = run_duel_episode(
            seed=seed, provider=provider, model=model, region=region, agent_runtime=agent_runtime,
            telemetry_path=telemetry_path, episode_id=episode_id, max_decisions_per_actor=MAX_DECISIONS_PER_ACTOR,
            llm_timeout=llm_timeout, black_personality=black_personality, white_personality=white_personality,
            communication_enabled=True, zero_decision_cost_messaging=True, flat_message_schema=True,
            message_cooldown_decisions=MESSAGE_COOLDOWN, initial_traps=initial_traps,
            black_model_payload_transform=black_transform, white_model_payload_transform=white_transform,
            on_decision=_on_decision,
        )

        decisions, lifecycle = _load_decisions_and_lifecycle(telemetry_path, episode_id)
        facts = extract_episode_social_facts(
            episode_number=i, decisions=decisions, lifecycle=lifecycle, episode_id=episode_id, seed=seed,
            trap_owner_snapshots=trap_owner_snapshots, black_id=constants.BLACK_SPY_ID, white_id=constants.WHITE_SPY_ID,
        )
        facts[constants.BLACK_SPY_ID].append(episode_outcome_facts(
            episode_number=i, remembering_actor_id=constants.BLACK_SPY_ID,
            completed_mission=bool(result.black_game_facts.get("mission_success")), escaped=result.black_game_facts.get("escape_time") is not None,
        ))
        facts[constants.WHITE_SPY_ID].append(episode_outcome_facts(
            episode_number=i, remembering_actor_id=constants.WHITE_SPY_ID,
            completed_mission=bool(result.white_game_facts.get("mission_success")), escaped=result.white_game_facts.get("escape_time") is not None,
        ))

        if memory_on:
            black_memory = build_memory_from_episode_facts(
                remembering_actor_id=constants.BLACK_SPY_ID, opponent_actor_id=constants.WHITE_SPY_ID,
                episode_number=i, new_facts=facts[constants.BLACK_SPY_ID], prior_memory=black_memory,
            )
            white_memory = build_memory_from_episode_facts(
                remembering_actor_id=constants.WHITE_SPY_ID, opponent_actor_id=constants.BLACK_SPY_ID,
                episode_number=i, new_facts=facts[constants.WHITE_SPY_ID], prior_memory=white_memory,
            )

        episodes.append({
            "episode_number": i, "seed": seed, "episode_id": episode_id, "memory_condition": condition_label,
            "is_reset_episode": is_reset_episode,
            "included_prior_episode_ids_black": included_prior_episode_ids_black,
            "included_prior_episode_ids_white": included_prior_episode_ids_white,
            "status": result.status, "stop_reason": result.stop_reason,
            "black_decisions": result.black_decision_count, "white_decisions": result.white_decision_count,
            "black_validation_failure_count": result.black_validation_failure_count, "white_validation_failure_count": result.white_validation_failure_count,
            "black_provider_failure_count": result.black_provider_failure_count, "white_provider_failure_count": result.white_provider_failure_count,
            "black_game_facts": result.black_game_facts, "white_game_facts": result.white_game_facts,
            "messages_sent": len(analyze_social_on_episode(decisions, lifecycle, episode_id, seed)["messages"]),
            "black_memory_snapshot_text": black_memory.render_text() if black_memory else None,
            "white_memory_snapshot_text": white_memory.render_text() if white_memory else None,
            "black_memory_size_chars": len(black_memory.render_text() or "") if black_memory else 0,
            "white_memory_size_chars": len(white_memory.render_text() or "") if white_memory else 0,
        })

    return {
        "black_personality": black_personality, "white_personality": white_personality, "memory_on": memory_on,
        "memory_condition": condition_label, "reset_after_episode": reset_after_episode,
        "seeds": seeds, "episodes": episodes,
    }
