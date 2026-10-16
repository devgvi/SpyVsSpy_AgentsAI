"""M5D spec section 34: deterministic world, episode status transitions,
aggregation denominators, and failure isolation - through the real
EpisodeRunner/ExperimentRunner/Game pipeline, using provider="mock" (no
network) so these stay fast and deterministic. Large `frame_dt` values are
used to fast-forward wall-clock time in a few tests; this never touches
`constants.MISSION_TIME`/`mission_time_total` itself (spec section 12:
"do not alter established timer semantics to speed benchmarking") - it
only controls how much *wall-clock* time the harness spends simulating a
fixed number of *game* seconds, exactly the same `dt` parameter the real
game loop already threads through every frame.
"""

from src.evaluation.compatibility import CompatibilityStatus
from src.evaluation.episode_runner import EpisodeRunner, EpisodeStatus
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.spec import ExperimentSpec, VariantSpec


def _mock_variant(variant_id="native_mock"):
    return VariantSpec(variant_id=variant_id, agent_runtime="native", provider="mock", model="mock-model")


# ---------------------------------------------------------------------------
# Deterministic world (spec section 11, 34)
# ---------------------------------------------------------------------------


def test_matched_seed_produces_equivalent_initial_world_across_native_and_strands():
    from src.game import Game

    game_native = Game(seed=99001, white_controller="llm", llm_provider="mock", llm_model="mock-model", agent_runtime="native")
    game_strands = Game(seed=99001, white_controller="llm", llm_provider="ollama", llm_model="gemma4:latest", agent_runtime="strands")

    a, b = game_native.state, game_strands.state
    assert a.seed == b.seed == 99001
    assert a.mission_time_total == b.mission_time_total
    assert sorted(a.items.required) == sorted(b.items.required)
    assert [(o.id, tuple(o.contains)) for o in a.level.objects] == [(o.id, tuple(o.contains)) for o in b.level.objects]
    assert (a.actors["white_spy"].x, a.actors["white_spy"].y) == (b.actors["white_spy"].x, b.actors["white_spy"].y)
    assert (a.actors["black_spy"].x, a.actors["black_spy"].y) == (b.actors["black_spy"].x, b.actors["black_spy"].y)


def test_hidden_truth_absent_from_observation_for_matched_seed():
    from src.game import Game
    from src.observation import build_observation

    game = Game(seed=99001, white_controller="llm", llm_provider="mock", llm_model="mock-model", agent_runtime="native")
    observation = build_observation(game.state, "white_spy")

    # The evaluator (this test) may read GameState directly to verify
    # reproducibility (spec section 11) - but the actor's own Observation
    # must never carry the raw level/items/generator objects.
    assert not hasattr(observation, "level")
    assert not hasattr(observation, "items")
    assert not hasattr(observation, "generator")


# ---------------------------------------------------------------------------
# Compatibility integration: N/A gameplay metrics for a preflight failure
# ---------------------------------------------------------------------------


def test_incompatible_episode_never_starts_gameplay_and_has_na_metrics(tmp_path):
    spec = ExperimentSpec(
        experiment_id="incompat_test",
        variants=(VariantSpec(variant_id="bad", agent_runtime="totally_unknown_runtime", provider="mock", model="mock-model"),),
        seeds=(1,),
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner()
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))

    assert result.status == EpisodeStatus.INCOMPATIBLE
    assert result.compatibility_status == CompatibilityStatus.CONFIGURATION_ERROR
    assert result.mission_success is None
    assert result.escape_success is None
    assert result.game_time is None
    assert result.wall_time is None
    assert result.decision_count == 0
    assert result.metrics == {}
    assert not (tmp_path / "decisions.jsonl").exists()  # no gameplay -> no decision telemetry either


# ---------------------------------------------------------------------------
# Episode status transitions
# ---------------------------------------------------------------------------


def test_decision_limit_reached_status(tmp_path):
    spec = ExperimentSpec(
        experiment_id="decision_limit_test",
        variants=(_mock_variant(),),
        seeds=(1,),
        max_decisions=2,
        max_episode_wall_time=15.0,
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner(poll_sleep=0.0)
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))

    assert result.status == EpisodeStatus.DECISION_LIMIT_REACHED
    assert result.compatibility_status == CompatibilityStatus.SUPPORTED


# ---------------------------------------------------------------------------
# M5E.4.1: on_decision hook (spec section 13's evaluator-side progress snapshot)
# ---------------------------------------------------------------------------


def test_on_decision_hook_fires_once_at_start_and_once_per_completed_decision(tmp_path):
    spec = ExperimentSpec(
        experiment_id="on_decision_test", variants=(_mock_variant(),), seeds=(1,),
        max_decisions=3, max_episode_wall_time=15.0,
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner(poll_sleep=0.0)
    calls = []
    runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"), on_decision=lambda n, state: calls.append(n))

    # One call for the initial state (n=0) plus exactly one call per
    # completed decision, in strictly increasing order - never skipped,
    # never repeated for the same count.
    assert calls[0] == 0
    assert calls == sorted(set(calls))
    assert calls[-1] == 3


def test_on_decision_hook_receives_the_real_game_state(tmp_path):
    spec = ExperimentSpec(
        experiment_id="on_decision_state_test", variants=(_mock_variant(),), seeds=(1,),
        max_decisions=1, max_episode_wall_time=15.0,
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner(poll_sleep=0.0)
    seen_states = []
    runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"), on_decision=lambda n, state: seen_states.append(state))

    assert len(seen_states) >= 2  # initial + at least one decision
    assert all(hasattr(s, "actors") and hasattr(s, "door_states") for s in seen_states)


def test_on_decision_none_default_preserves_existing_behavior(tmp_path):
    """Regression: every pre-M5E.4.1 caller passes no `on_decision` at
    all - must behave byte-identically to before the hook existed."""
    spec = ExperimentSpec(
        experiment_id="on_decision_default_test", variants=(_mock_variant(),), seeds=(1,),
        max_decisions=2, max_episode_wall_time=15.0,
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner(poll_sleep=0.0)
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))
    assert result.status == EpisodeStatus.DECISION_LIMIT_REACHED


class _SlowMockClient:
    provider = "mock"
    model = "mock-model"

    def __init__(self, delay=0.3):
        self.delay = delay
        self.calls = 0

    def generate_action(self, *args, **kwargs):
        self.calls += 1
        import time as _time

        _time.sleep(self.delay)
        return {"action": "wait"}


def test_decision_limit_counts_completed_decisions_not_decision_attempts(monkeypatch, tmp_path):
    """Regression: LLMController._decision_seq counts every tick a fresh-
    enough observation *asked* for a decision, including ticks where a
    previous one is still in flight - evaluation mode pauses game time
    while thinking, but the digest can still shift tick to tick, so a slow
    model lets _decision_seq race far ahead of how many decisions actually
    completed. max_decisions must cap on genuinely completed decisions
    (the real telemetry event count), not that counter - verified live
    with gemma4:latest during M5D's own initial experiment, where a naive
    _decision_seq-based cap stopped every episode after exactly 1 real
    decision despite max_decisions=3.
    """
    import src.game as game_module

    def build_slow_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return _SlowMockClient(delay=0.3)

    monkeypatch.setattr(game_module.Game, "_build_llm_client", build_slow_mock)

    spec = ExperimentSpec(
        experiment_id="slow_decision_test",
        variants=(_mock_variant(),),
        seeds=(1,),
        max_decisions=2,
        max_episode_wall_time=15.0,
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner(poll_sleep=0.0)
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))

    assert result.status == EpisodeStatus.DECISION_LIMIT_REACHED
    assert result.decision_count == 2


def test_timed_out_status_when_wall_time_budget_exceeded(tmp_path):
    spec = ExperimentSpec(
        experiment_id="timeout_test",
        variants=(_mock_variant(),),
        seeds=(1,),
        max_episode_wall_time=0.0,
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner()
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))

    assert result.status == EpisodeStatus.TIMED_OUT


def test_aborted_status_when_harness_game_time_cap_cuts_episode_short(tmp_path):
    spec = ExperimentSpec(
        experiment_id="abort_test",
        variants=(_mock_variant(),),
        seeds=(1,),
        max_episode_game_time=5.0,  # well below the real 180s mission deadline
        max_decisions=10_000,
        max_episode_wall_time=30.0,
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner(frame_dt=1.0, poll_sleep=0.0)
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))

    assert result.status == EpisodeStatus.ABORTED
    assert result.error_type == "game_time_budget_exceeded"


def test_validation_failures_alone_do_not_trip_max_runtime_failures(monkeypatch, tmp_path):
    """M6A.2: real bug found and fixed live - this stop-check used to
    read the combined `fallback_count`, incremented identically for a
    genuine provider/infrastructure failure and an ordinary, cleanly-
    repaired-or-fallen-back MODEL VALIDATION failure alike (e.g. a
    hallucinated `trap_type`). A live M6A.2 pilot Strands episode hit
    21/39 such fallbacks - almost all `invalid_trap_type`, never a real
    infra problem - and this check aborted the episode at
    `max_runtime_failures=20` before it ever finished. `duel_runner.py`
    was fixed for exactly this conflation back in M5F.3B.1; this harness
    never got the same fix until now. A repeated invalid `place_trap`
    response (never a real provider exception) must never trip a *very*
    low `max_runtime_failures` cap - only `provider_failure_count`
    (always 0 here) should be checked against it."""
    import src.game as game_module
    from src.llm.clients.mock_client import MockModelClient

    invalid_response = {"action": "place_trap", "target_id": "desk_01", "trap_type": "not_a_real_trap_type", "reason": "x"}

    def build_invalid_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return MockModelClient(response=[dict(invalid_response) for _ in range(120)])

    monkeypatch.setattr(game_module.Game, "_build_llm_client", build_invalid_mock)

    spec = ExperimentSpec(
        experiment_id="validation_noise_test", variants=(_mock_variant(),), seeds=(1,),
        # A tight, unthrottled test loop (poll_sleep=0.0) races the
        # engine's own real 180s mission deadline to completion in just a
        # few real seconds regardless of any harness-level game-time cap
        # (that clock is `constants.MISSION_TIME`, fixed independently of
        # `EpisodeSpec.max_episode_game_time`) - only ~7-8 decisions fit
        # in that window given `LLMController`'s own 0.5s real-time
        # decision throttle (caught live: an earlier version of this test
        # used `max_decisions=15` and got COMPLETED at decision 8
        # instead). 6 is comfortably below that natural ceiling and still
        # well above `max_runtime_failures=2` - proof enough that many
        # validation failures alone don't trip it.
        max_decisions=6, max_runtime_failures=2, max_episode_wall_time=30.0,
    )
    episode_spec = spec.expand()[0]
    # Deliberately NOT fast-forwarding frame_dt here (unlike the game-time
    # cap test above): a large frame_dt races the mission's own 180s clock
    # to completion almost instantly, since `LLMController`'s own 0.5s
    # decision throttle is real-wall-clock-based, not game-time-based -
    # that starved this test of the many real decisions it needs to
    # accumulate validation failures, an easy trap for a test to fall into
    # (caught by this test itself initially asserting COMPLETED instead).
    runner = EpisodeRunner(poll_sleep=0.0)
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))

    assert result.status == EpisodeStatus.DECISION_LIMIT_REACHED
    assert result.error_type is None


def test_max_validation_failures_caps_episode_separately(monkeypatch, tmp_path):
    """The new, separate `max_validation_failures` cap (spec sections 25,
    45 of M6A.2 - mirrors `duel_runner.py`'s own established pattern)
    does correctly stop an episode that accumulates too much genuine
    model-output noise, once a caller opts in - `None` (every pre-M6A.2
    caller) never enables this check at all, confirmed by the test above."""
    import dataclasses

    import src.game as game_module
    from src.llm.clients.mock_client import MockModelClient

    invalid_response = {"action": "place_trap", "target_id": "desk_01", "trap_type": "not_a_real_trap_type", "reason": "x"}

    def build_invalid_mock(self, provider, llm_model, llm_base_url, llm_api_key):
        return MockModelClient(response=[dict(invalid_response) for _ in range(120)])

    monkeypatch.setattr(game_module.Game, "_build_llm_client", build_invalid_mock)

    spec = ExperimentSpec(
        experiment_id="validation_cap_test", variants=(_mock_variant(),), seeds=(1,),
        max_decisions=6, max_runtime_failures=20, max_episode_wall_time=30.0,
    )
    episode_spec = dataclasses.replace(spec.expand()[0], max_validation_failures=3)
    runner = EpisodeRunner(poll_sleep=0.0)
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))

    assert result.status == EpisodeStatus.ABORTED
    assert result.error_type == "max_validation_failures_exceeded"


def test_completed_status_with_legitimate_mission_loss_at_natural_deadline(tmp_path):
    # The engine's own 180s deadline is never shortened here - only the
    # harness's wall-clock-per-game-second ratio (frame_dt) is large, so
    # the same fixed 180 game-seconds passes in a fraction of a real second.
    spec = ExperimentSpec(
        experiment_id="natural_deadline_test",
        variants=(_mock_variant(),),
        seeds=(1,),
        max_decisions=10_000,
        max_episode_wall_time=30.0,
        max_episode_game_time=500.0,  # above the real deadline, so it never cuts this short
    )
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner(frame_dt=10.0, poll_sleep=0.0)
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))

    # A legitimate mission loss inside a COMPLETED episode - status and
    # outcome are different axes (spec section 9).
    assert result.status == EpisodeStatus.COMPLETED
    assert result.mission_success is False
    assert result.game_time == 180.0


def test_failed_status_on_unexpected_engine_exception(monkeypatch, tmp_path):
    import src.game as game_module

    def exploding_step(self, dt):
        raise RuntimeError("synthetic engine failure")

    monkeypatch.setattr(game_module.Game, "_step", exploding_step)

    spec = ExperimentSpec(experiment_id="fail_test", variants=(_mock_variant(),), seeds=(1,))
    episode_spec = spec.expand()[0]
    runner = EpisodeRunner()
    result = runner.run(episode_spec, telemetry_path=str(tmp_path / "decisions.jsonl"))

    assert result.status == EpisodeStatus.FAILED
    assert result.error_type == "RuntimeError"
    assert "synthetic engine failure" in result.error_message


# ---------------------------------------------------------------------------
# Aggregation denominators (spec section 24)
# ---------------------------------------------------------------------------


def test_requested_supported_completed_denominators_stay_distinct(tmp_path):
    spec = ExperimentSpec(
        experiment_id="agg_test",
        variants=(
            _mock_variant("supported_variant"),
            VariantSpec(variant_id="unsupported_variant", agent_runtime="totally_unknown_runtime", provider="mock", model="mock-model"),
        ),
        seeds=(1, 2),
        max_decisions=2,
        max_episode_wall_time=15.0,
    )
    runner = ExperimentRunner(episode_runner=EpisodeRunner(poll_sleep=0.0), output_root=str(tmp_path))
    result = runner.run(spec)
    summary = result.to_summary_dict()

    assert summary["requested"] == 4
    assert summary["supported"] == 2  # only the "supported_variant" episodes
    assert summary["completed"] <= summary["supported"]
    assert summary["successful"] <= summary["completed"]

    supported_summary = summary["by_variant"]["supported_variant"]
    unsupported_summary = summary["by_variant"]["unsupported_variant"]
    assert supported_summary["requested"] == 2
    assert supported_summary["supported"] == 2
    assert unsupported_summary["requested"] == 2
    assert unsupported_summary["supported"] == 0
    assert unsupported_summary["compatibility_statuses"] == {"CONFIGURATION_ERROR": 2}


def test_output_files_are_written(tmp_path):
    spec = ExperimentSpec(experiment_id="output_test", variants=(_mock_variant(),), seeds=(1,), max_decisions=1, max_episode_wall_time=10.0)
    runner = ExperimentRunner(episode_runner=EpisodeRunner(poll_sleep=0.0), output_root=str(tmp_path))
    runner.run(spec)

    out_dir = tmp_path / "output_test"
    assert (out_dir / "experiment.json").exists()
    assert (out_dir / "episodes.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "decisions.jsonl").exists()


def test_decisions_jsonl_episode_id_matches_episodes_jsonl(tmp_path):
    """Regression: decisions.jsonl telemetry used to stamp GameState's own
    internal UUID as episode_id, not the harness's own
    {experiment}__{variant}__seed{seed} id every other output file uses -
    so raw decision telemetry could never actually be joined against
    episodes.jsonl/summary.json by that id (caught by an M5D.1 integrity
    check run against real pilot output)."""
    import json

    spec = ExperimentSpec(experiment_id="correlation_test", variants=(_mock_variant(),), seeds=(1,), max_decisions=1, max_episode_wall_time=10.0)
    runner = ExperimentRunner(episode_runner=EpisodeRunner(poll_sleep=0.0), output_root=str(tmp_path))
    runner.run(spec)

    out_dir = tmp_path / "correlation_test"
    with open(out_dir / "episodes.jsonl") as f:
        episode_ids = {json.loads(line)["episode_id"] for line in f}
    with open(out_dir / "decisions.jsonl") as f:
        decision_episode_ids = {json.loads(line)["episode_id"] for line in f if json.loads(line).get("event_type") == "llm_decision"}

    assert decision_episode_ids
    assert decision_episode_ids <= episode_ids


# ---------------------------------------------------------------------------
# Failure isolation (spec section 33)
# ---------------------------------------------------------------------------


class _FlakyEpisodeRunner(EpisodeRunner):
    def __init__(self, fail_on_seed: int, **kwargs):
        super().__init__(**kwargs)
        self.fail_on_seed = fail_on_seed

    def run(self, episode_spec, telemetry_path):
        if episode_spec.seed == self.fail_on_seed:
            raise RuntimeError("synthetic episode failure")
        return super().run(episode_spec, telemetry_path)


def test_one_episode_failure_does_not_stop_the_experiment(tmp_path):
    spec = ExperimentSpec(
        experiment_id="flaky_test",
        variants=(_mock_variant(),),
        seeds=(1, 2, 3),
        max_decisions=2,
        max_episode_wall_time=15.0,
    )
    flaky_runner = _FlakyEpisodeRunner(fail_on_seed=2, poll_sleep=0.0)
    runner = ExperimentRunner(episode_runner=flaky_runner, output_root=str(tmp_path))
    result = runner.run(spec)

    assert len(result.episodes) == 3
    statuses = {e.seed: e.status.value for e in result.episodes}
    assert statuses[1] == "DECISION_LIMIT_REACHED"
    assert statuses[2] == "FAILED"
    assert statuses[3] == "DECISION_LIMIT_REACHED"
    assert "synthetic episode failure" in next(e.error_message for e in result.episodes if e.seed == 2)
