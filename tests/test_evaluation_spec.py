"""M5D spec section 34: "Spec expansion" - 2 variants x 3 seeds = 6 stable
episode specs, deterministic, plus round-trip serialization."""

from src.evaluation.spec import EpisodeSpec, ExperimentSpec, VariantSpec


def _two_variants():
    return (
        VariantSpec(variant_id="native_gemma4", agent_runtime="native", provider="ollama", model="gemma4:latest"),
        VariantSpec(variant_id="strands_gemma4", agent_runtime="strands", provider="ollama", model="gemma4:latest"),
    )


def test_variant_x_seed_expansion_is_2x3_equals_6():
    spec = ExperimentSpec(experiment_id="exp1", variants=_two_variants(), seeds=(1, 2, 3))
    episodes = spec.expand()

    assert len(episodes) == 6
    assert all(isinstance(e, EpisodeSpec) for e in episodes)
    assert len(set(e.episode_id for e in episodes)) == 6  # all unique


def test_expansion_order_is_deterministic_and_stable():
    spec = ExperimentSpec(experiment_id="exp1", variants=_two_variants(), seeds=(1, 2, 3))
    first = [e.episode_id for e in spec.expand()]
    second = [e.episode_id for e in spec.expand()]
    assert first == second
    assert first == [
        "exp1__native_gemma4__seed1",
        "exp1__native_gemma4__seed2",
        "exp1__native_gemma4__seed3",
        "exp1__strands_gemma4__seed1",
        "exp1__strands_gemma4__seed2",
        "exp1__strands_gemma4__seed3",
    ]


def test_episode_spec_carries_full_correlation_chain():
    spec = ExperimentSpec(experiment_id="exp1", variants=_two_variants(), seeds=(7,))
    episode = spec.expand()[0]
    assert episode.experiment_id == "exp1"
    assert episode.variant_id == "native_gemma4"
    assert episode.seed == 7
    assert episode.episode_id == "exp1__native_gemma4__seed7"


def test_experiment_spec_json_round_trip():
    spec = ExperimentSpec(experiment_id="exp1", variants=_two_variants(), seeds=(1, 2))
    restored = ExperimentSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.expand()[0].episode_id == spec.expand()[0].episode_id


def test_variant_spec_dict_round_trip():
    variant = VariantSpec(variant_id="v1", agent_runtime="strands", provider="ollama", model="gemma4:latest", llm_timeout=45.0)
    restored = VariantSpec.from_dict(variant.to_dict())
    assert restored == variant
