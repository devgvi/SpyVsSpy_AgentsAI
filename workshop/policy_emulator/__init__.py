"""Stage 3 - behavioural policy emulator (first milestone).

A cheap, interpretable stochastic policy fitted to the logged decisions of one
LLM + personality configuration, plugged into the real game through the
existing controller contract (`get_action(observation) -> GameAction`).

Modules:
    schema      versions, policy identities, action categories, target classes
    features    PolicyFeatureExtractorV1 (observable-only features)
    dataset     the canonical normalized dataset (Phase-A full matches, Stage-2 snapshots)
    policies    B0 frequency, B1 opportunity-conditioned table, target policy, masking, artifacts
    evaluation  logged-state replay: leave-one-seed-out and controlled-snapshot validation
    report      the standalone HTML report
    surrogate   SurrogateController

Spec: Archive/STAGE3_BEHAVIORAL_POLICY_EMULATOR_V2.md. Nothing here calls a provider.
"""
