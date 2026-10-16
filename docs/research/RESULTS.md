# Results

> In this public repository, file names under `workshop/` and `workshop_runs/` refer to the frozen research archive, which is distributed separately (see [Reproducibility](#reproducibility)).


What the research found, with the frozen numbers, where each number comes from, and how to check it. For the
narrative, read [STORY.md](STORY.md); for the full study, [the paper](../../paper/SPY_VS_AI_PAPER.md). Terms are in
the [glossary](../project/GLOSSARY.md).

Distances are to REAL (lower is closer). P is the bootstrap probability over matched configurations (10,000
resamples). The research models were Claude Sonnet 5.5 (`sonnet55`) and GPT-6.1 Sol (`gpt61sol`) on Amazon Bedrock,
with four personalities: cautious, opportunist, racer, saboteur.

Contents:
- [Summary](#summary)
- [1. Imitation is not enough](#1-imitation-is-not-enough)
- [2. Selective escalation works](#2-selective-escalation-works)
- [3. It generalizes under environment shift](#3-it-generalizes-under-environment-shift)
- [4. Distillation cuts provider calls, but is not safely promotable](#4-distillation-cuts-provider-calls-but-is-not-safely-promotable)
- [5. The harness matters](#5-the-harness-matters)
- [What did not work](#what-did-not-work-or-misled-us)
- [Final status](#final-status)
- [Remaining uncertainty](#remaining-uncertainty)
- [The experiments](#the-experiments)
- [The claims](#the-claims)
- [Reproducibility](#reproducibility)

## Summary

> **A cheap surrogate can handle a large share of decisions, and selective frontier-LLM escalation recovers much of
> the behavioural fidelity a pure surrogate loses. Distilling the LLM's hard-case decisions back into the surrogate
> reduces provider demand further, but the benefit is heterogeneous, and safe promotion of the distilled policy was
> not established.**

```text
MODEL BEHAVIOUR  ≠  AGENT BEHAVIOUR  ≠  MATCH OUTCOME                      (lesson J)
local decision fidelity  ≠  closed-loop trajectory fidelity               (lesson A, CONFIRMED)
```

A model's isolated choices, an agent's behaviour over a match, and who wins are three different things. A policy that
predicts single decisions well can still produce trajectories unlike the reference, because its own choices shape the
states it meets next: agent behaviour compounds (lesson B). Every finding below is one of the 33 frozen claims, listed
in [The claims](#the-claims) with their status and evidence.

## 1. Imitation is not enough

**The data (Phase A).** 48 REAL matches (Claude Sonnet 5.5 and GPT-6.1 Sol, four personalities, Level 1) gave 4,062
decision records, 3,951 of them clean, with 54 model-visible features (dataset hash `sha256:90a0f39349ab509a`).

**Offline, B2 is the best imitator** (leave-one-seed-out,
`workshop/policy_reports/phase-a-v2/report.html`):

| Emulator | Top-1 | Log loss |
|---|---|---|
| B0 frequency | 0.487 | 1.097 |
| B1 table | 0.780 | 0.722 |
| B2 logistic | 0.891 | 0.377 |
| B3 tree | 0.849 | 0.600 |

**In closed loop, B2 is not adequate alone.** After replicates, B2 was selected over B1: state visitation 0.058 vs
0.128 (`workshop/policy_reports/phase-a-v2/closed_loop_report.html`). But it showed a
repetition pathology. Its decisions in runs of four or more identical actions were 13.7% against 5.9% for REAL on
Level 1, and 32.0% against 2.8% on the shifted levels. Offline metrics were not sufficient to validate an agent
(lesson E).

## 2. Selective escalation works

**The router.** R2.1 escalates on a repetition loop OR a strong B1/B2 disagreement. In held-out validation it passed
10 of 11 gates and escalated 13.5% of decisions
(`workshop/policy_reports/phase-a-v2/hybrid_router_v2_validation.html`).

**The preregistered Level-1 trial** (M4D: seeds 8001-8004, 32 configurations) gave **decision A** (lesson C,
CONFIRMED; `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.html`):

| | B2 | Hybrid R2.1 | P(Hybrid closer) |
|---|---|---|---|
| state visitation | 0.0662 | 0.0338 | 0.9991 |
| trajectory distance | 0.6399 | 0.4869 | 0.9826 |
| decisions in runs of 4+ (REAL 5.9%) | 13.7% | 6.8% | |
| provider calls (REAL 2,650) | 0 | 463 (ratio 0.175) | |

**Disagreement was the more useful signal** (lesson F). When the LLM changed B2's action at a disagreement-only
escalation, the change was more REAL-typical in 67% (M4D), 74% (4B) and 88% (4C) of cases. At repetition-only
escalations, it was 17%, 9% and 21%. Repetition remains important as evidence of pathology; whether the repetition
trigger is needed at all is UNRESOLVED.

## 3. It generalizes under environment shift

**The shift is real** (lesson D). On Levels 2 and 3, B2 met mostly states unlike its training data, and the router
was predicted to escalate about twice as often
(`workshop/policy_reports/stage4/stage4a_shift_diagnostics.html`):

| | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| B2 LOW/OOD share | 0.255 | 1.000 | 0.992 |
| predicted R2.1 escalation | 0.201 | 0.381 | 0.389 |

**Locally better can be globally worse** (lesson G). In the provider-backed pilot (recommendation B: mixed, stop and
analyze), the Hybrid on Level 3 was closer to REAL on state visitation (0.104 vs 0.131) but further on the historical
trajectory metric (1.139 vs 0.954)
(`workshop/policy_reports/stage4/stage4b_pilot_report.html`).

**The metric mattered.** The historical trajectory metric mixed counts with rates and compared timing on thin
support; its robustness is NOT SUPPORTED
(`workshop/policy_reports/stage4/stage4b1_trajectory_diagnostic.html`).
`trajectory-v2` was designed offline and frozen before any ranking and before the next preregistration
(`workshop/policy_reports/stage4/stage4b2_metric_robustness.html`). The
historical metric was kept as it was.

**The preregistered Stage-4C trial** gave **decision A - confirmed generalization**
(`workshop/policy_reports/stage4/stage4c_confirmatory_report.html`):

| cell | B2 | Hybrid R2.1 | P(Hybrid closer) |
|---|---|---|---|
| Level 2 state visitation | 0.1059 | 0.0600 | 1.0 |
| Level 2 trajectory-v2 | 1.1118 | 0.6280 | 1.0 |
| Level 3 state visitation | 0.1008 | 0.0483 | 1.0 |
| Level 3 trajectory-v2 | 1.1270 | 0.5578 | 1.0 |

Hybrid / REAL provider calls were 0.361 pooled (Level 2 0.385, Level 3 0.340): about twice the Level-1 ratio, with
much better fidelity than B2. That call demand rises under shift is recorded as EXPLORATORY. Final status of
Hybrid-v1 / R2.1: **CONFIRMED / GENERALIZED RESEARCH BASELINE**.

## 4. Distillation cuts provider calls, but is not safely promotable

**Stage 5A, offline**
(`workshop/policy_reports/stage5/stage5a_candidate_training.html`). B2-v2
(candidate C1) was trained on the Phase-A clean data plus 854 hard examples. Development log loss went 0.4412 →
0.2896, top-1 0.852 → 0.900, and expected R2.1 escalation 0.2706 → 0.2373 (-12.3%). Decision A. Weighting the hard
examples more made things worse.

**Stage 5B, call-free closed loop**, 864 matches with R2.1 in shadow
(`workshop/policy_reports/stage5/stage5b_closed_loop.html`):

- shadow escalation went 0.3575 → 0.2915 (-18.5%);
- timeouts went 0.374 → 0.219, and mission completion 0.234 → 0.332;
- by personality: cautious -39.6%, saboteur -56.9%, opportunist -11.7%, racer +24.6%.

Decision A.

**Stage 5C, with real providers.** The preregistered trial ran on Levels 2-3 after Amendment 02: 64 configurations
and 192 provider-backed matches
(`workshop/policy_reports/stage5/stage5c_confirmatory_report.html`). The
hard-example distillation reduced real provider calls by 14.6% (lesson H, CONFIRMED):

| efficiency | value |
|---|---|
| provider calls REAL / Hybrid-v1 / Hybrid-v2 | 7,504 / 3,044 / 2,599 |
| relative reduction, Hybrid-v2 vs Hybrid-v1 | 14.6% (95% CI 6.8%-22.0%) |
| P(Hybrid-v2 fewer calls) | 0.9998 |
| Hybrid-v2 / REAL, Hybrid-v1 / REAL | 0.346, 0.406 |
| Level 2, Level 3 reduction | 15.3%, 14.0% |

**One non-inferiority cell was inconclusive:**

| non-inferiority cell | Hybrid-v1 | Hybrid-v2 | difference | 95% CI | margin | result |
|---|---|---|---|---|---|---|
| Level 2 state visitation | 0.0565 | 0.0573 | +0.0008 | [-0.0171, +0.0182] | 0.0168 | not established |
| Level 2 trajectory-v2 | 0.6980 | 0.6746 | -0.0234 | [-0.1396, +0.0576] | 0.1303 | non-inferior |
| Level 3 state visitation | 0.0445 | 0.0465 | +0.0020 | [-0.0130, +0.0170] | 0.0262 | non-inferior |
| Level 3 trajectory-v2 | 0.6916 | 0.6226 | -0.0689 | [-0.1503, +0.0394] | 0.2371 | non-inferior |

The Level-2 state point estimate shows no measurable degradation, but the upper bound (0.0182) exceeds the frozen
margin (0.0168) by 0.0014. The cell is inconclusive, not refuted; its prior-only power at four seeds was 0.44. No
catastrophic personality or identity regression occurred. Under the frozen rule:

> **Decision B - provider reduction is real but Hybrid-v2 is not safely promotable under the frozen confirmatory rule.**

Hybrid-v2 is not a failure, since its efficiency gain is confirmed, and it is not fully validated. Its final status is
**MORE EFFICIENT BUT NOT SAFELY PROMOTABLE UNDER THE CONFIRMATORY RULE**, and Hybrid-v1 stays the default research
policy.

**Where the savings came from.** About 69% of the avoided calls came from fewer escalation-prone decisions and about
31% from shorter trajectories. Escalation probability given a signal, and repairs or retries, contributed nothing
measurable. B2-v2 changed closed-loop behaviour so that the router met fewer hard states.

**The gains were heterogeneous** across personalities and identities (lesson I):

| provider calls by personality | REAL | Hybrid-v1 | Hybrid-v2 | ratio v2 / v1 |
|---|---|---|---|---|
| cautious | 2,372 | 802 | 581 | 0.72 |
| opportunist | 1,787 | 967 | 913 | 0.94 |
| racer | 1,674 | 732 | 809 | 1.11 |
| saboteur | 1,671 | 543 | 296 | 0.55 |

This pattern is consistent with uneven hard-example coverage across personalities (cautious and saboteur dominated
the promoted examples), but the experiment did not establish that imbalance as the causal mechanism (UNRESOLVED).

## 5. The harness matters

The observation representation, prompts, repair and fallback, scheduling and execution semantics all changed
behaviour during the project (lesson K). The LLM is one component of the agent:

```text
model + observation + representation + memory + prompt / personality + action schema + router
      + repair / fallback + scheduling + execution semantics + environment history
```

## What did not work, or misled us

- The first tournament's standings seemed to rank personalities, but seed and start-room effects confounded them
  (NOT SUPPORTED). Later trials counterbalanced orientations and compared matched configurations.
- Offline quality overstated closed-loop reliability (B2's loop pathology).
- B1 was not adequate in closed loop (NOT SUPPORTED as adequate).
- An early "critical state" router trigger spent calls with no local gain and was dropped from R2.1.
- Repetition alone was a weak local correction signal.
- The historical trajectory metric was not robust under shift.
- Locally REAL-typical corrections did not guarantee global fidelity.
- B2-v2 did not improve every personality (racer regressed) or every identity.
- Infrastructure failed twice in provider-backed trials: an expired AWS session in M4D, and a provider service error
  that tripped a fail-fast check in Stage 5C. Both were handled by recorded amendments, without rewriting data.

## Final status

```text
Hybrid-v1 / R2.1   CONFIRMED / GENERALIZED RESEARCH BASELINE          (default research policy)
Hybrid-v2 / R2.1   MORE EFFICIENT BUT NOT SAFELY PROMOTABLE UNDER THE CONFIRMATORY RULE
Stage 4C decision  A — confirmed generalization
Stage 5C decision  B — provider reduction is real but Hybrid-v2 is not safely promotable under the frozen confirmatory rule
MAIN PROVIDER-BACKED RESEARCH LINE CLOSED
```

The statuses come from the FINAL-1 release manifest,
[freeze_manifest.json](../../workshop/policies/final-release/freeze_manifest.json) (`status`, manifest hash
`sha256:e5943690ede2244c`). The whole story, with every stage's table, is in
`workshop/policy_reports/final_research_synthesis.md`.

**The main conclusion:** a useful agent architecture can be layered by decision cost:

```text
cheap deterministic logic  →  cheap learned policy  →  selective escalation  →  frontier reasoning
```

The design question becomes: **what is the cheapest mechanism that makes this decision well enough in closed loop?**

## Remaining uncertainty

The project does not establish (UNRESOLVED or out of scope) that:

- Hybrid-v1 is optimal among possible routers or surrogates;
- the repetition trigger is needed, or that per-model routers would do better;
- B2-v2 generalizes to arbitrary environments;
- hard-example distillation always reduces cost, or that its heterogeneity is caused by training-example imbalance;
- the findings transfer to non-game domains or to every model family.

Those are follow-up research questions (other decision layers, environments, harnesses, more balanced active learning,
multi-environment generalization), not unfinished requirements of this project.

## The experiments

Oldest first. Engineering smoke tests and pilots of the runtime are left out. Each report exists as html, md and json.

| Experiment | Question | Design | Result | Evidence |
|---|---|---|---|---|
| First personality tournament | do the personalities rank? | 60 self-play matches, Sonnet 5.5 and Opus 5.5, 10 pairings × 3 seeds | standings confounded by seed and start rooms (NOT SUPPORTED); led to counterbalancing | `workshop/tournament_findings.md` |
| Stage-2 behaviour grid | how do personality and model change single decisions? | 8 frozen snapshots × 4 personalities × 2 models, 352 decisions | controlled per-decision differences | `workshop_runs/behavior-grid-stage2-v1-sonnet55-vs-gpt61sol/analysis/report.html` |
| Phase A data | the training data of every surrogate | 48 REAL matches, Level 1 | 4,062 records, 3,951 clean | dataset `sha256:90a0f39349ab509a` |
| Stage 3 M2: offline surrogates | which cheap model best predicts REAL? | B0, B1, B2, B3, leave-one-seed-out | B2 best offline | `workshop/policy_reports/phase-a-v2/report.html` |
| M3: closed loop | which surrogate behaves more like REAL in the game? | 48 configurations, call-free, 3 replicates (288 runs) | B2 selected | `workshop/policy_reports/phase-a-v2/closed_loop_report.html` |
| M4A-M4C: router | can a router send only the hard decisions to the LLM? | call-free replay of R1-R3; 8-match provider pilot; held-out validation | R2.1, 10/11 gates | `workshop/policy_reports/phase-a-v2/hybrid_router_v2_validation.html`, [router.json](../../workshop/policies/phase-a-v2/hybrid-router-v2/router.json) |
| M4D: preregistered Level-1 trial | is Hybrid R2.1 closer to REAL than B2? | seeds 8001-8004, 32 configurations × REAL / B2 / Hybrid | **A**: confirmed | `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.html`, `workshop/policies/phase-a-v2/hybrid-r21-confirmatory/preregistration.json` |
| 4A: shift diagnostics | how different are Levels 2 and 3? | call-free | strong shift | `workshop/policy_reports/stage4/stage4a_shift_diagnostics.html` |
| 4B: provider-backed pilot | a first look under shift | 16 configurations × REAL / B2 / Hybrid | **B**: mixed, stop and analyze | `workshop/policy_reports/stage4/stage4b_pilot_report.html` |
| 4B.1, 4B.2: metric | why the conflict? | offline diagnostic; metric robustness study | trajectory-v2, frozen before any ranking | `workshop/policy_reports/stage4/stage4b2_metric_robustness.html` |
| 4C: preregistered shift trial | does Hybrid R2.1 generalize? | fresh seeds, 32 configurations per level × REAL / B2 / Hybrid | **A**: confirmed generalization | `workshop/policy_reports/stage4/stage4c_confirmatory_report.html`, `workshop/policies/stage4/stage4c_preregistration.json` |
| 5A: hard-example promotion | can hard examples make B2 better and less escalation-prone? | C0 control, C1, C2 (×2 / ×4 weights) | **A**: C1 = B2-v2 | `workshop/policy_reports/stage5/stage5a_candidate_training.html`, [manifest.json](../../workshop/policies/stage5/b2-v2-candidate/manifest.json) |
| 5B: call-free selection | does B2-v2 hold up in closed loop? | 864 call-free matches, R2.1 in shadow | **A**: viable; racer regressed | `workshop/policy_reports/stage5/stage5b_closed_loop.html` |
| 5C: provider confirmation | fewer real calls without losing fidelity? | Levels 2-3, seeds 9200-9203, 64 configurations × REAL / Hybrid-v1 / Hybrid-v2 (192 matches) | **B**: provider reduction real, not safely promotable | `workshop/policy_reports/stage5/stage5c_confirmatory_report.html`, `workshop/policies/stage5/stage5c_preregistration.json` |
| FINAL-1: freeze | one release over everything | manifest, claims, registries, synthesis, offline reproduction | REPRODUCED | [freeze_manifest.json](../../workshop/policies/final-release/freeze_manifest.json), `workshop/policies/final-release/claims.json` |

## The claims

<!-- claims:begin -->
Generated by `python -m tools.build_claims_index` from the frozen claims registry `workshop/policies/final-release/claims.json`
(final-claims-v1, claims hash `sha256:db0d430306b83476`). The JSON is authoritative: it also holds each claim's
evidence numbers and limitations.

33 claims: CONFIRMED 10, SUPPORTED 12, EXPLORATORY 1, UNRESOLVED 5, NOT SUPPORTED 5.

### Lessons

| | Lesson |
|---|---|
| A | Local fidelity is not closed-loop fidelity |
| B | Agent behavior compounds |
| C | Selective frontier reasoning works |
| D | Environment shift matters |
| E | Repetition is a real surrogate pathology |
| F | Disagreement was a more useful escalation signal than repetition |
| G | Local REAL-typical corrections can still worsen global trajectories |
| H | Hard-example distillation can reduce future provider demand |
| I | Distillation benefit is heterogeneous |
| J | Model behavior is not agent behavior (MODEL BEHAVIOR != AGENT BEHAVIOR != MATCH OUTCOME) |
| K | Harness / runtime matters |

### CONFIRMED

| Claim | Lesson | Statement | Stage | Evidence |
|---|---|---|---|---|
| `closed_loop_not_local` | A | Strong local decision prediction does not guarantee REAL-like closed-loop trajectories. | Stage 3 M2 / M3, M4D, Stage 4C | `workshop/policy_reports/phase-a-v2/report.json`, `workshop/policy_reports/phase-a-v2/closed_loop_report.json`, `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.json`, `workshop/policy_reports/stage4/stage4c_confirmatory_report.json` |
| `selective_frontier_reasoning_works` | C | Selective frontier reasoning (B2-v1 + router R2.1) substantially restores REAL-like behaviour while using a fraction of REAL provider calls. | Stage 3 M4D, Stage 4C | `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.json`, `workshop/policy_reports/stage4/stage4c_confirmatory_report.json` |
| `hard_example_distillation_reduces_calls` | H | Hard-example distillation reduces real provider demand: Hybrid-v2 makes fewer provider calls than Hybrid-v1. | Stage 5C (Amendment 02 population) | `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `b2_better_than_b1_offline` |  | B2 (logistic) predicts the individual next action better than B1 (table). | Stage 3 M2 | `workshop/policy_reports/phase-a-v2/report.json` |
| `b2_better_than_b1_closed_loop` |  | B2 has better closed-loop fidelity than B1 overall. | Stage 3 M3 | `workshop/policy_reports/phase-a-v2/closed_loop_report.json` |
| `hybrid_r21_level1_confirmed` |  | Hybrid R2.1 improves state-visitation and trajectory fidelity over B2 on fresh Level-1 configurations. | Stage 3 M4D | `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.json` |
| `hybrid_r21_generalizes` |  | Hybrid R2.1 generalizes under environment shift: it is closer to REAL than B2 on state visitation and trajectory-v2 on Levels 2 and 3. | Stage 4C | `workshop/policy_reports/stage4/stage4c_confirmatory_report.json` |
| `hybrid_cheaper_than_real` |  | Hybrid R2.1 is materially cheaper in provider calls than REAL. | Stage 3 M4D, Stage 4C, Stage 5C | `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.json`, `workshop/policy_reports/stage4/stage4c_confirmatory_report.json`, `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `stage5c_ni_three_cells` |  | Hybrid-v2 is non-inferior to Hybrid-v1 on trajectory-v2 (Levels 2 and 3) and on Level-3 state visitation. | Stage 5C | `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `stage5c_no_catastrophic_guard` |  | No catastrophic personality or identity guard fired for Hybrid-v2. | Stage 5C | `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |

### SUPPORTED

| Claim | Lesson | Statement | Stage | Evidence |
|---|---|---|---|---|
| `agent_behavior_compounds` | B | Small local action differences compound: they change future observations, the state distribution, the available actions, mission progress and the difficulty of later decisions. | Stage 3 M3, Stage 4B.1, Stage 5C | `workshop/policy_reports/phase-a-v2/closed_loop_report.json`, `workshop/policy_reports/stage4/stage4b1_trajectory_diagnostic.json`, `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `environment_shift_matters` | D | Environment shift (Levels 2 and 3) increases LOW/OOD exposure and the demand for provider escalation. | Stage 4A, Stage 4C | `workshop/policy_reports/stage4/stage4a_shift_diagnostics.json`, `workshop/policy_reports/stage4/stage4c_confirmatory_report.json`, `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.json` |
| `repetition_surrogate_pathology` | E | Closed-loop repeated actions are materially more common for pure surrogate behaviour than for REAL. | Stage 3 M3, M4D, Stage 4C, Stage 5B | `workshop/policy_reports/phase-a-v2/closed_loop_report.json`, `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.json`, `workshop/policy_reports/stage4/stage4c_confirmatory_report.json`, `workshop/policy_reports/stage5/stage5b_closed_loop.json` |
| `disagreement_beats_repetition_trigger` | F | The B1/B2 disagreement trigger was a more useful escalation signal than the repetition trigger. | Stage 3 M4D, Stage 4B, Stage 4C, Stage 5C | `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.json`, `workshop/policy_reports/stage4/stage4b_pilot_report.json`, `workshop/policy_reports/stage4/stage4c_confirmatory_report.json`, `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `local_corrections_can_worsen_global` | G | A locally more REAL-typical decision does not guarantee a globally more REAL-like trajectory. | Stage 4B, Stage 4B.1 | `workshop/policy_reports/stage4/stage4b_pilot_report.json`, `workshop/policy_reports/stage4/stage4b1_trajectory_diagnostic.json` |
| `distillation_benefit_heterogeneous` | I | The distillation benefit is heterogeneous across personalities and model / personality identities. | Stage 5C (with Stage 5B as the call-free precursor) | `workshop/policy_reports/stage5/stage5c_confirmatory_report.json`, `workshop/policy_reports/stage5/stage5b_closed_loop.json` |
| `model_agent_outcome_distinction` | J | MODEL BEHAVIOR != AGENT BEHAVIOR != MATCH OUTCOME: the same model yields different agent behaviour under different harnesses, and the same agent behaviour yields different match outcomes under different seeds / sides. | M5-M6 harness milestones, first tournament, Phase A, Stage 3 | `workshop/tournament_findings.md`, `docs/m6_final_synthesis.md`, `workshop/policy_reports/phase-a-v2/closed_loop_report.json` |
| `harness_runtime_matters` | K | Final agent behaviour depends on the whole harness: LLM, observation, representation, memory, action schema, personality prompt, router, repair / fallback, execution semantics, scheduling and environment history. | M5-M6, Stage 4A, Stage 5C | `docs/m6_final_synthesis.md`, `workshop/policy_reports/stage4/stage4a_shift_diagnostics.json`, `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `hybrid_recovers_escape` |  | Hybrid closely recovers REAL mission completion / escape behaviour that B2 alone loses under shift. | Stage 4C, Stage 5C | `workshop/policy_reports/stage4/stage4c_confirmatory_report.json`, `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `stage5a_candidate_offline` |  | The Stage-5A candidate (B2-v2 = C1) is more accurate and less escalation-prone than B2-v1 on the DEV split. | Stage 5A | `workshop/policy_reports/stage5/stage5a_candidate_training.json`, `workshop/policies/stage5/stage5a_development.json` |
| `stage5b_closed_loop_viable` |  | B2-v2 is closed-loop viable call-free: it lowers shadow escalation without degrading behaviour. | Stage 5B | `workshop/policy_reports/stage5/stage5b_closed_loop.json`, `workshop/policies/stage5/stage5b_decision.json` |
| `stage5c_mechanism` |  | The Stage-5C provider reduction came mainly from fewer escalation-prone states, not from less repair / retry overhead. | Stage 5C | `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |

### EXPLORATORY

| Claim | Lesson | Statement | Stage | Evidence |
|---|---|---|---|---|
| `call_demand_rises_under_shift` |  | Provider-call demand of Hybrid R2.1 rises under environment shift. | Stage 3 M4D, Stage 4B, Stage 4C | `workshop/policy_reports/phase-a-v2/hybrid_r21_confirmatory_report.json`, `workshop/policy_reports/stage4/stage4b_pilot_report.json`, `workshop/policy_reports/stage4/stage4c_confirmatory_report.json` |

### UNRESOLVED

| Claim | Lesson | Statement | Stage | Evidence |
|---|---|---|---|---|
| `stage5c_level2_state_ni` |  | Hybrid-v2 is non-inferior to Hybrid-v1 on Level-2 state visitation. | Stage 5C | `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `heterogeneity_from_hard_example_coverage` |  | Uneven hard-example representation explains the heterogeneous Stage-5C gains. | Stage 5A, Stage 5C | `workshop/policy_data/stage5a/manifest.json`, `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `hybrid_v1_optimal` |  | Hybrid-v1 / R2.1 is the optimal fidelity / cost operating point. | - | `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `repetition_trigger_needed` |  | The repetition trigger is necessary in R2.1. | - | `workshop/policy_reports/stage4/stage4c_confirmatory_report.json`, `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `per_model_routers` |  | Model- or personality-specific routers would help. | - | `workshop/policy_reports/stage4/stage4c_confirmatory_report.json` |

### NOT SUPPORTED

| Claim | Lesson | Statement | Stage | Evidence |
|---|---|---|---|---|
| `b1_adequate_closed_loop` |  | B1 (table) is an adequate closed-loop surrogate. | Stage 3 M3 | `workshop/policy_reports/phase-a-v2/closed_loop_report.json` |
| `historical_trajectory_metric_robust` |  | The frozen (historical) trajectory metric is robust under environment shift. | Stage 4B.1, Stage 4B.2 | `workshop/policy_reports/stage4/stage4b1_trajectory_diagnostic.json`, `workshop/policy_reports/stage4/stage4b2_metric_robustness.json` |
| `tournament_standings_rank_personalities` |  | The first tournament's standings rank the personalities (e.g. 'sabotage wins'). | first personality tournament | `workshop/tournament_findings.md` |
| `stage5c_level2_state_degraded` |  | Hybrid-v2 degrades Level-2 state visitation. | Stage 5C | `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
| `hybrid_v2_safely_promotable` |  | Hybrid-v2 is safely promotable under the frozen confirmatory rule. | Stage 5C | `workshop/policy_reports/stage5/stage5c_confirmatory_report.json` |
<!-- claims:end -->

## Reproducibility

The project can be checked at two levels, and neither makes a provider call.

**In the public repository.** It is lightweight: the game, the runtime, the frozen policy artifacts the runtime
plays, curated results and example replays. Run:

```bash
python -m tools.validate_public_repo
```

It checks offline that:

- every file of B1, B2-v1, B2-v2, R2.1 and their target and coverage models is byte-identical to the hash recorded in
  the frozen research manifests (`PUBLIC_EXPORT_PROVENANCE.json` lists each one), and every policy loads;
- the game code fingerprint equals the one the research recorded (`sha256:f1a7529308c66f48`);
- every example replay reproduces its recording exactly;
- the runtime runs, scans find no secret or local path, and the test suite passes.

**With the research archive.** The full paper reproduction artifact is distributed separately: it holds the frozen
reports, manifests, registries, bootstrap outputs, datasets and reproduction scripts. It is planned as
`spy-vs-ai-research-artifact-v1` and is not built or published yet. With it:

```bash
python -m workshop.scripts.final_research_release --verify-manifest   # seconds: every frozen artifact hash
python -m workshop.scripts.final_research_release --reproduce         # about 4 minutes: Stages 3-5 recomputed
```

The stored reproduction record reads **REPRODUCED (scientific equality)** in 212.5 s, with 0 provider calls, 0
network attempts, 0 credential reads and 0 of 545 frozen files rewritten. It recomputes:

- **Stage 3:** the M4D point values and the decision;
- **Stage 4:** the Stage-4C point values;
- **Stage 5A:** the selection;
- **Stage 5B:** the full analysis from the 864 matches;
- **Stage 5C:** the decision, every point value, the guards and the decomposition;
- **FINAL-1:** the claims, registries and synthesis.

The official 10,000-resample bootstrap outputs are authoritative and are not recomputed. Their stored outputs, code
hashes and seeds are verified, the decisions are recomputed from the frozen aggregates, and small deterministic smoke
bootstraps check that the code is stable without ever deciding anything. For example, the smoke Level-2 state upper
bound is 0.0165, which would pass the 0.0168 margin. The frozen 10,000-resample bound is 0.0182, which does not, and
the frozen value is the one that counts.
