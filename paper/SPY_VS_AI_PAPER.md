# Spy vs AI: How Much Frontier-LLM Reasoning Does an Agent Actually Need?

> Public repository note: the reproduction command of Appendix D runs in the frozen research archive, which is distributed separately; what this repository verifies is described in [docs/research/RESULTS.md](../docs/research/RESULTS.md#reproducibility).


## A Closed-Loop Study of Behavioral Surrogates, Selective Escalation, Environment Shift, and Hard-Example Distillation

**Status:** Final narrative draft  
**Research line:** Main provider-backed research line closed  
**Primary scientific baseline:** Hybrid-v1 / R2.1  
**Efficient candidate:** Hybrid-v2 / R2.1  
**Final confirmatory decision:** B — provider reduction confirmed, safe promotion not established

---

## Abstract

Large language models are increasingly used as the decision-making core of autonomous agents, but many agent architectures implicitly assume that every consequential decision should be delegated to a frontier model. This work asks a different question: **how much frontier-model reasoning does an autonomous agent actually need?**

We study this question in *Spy vs AI*, a partially observable multi-agent game in which autonomous spies move between rooms, inspect and search objects, collect mission items, remember previously observed state, place and disarm traps, encounter opponents, and attempt to complete a mission and escape. The environment provides a closed-loop setting in which each action changes the future state distribution and therefore the agent's future observations.

We first construct cheap behavioral surrogates of LLM decisions. A multinomial logistic surrogate, B2, achieves strong offline predictive performance, but closed-loop evaluation reveals substantial behavioral pathologies, including increased repetition and trajectory drift. This establishes an important distinction: **per-decision imitation does not imply agent-level behavioral fidelity**.

We then introduce a Hybrid architecture in which the surrogate handles ordinary decisions while a frozen router selectively escalates difficult states to a frontier LLM. The most useful escalation signal is strong disagreement between two cheap policies rather than simple repetition alone. In confirmatory shifted environments, Hybrid behavior is substantially closer to REAL LLM behavior than the pure surrogate while using only a fraction of the provider calls.

Finally, we test whether the difficult decisions that required LLM escalation can themselves be distilled back into the surrogate. The resulting B2-v2 candidate reduces real provider demand by **14.6%** relative to the previous Hybrid system, with **P(Hybrid-v2 uses fewer calls) = 0.9998**. Behavioral non-inferiority is established on three of four preregistered confirmatory checks. The remaining Level-2 state-visitation check is inconclusive rather than showing a meaningful point degradation: the estimated difference is **+0.0008**, while the 95% upper confidence bound (**0.0182**) narrowly exceeds the frozen non-inferiority margin (**0.0168**). Under the preregistered rule, the final decision is therefore **B: the provider reduction is real, but Hybrid-v2 is not safely promotable**.

The results suggest that agent intelligence can be decomposed into layers of increasing cost. Frontier reasoning need not be the default for every decision; instead, it can be treated as a selectively allocated resource. At the same time, the study shows that agent behavior depends on far more than the underlying model: observation design, memory, action schemas, routing, fallback behavior, runtime semantics, and trajectory history all materially shape the resulting agent.

---

# 1. Introduction

The dominant pattern in many LLM-based agent systems is simple: observe the current state, send the observation to a large model, parse the answer, execute the resulting action, and repeat. This architecture is attractive because a sufficiently capable model can often reason over many heterogeneous situations without task-specific training.

It is also expensive.

The cost is not only monetary. Frontier-model inference introduces latency, provider dependence, failure modes, rate limits, operational complexity, and a tendency to treat every decision as if it required the same level of reasoning.

Many agent decisions are not equally difficult.

A move through a known corridor, a search of an already-inspected object, or a repeated local tactical choice may not require the same level of reasoning as interpreting a novel state, recovering from an unexpected lock, responding to an opponent, or deciding how to resolve conflicting objectives.

This motivates the central question of this work:

> **How much frontier-LLM reasoning does an autonomous agent actually need?**

Rather than treating the LLM as the mandatory controller for every decision, we investigate whether agent decision-making can be decomposed into multiple layers:

```text
cheap deterministic or learned policy
        ↓
selective routing
        ↓
frontier reasoning only when necessary
```

The project evolved through four increasingly difficult questions:

1. **Can a cheap policy imitate LLM decisions?**
2. **Does that imitation survive closed-loop execution?**
3. **Can a router selectively recover the missing fidelity with fewer LLM calls?**
4. **Can hard LLM decisions be distilled back into the cheap policy to reduce future LLM dependence?**

The main contribution is not a single model or router. It is the experimental evidence that these questions must be answered at the **agent level**, in closed loop.

A recurring result throughout the project is:

```text
MODEL BEHAVIOR
≠
AGENT BEHAVIOR
≠
MATCH OUTCOME
```

A policy can predict isolated actions well and still produce a very different agent trajectory. A locally better action can push the agent into a future state distribution that produces globally worse behavior. A more capable model can behave differently depending on the observation contract, memory, action schema, routing logic, fallback behavior, and runtime.

The project therefore became a study not only of LLMs, but of the **allocation of reasoning inside agent systems**.

---

# 2. Contributions

This work makes five main contributions.

**First**, it provides a closed-loop environment in which scripted, surrogate, Hybrid, and REAL LLM policies can be compared under a shared engine and action/observation contract.

**Second**, it shows empirically that strong offline next-action imitation is insufficient for agent fidelity: the selected surrogate predicts individual actions well but develops closed-loop state-distribution drift and repetition pathologies.

**Third**, it validates a selective-escalation architecture in which a cheap surrogate handles most decisions while difficult states are routed to a frontier model.

**Fourth**, it confirms that the selective architecture generalizes under tested environment shift, where the surrogate encounters substantially more LOW/OOD states.

**Fifth**, it demonstrates that historical hard LLM decisions can be distilled back into the surrogate to reduce future provider demand, while also showing that this benefit is heterogeneous and not sufficient for unconditional safe promotion.

---

# 4. Research Questions

We organize the study around five research questions.

## RQ1 — Can a cheap surrogate reproduce frontier-LLM decisions?

We ask whether a lightweight policy trained from historical LLM decisions can predict the next action sufficiently well to replace a frontier model for many decisions.

## RQ2 — Does local decision fidelity translate into closed-loop behavioral fidelity?

We test whether a surrogate that performs well on held-out decisions also produces similar state visitation, trajectories, mission progression, and failure modes when its own actions determine its future observations.

## RQ3 — Can selective escalation restore fidelity while reducing provider use?

We introduce a router that allows the cheap surrogate to act normally but escalates selected difficult states to a frontier LLM.

## RQ4 — Does selective escalation generalize under environment shift?

We evaluate the same frozen architecture on new levels with substantially different state distributions and higher LOW/OOD exposure.

## RQ5 — Can hard LLM decisions teach the surrogate and reduce future provider dependence?

We promote selected historical escalation cases into a new surrogate candidate, then test whether this reduces real provider calls without unacceptable behavioral degradation.

---

# 4. The Spy vs AI Environment

## 3.1 Game setting

*Spy vs AI* is a top-down multi-room environment inspired by classic competitive spy games but implemented as an original research environment.

Two autonomous spies operate in the same world.

Agents can perform actions including:

```text
MOVE
INSPECT
SEARCH
PLACE_TRAP
DISARM
COLLECT
APPROACH
AVOID
WAIT
EXIT
```

The world contains multiple rooms, doors, locks, keys, searchable objects, mission-relevant objects, traps, partial information, opponent encounters, persistent memory, mission objectives, and terminal outcomes such as escape, timeout, elimination, or incomplete mission state.

## 3.2 Partial observability

The controller does not receive the complete game state.

The engine owns world truth:

```text
GameState / world truth
        ↓
ObservationBuilder(actor_id)
        ↓
Observation
        ↓
Controller
        ↓
GameAction
        ↓
Engine
```

Observations contain actor-specific information such as visited rooms, known locks, visible objects, detected traps, mission progress, inventory, and the last seen opponent position.

## 3.3 Memory

Later versions of the environment introduced persistent memory including visited rooms, known locked objects, room-based visibility, last-seen opponent information with timestamps, and known mission-relevant information.

Memory is part of the agent harness and therefore part of the observed agent behavior.

---

# 5. Agent Architecture

## 4.1 REAL LLM controller

The REAL controller converts the current observation into a structured provider request.

Conceptually:

```text
Observation
   ↓
serialization
   ↓
prompt + personality
   ↓
provider
   ↓
model
   ↓
structured output
   ↓
validation
   ↓
repair if needed
   ↓
fallback if needed
   ↓
GameAction
```

The controller contract includes serialization, validation, provider normalization, invalid-output repair, fallback, telemetry, decision identifiers, latency, and token accounting.

The LLM never directly modifies world state.

## 4.2 Personalities

The main research line uses four behavioral personalities:

```text
cautious
opportunist
racer
saboteur
```

## 4.3 Provider/model identities

The main provider-backed experiments used Amazon Bedrock in `us-east-1`, including two model families:

```text
Anthropic Claude Sonnet 5.5
OpenAI GPT-6.1 Sol through Bedrock
```

The playable runtime later supports other provider paths, but runtime compatibility should not be confused with scientific evaluation.

---

# 6. Why Offline Accuracy Was Not Enough

The first major research direction was behavioral imitation.

Historical REAL decisions were collected and used to train lightweight surrogate policies.

Three main candidates were evaluated:

- **B1** — empirical/opportunity-conditioned policy;
- **B2** — multinomial logistic policy;
- **B3** — shallow CART policy.

Representative offline results favored B2:

```text
top-1 accuracy: 0.891
macro F1:       0.557
log loss:       0.377
Brier score:    0.168
```

B1 remained useful because it captured a different behavioral representation and later became valuable as a disagreement reference.

At this point, the natural conclusion might have been that B2 could simply replace the LLM.

Closed-loop evaluation showed otherwise.

---

# 7. Closed-Loop Failure Modes

## 6.1 The compounding-error problem

Open-loop evaluation asks:

> Given a state drawn from historical REAL behavior, what action would the surrogate choose?

Closed-loop evaluation asks:

> What happens after the surrogate's own actions change the next state?

These are fundamentally different questions.

A local decision changes the next room, which objects become visible, which mission state becomes reachable, whether the opponent is encountered, which memories are updated, which actions become legal, and which future decisions are difficult.

Therefore local errors compound.

## 6.2 Repetition pathology

Pure surrogate behavior showed substantially more long repetition sequences than REAL.

This became one of the clearest examples of a policy that looked good offline but behaved differently as an agent.

## 6.3 Trajectory drift

The surrogate also produced different state visitation and different trajectory statistics.

This yielded the first central conclusion:

> **Strong per-decision prediction does not guarantee REAL-like closed-loop agent behavior.**

---

# 8. Selective Escalation

Instead of discarding the surrogate, we changed the problem.

The surrogate would remain the default controller, but the system would invoke a frontier LLM only in selected difficult states.

The resulting architecture was:

```text
Observation
   ↓
B2
   ↓
R2.1 router
  /   \
no     yes
|       |
B2      REAL LLM
|       |
└── final action
```

The router evolved through replay and provider-backed pilot experiments.

Two important signals were repetition and strong B1/B2 disagreement.

The later experiments showed that disagreement was the more useful local escalation trigger.

---

# 9. Stage 3 — Establishing the Hybrid Baseline

Stage 3 compared REAL, B2 and Hybrid closed-loop behavior.

The selected B2 model remained strong at individual prediction, but pure B2 developed significantly more closed-loop pathology.

In the final Stage-3 comparison, Hybrid improved multiple behavioral metrics relative to B2, including state visitation, trajectory distance, outcome distance, LOW/OOD behavior, and repetition loops.

Provider usage was much lower than REAL.

The key result was not that B2 was poor. It was that B2 handled many states cheaply, frontier reasoning remained useful for selected difficult states, and the Hybrid architecture offered a better cost/fidelity trade-off.

---

# 10. Stage 4 — Generalization Under Environment Shift

## 9.1 Why environment shift mattered

A surrogate can look effective if evaluated mainly inside the state distribution from which it learned.

Stage 4 introduced new levels with different room structures and state distributions.

The shifted environments dramatically increased LOW/OOD exposure.

The question became:

> Does the frozen Hybrid architecture still help when the world changes?

## 9.2 Initial mixed result

The first shifted-environment evaluation produced a mixed result.

Hybrid improved state-visitation fidelity relative to B2, but on Level 3 it appeared worse on the historical trajectory metric.

Rather than immediately tuning the router, we performed an offline trajectory diagnostic.

This revealed that the historical trajectory metric mixed raw counts, timing, missing events, and incomplete support.

## 9.3 Trajectory-v2

A new candidate trajectory metric was therefore designed.

It introduced more robust treatment of event attainment, timing support, active time, exploration, opportunity-normalized behavior, and redundant component families.

The historical metric remained authoritative for earlier frozen stages.

Trajectory-v2 was frozen before fresh confirmatory use.

This was an important methodological constraint: the metric was improved, but history was not rewritten.

---

# 11. Stage 4C — Confirmed Generalization

Stage 4C ran a fresh provider-backed confirmatory trial on Levels 2 and 3.

Hybrid-v1 significantly improved both primary measures relative to B2-v1.

## State-visitation TVD

```text
Level 2
B2:     0.1059
Hybrid: 0.0600

Level 3
B2:     0.1008
Hybrid: 0.0483
```

## Trajectory-v2 distance

```text
Level 2
B2:     1.112
Hybrid: 0.628

Level 3
B2:     1.127
Hybrid: 0.558
```

Provider efficiency remained substantial:

```text
Hybrid / REAL provider calls:
~0.361 pooled
```

Hybrid also reduced long repetition pathology relative to pure B2.

Stage 4C therefore established:

> **Selective frontier reasoning generalized under the tested environment shift and preserved substantially more REAL-like behavior than the pure surrogate.**

Hybrid-v1 / R2.1 became the confirmed generalized research baseline.

---

# 12. Local Corrections Can Still Hurt Globally

The Stage-4 diagnostic work revealed a subtle but important result.

On disagreement-triggered escalations, many LLM corrections changed the action toward a locally more REAL-typical choice.

Yet those local corrections did not always produce a globally more REAL-like trajectory.

Why?

Because a different action changes the future state distribution.

Therefore:

> **A locally REAL-typical decision does not guarantee a globally REAL-like trajectory.**

This is another reason why agent evaluation cannot be reduced to isolated next-action accuracy.

---

# 13. Stage 5 — Learning From Hard Decisions

Once selective escalation worked, we asked whether the LLM's hard decisions could themselves improve the surrogate.

Every escalation produces a useful tuple:

```text
state
surrogate proposal
router reason
REAL LLM decision
```

Stage 5 promoted selected historical hard examples into a new B2 candidate while protecting confirmatory data from training.

---

# 14. Stage 5A — Offline Retraining

Several retraining strategies were evaluated.

The selected candidate, C1 / B2-v2, added historical hard examples without extra weighting.

Representative development results:

```text
             logloss   top1    macroF1   Brier
B2-v1        0.4412    0.8521  0.5994    0.2184
B2-v2/C1     0.2896    0.9002  0.6395    0.1467
```

Expected escalation also fell.

This supported the hypothesis that hard-example distillation could reduce future LLM dependence.

However, this remained open-loop evidence.

---

# 15. Stage 5B — Closed-Loop, Call-Free Selection

B2-v2 was next run closed-loop without real provider calls.

The frozen R2.1 router was evaluated in shadow.

Pooled shadow escalation decreased from approximately:

```text
B2-v1: 0.3575
B2-v2: 0.2915
```

or about:

```text
18.5% relative reduction
```

Repetition improved.

Historical-reference state and trajectory diagnostics also improved overall.

But the improvement was not uniform.

The `racer` personality regressed in escalation demand, while `cautious` and `saboteur` improved strongly.

This heterogeneity became an important Stage-5 warning.

---

# 16. Stage 5C — Real Provider Confirmation

Stage 5C was the final provider-backed experiment in the main research line.

The confirmatory comparison included:

```text
REAL
Hybrid-v1 = B2-v1 + frozen R2.1
Hybrid-v2 = B2-v2 + frozen R2.1
```

on shifted Levels 2 and 3.

The reduced preregistered design contained 64 matched configurations and 192 provider-backed matches.

All included configurations completed successfully.

## 15.1 Provider efficiency

Observed provider calls:

```text
REAL:       7504
Hybrid-v1:  3044
Hybrid-v2:  2599
```

Hybrid-v2 therefore reduced calls relative to Hybrid-v1 by:

```text
14.6%
```

with:

```text
95% CI:
6.8% to 22.0%

P(Hybrid-v2 fewer calls):
0.9998
```

Hybrid-v2 used approximately:

```text
34.6% of REAL provider calls
```

The provider-efficiency result was therefore strongly supported.

## 15.2 Behavioral non-inferiority

Trajectory-v2 non-inferiority passed on both levels.

State-visitation non-inferiority passed on Level 3.

The Level-2 state-visitation cell was the only failed confirmatory criterion.

For that cell:

```text
D(Hybrid-v1, REAL): 0.0565
D(Hybrid-v2, REAL): 0.0573

difference:
+0.0008

95% CI:
[-0.0171, +0.0182]

frozen non-inferiority margin:
0.0168
```

The measured point difference was essentially zero.

However, the upper confidence bound exceeded the preregistered margin by:

```text
0.0014
```

Therefore non-inferiority was not established for that cell.

Under the frozen rule, the final decision was:

> **Decision B — the provider reduction is real, but Hybrid-v2 is not safely promotable.**

This is not equivalent to saying that Hybrid-v2 showed meaningful behavioral degradation.

It means the confirmatory design did not establish every required non-inferiority criterion.

---

# 17. Where the Stage-5C Savings Came From

The provider reduction was decomposed mechanistically.

Approximately:

```text
69%
fewer escalation-prone decisions

31%
shorter trajectories

~0%
repair/retry changes
```

This result matters because it shows that B2-v2 did more than reduce provider overhead.

It changed the closed-loop state/action distribution such that the router encountered fewer states requiring frontier reasoning.

---

# 18. Heterogeneous Distillation Effects

One of the most important practical findings was that B2-v2 did not help every behavioral identity equally.

Provider calls by personality changed approximately as follows:

```text
cautious
802 → 581

opportunist
967 → 913

racer
732 → 809

saboteur
543 → 296
```

The same aggregate system therefore contained both large efficiency gains, almost no gain, and one subgroup with increased provider demand.

This is a meaningful weakness of naive hard-example distillation.

The training data itself was unevenly distributed across personalities, with many more hard examples for cautious and saboteur than racer.

The observed subgroup pattern is consistent with this imbalance, but the study was not designed to establish that causal mechanism.

The appropriate conclusion is:

> **Hard-example distillation can reduce provider demand, but its benefits are not uniformly distributed across agent behavioral regimes.**

This result should not be hidden behind the pooled average.

---

# 19. Model-Level Heterogeneity

Provider-call changes also differed across model families. Approximately:

```text
GPT family:
1346 → 969
~28% reduction

Sonnet family:
1698 → 1630
~4% reduction
```

Some model/personality identities increased call demand. This suggests that hard-example distillation should not be treated as universally beneficial across identities even when the pooled result is positive.

---

# 26. The Agent Harness Is Part of the Intelligence

Throughout the project, observed behavior depended not only on the underlying model but also on:

```text
observation
representation
memory
prompt/personality
action schema
router
repair
fallback
scheduling
execution semantics
environment history
```

This has a broader implication for agent benchmarking.

If two systems use the same LLM with different harnesses, they may behave very differently.

Conversely, two models tested in different agent runtimes are not being compared in isolation.

A useful abstraction is:

```text
agent behavior
=
model
+ harness
+ environment history
```

---

# 20. Negative Results and Methodological Lessons

The project accumulated several results that are important precisely because they did not support the initial simplifications.

## 19.1 Early standings were confounded

Early tournament-style win/loss standings were affected by seed/spawn structure and could not be treated as clean evidence of personality or model superiority.

## 19.2 Offline predictive performance overstated agent reliability

B2 looked strong offline but developed closed-loop loop/pathology behavior.

## 19.3 Repetition alone was not a sufficient routing signal

Repetition was useful as a symptom of failure, but strong B1/B2 disagreement was a better local escalation signal.

## 19.4 Metrics can create false confidence

The historical trajectory metric produced a result that triggered deeper investigation.

The correct response was not to tune the system until the metric agreed, but to audit the metric and preregister a replacement for future confirmation.

## 19.5 More REAL-like locally does not mean more REAL-like globally

Closed-loop path dependence means that local correction quality and global trajectory quality are not interchangeable.

## 19.6 Distillation does not improve every subgroup

The pooled efficiency result can hide meaningful personality/identity heterogeneity.

---

# 21. Final Scientific Status

## Hybrid-v1 / R2.1

```text
CONFIRMED / GENERALIZED RESEARCH BASELINE
```

Hybrid-v1 is the conservative scientific reference.

## Hybrid-v2 / R2.1

```text
MORE EFFICIENT
BUT NOT SAFELY PROMOTABLE UNDER THE FROZEN CONFIRMATORY RULE
```

Hybrid-v2 remains useful and runnable, but it does not silently replace Hybrid-v1 as the confirmed research baseline.

---

# 22. Threats to Validity

## 21.1 Single environment family

The study uses one game family.

The findings may not generalize to software agents, robotics, web navigation, or real-world operational tasks.

## 21.2 Limited shifted environments

Generalization was tested on a small number of shifted levels rather than a broad family of procedurally generated environments.

## 21.3 Limited model families

The provider-backed confirmatory work focused on two model families through Bedrock.

## 21.4 Personality-specific data imbalance

Hard-example data was unevenly distributed across behavioral identities.

This limits conclusions about why some personalities benefited more than others.

## 21.5 Metric evolution

Trajectory-v2 was introduced after limitations were discovered in the historical metric.

The project preserved historical results and preregistered the newer metric before confirmatory use, but the evolution itself is a methodological dependency.

## 21.6 Reduced Stage-5C sample size

Stage 5C was deliberately reduced for feasibility before scientific outcome inspection.

This reduced precision, especially for the Level-2 state-visitation non-inferiority cell.

## 21.7 Provider/runtime effects

Provider errors, retries, repair, latency, and fallback can affect closed-loop behavior.

These are part of the deployed agent system, but they complicate attempts to attribute behavior exclusively to the model.

---

# 23. Practical Implications

The strongest practical implication is not that every agent should use this exact B2/R2.1 architecture.

It is the more general design principle:

> **Use the cheapest decision mechanism that is sufficiently reliable for the current state, and reserve frontier reasoning for states where its expected value is high enough to justify the cost.**

Possible layers include:

```text
rules
small classifier
surrogate policy
decision-specific model
frontier LLM
```

The right architecture depends on cost, latency, risk, distribution shift, state novelty, and consequence of error.

---

# 24. Related Work Positioning

This paper sits at the intersection of LLM agents, agent orchestration, model routing, imitation/distillation, active learning, selective prediction, closed-loop evaluation, and distribution shift. Its distinctive emphasis is not simply on routing prompts to cheaper models, but on **closed-loop behavioral fidelity under selective reasoning**.

A final submission should include a verified literature review that explicitly distinguishes:

```text
model routing
from
agent-action routing

offline imitation
from
closed-loop behavioral fidelity

cost optimization
from
behavior-preserving reasoning allocation
```

No external literature claims are inserted here without verification.

---

# 25. Future Work

## 23.1 Intermediate decision models

A natural next step is to insert a bounded decision model between B2-v2 and the frontier LLM.

Candidates include systems such as:

```text
CLM
Jev
other contrastive/ranking models
```

The question would be:

> Can a smaller decision-specific model resolve the states that B2-v2 still finds difficult, leaving only the hardest cases for the frontier LLM?

## 23.2 Generalization to another environment family

The strongest extension would not be simply a fourth level.

It would be a second environment with the same agent/runtime abstractions.

If the same local-vs-closed-loop and selective-escalation phenomena reappear, the external validity of the findings would increase substantially.

## 23.3 Harness isolation

Future work could hold the LLM constant while varying memory, observation compaction, action schema, router, repair, and scheduling.

This would directly measure how much observed "agent intelligence" comes from the harness rather than the model.

## 23.4 Value-aware routing

R2.1 routes based mainly on behavioral signals.

A future router could estimate:

```text
probability surrogate is wrong
×
cost of being wrong
×
value of frontier correction
```

This would turn escalation into an explicit value-of-reasoning decision.

## 23.5 Balanced active learning

The heterogeneous Stage-5 result suggests that future hard-example promotion should consider behavioral coverage rather than simply aggregating difficult examples globally.

---

# 26. Conclusion

Spy vs AI began as a game controlled by large language models.

It ended as an investigation into how much of an agent actually needs a large language model.

The project produced four major lessons.

First, **offline action prediction is not enough**. Agent behavior is closed-loop, path-dependent, and sensitive to compounding local differences.

Second, **selective frontier reasoning works**. A cheap surrogate can handle many decisions while a router reserves expensive reasoning for selected difficult states.

Third, **generalization matters**. Environment shift increases surrogate uncertainty and escalation demand, but the Hybrid architecture remained substantially more REAL-like than the pure surrogate in the tested shifted levels.

Fourth, **hard examples can teach the surrogate**, reducing real provider demand further. In Stage 5C, Hybrid-v2 reduced provider calls by 14.6% relative to Hybrid-v1. Yet the benefit was heterogeneous, and one preregistered non-inferiority criterion remained inconclusive. The correct scientific conclusion was therefore not "fully validated" and not "failed", but:

> **More efficient, with real provider savings, but not safely promotable under the frozen confirmatory rule.**

The broader implication is that autonomous-agent design should not begin with the assumption that the strongest available model must make every decision.

A better question is:

> **Which decisions actually require frontier reasoning, and what is the cheapest mechanism capable of handling the rest without degrading the agent in closed loop?**

That question is likely to remain central as agent systems become more capable, more expensive, and more deeply embedded in real workflows.

---

# Appendix A — Research Progression

```text
Game foundations
        ↓
REAL LLM controller
        ↓
behavioral data
        ↓
B1 / B2 / B3
        ↓
B2 selected offline
        ↓
closed-loop pathology
        ↓
Hybrid / R2.1
        ↓
Stage 3 Hybrid baseline
        ↓
environment shift
        ↓
Stage 4 diagnostics
        ↓
trajectory-v2
        ↓
Stage 4C confirmed generalization
        ↓
hard-example promotion
        ↓
B2-v2
        ↓
Stage 5A offline selection
        ↓
Stage 5B closed-loop call-free selection
        ↓
Stage 5C real provider confirmation
        ↓
Decision B
        ↓
Final scientific freeze
```

---

# Appendix B — Final Key Numbers

## Stage 4C

```text
State visitation TVD

Level 2:
B2-v1     0.1059
Hybrid-v1 0.0600

Level 3:
B2-v1     0.1008
Hybrid-v1 0.0483
```

```text
Trajectory-v2 distance

Level 2:
B2-v1     1.112
Hybrid-v1 0.628

Level 3:
B2-v1     1.127
Hybrid-v1 0.558
```

```text
Hybrid-v1 / REAL provider-call ratio:
~0.361 pooled
```

## Stage 5C

```text
REAL calls:       7504
Hybrid-v1 calls:  3044
Hybrid-v2 calls:  2599

Hybrid-v2 vs Hybrid-v1 reduction:
14.6%

95% CI:
6.8%–22.0%

P(Hybrid-v2 fewer):
0.9998

Hybrid-v2 / REAL:
0.3463
```

Level-2 state-visitation non-inferiority:

```text
Hybrid-v1 distance: 0.0565
Hybrid-v2 distance: 0.0573
Delta:              +0.0008
95% CI:              [-0.0171, +0.0182]
margin:              0.0168
verdict:             FAIL under frozen rule
```

---

# Appendix C — Personality Heterogeneity in Stage 5C

```text
cautious:
Hybrid-v1 802
Hybrid-v2 581

opportunist:
Hybrid-v1 967
Hybrid-v2 913

racer:
Hybrid-v1 732
Hybrid-v2 809

saboteur:
Hybrid-v1 543
Hybrid-v2 296
```

Interpretation:

> Aggregate provider savings should not be interpreted as uniform improvement across behavioral identities.

---

# Appendix D — Reproducibility

The final frozen research release is reproduced offline with:

```bash
python -m workshop.scripts.final_research_release --reproduce
```

The final release verifies frozen official outputs and performs deterministic offline checks without provider calls.

The repository's frozen research archive remains the authoritative source for exact protocol hashes, amendments, data registries, and report paths.

---

# Appendix E — Draft Figure Plan

The final paper should include at least:

1. System architecture evolution: REAL → B2 → Hybrid-v1 → Hybrid-v2.
2. Local vs closed-loop fidelity.
3. Stage-4 generalization.
4. Provider-call trade-off.
5. Stage-5C non-inferiority.
6. Personality heterogeneity.
7. Mechanism of call reduction.
8. Project timeline.

---

# Appendix F — Paper Status

This is a **full narrative draft**, not yet a publication-formatted submission.

Before publication, the repository should supply:

- exact figure assets;
- formal references to frozen internal reports/manifests;
- author list and affiliations;
- target venue formatting;
- bibliography for related work;
- final reproducibility appendix links.

---
# Appendix G — Publication Completion Checklist

Before submission:

```text
add verified bibliography
add author list/affiliations
generate final figures from frozen reports
map each result table to authoritative artifact
select venue/template
format citations
add artifact/reproducibility statement
proofread terminology against frozen docs
```
