# The story of Spy vs AI

How a spy game became a study of how much frontier-LLM reasoning an agent needs, and then a playable game again. Each
chapter says why the next one followed. The numbers come from the frozen reports, and
[RESULTS.md](RESULTS.md) lists them with their sources. Terms are defined in the
[glossary](../project/GLOSSARY.md), and the game itself is introduced in [the guide](../README.md).

- **Part I - building the game and its agents:** [1 Game foundations](#1-game-foundations) ·
  [2 Partial information](#2-partial-information) · [3 Memory](#3-memory) ·
  [4 The LLM controller](#4-the-llm-controller) · [5 Providers](#5-provider-abstraction-and-a-first-lesson)
- **Part II - how much LLM does an agent need?** [6 Imitating the LLM](#6-surrogate-policies-can-a-cheap-model-imitate-the-llm) ·
  [7 The closed-loop pathology](#7-the-closed-loop-pathology) ·
  [8 Hybrid routing](#8-hybrid-routing-call-the-llm-only-when-it-matters) ·
  [9 Environment shift](#9-environment-shift-does-it-generalize) ·
  [10 Distillation](#10-hard-example-distillation-can-the-llm-teach-the-imitator)
- **Part III - after the research:** [11 Freezing the research](#11-freezing-the-research) ·
  [12 The playable runtime](#12-the-playable-runtime) · [13 Gameplay polish](#13-gameplay-polish) ·
  [What it adds up to](#what-it-adds-up-to)

## Part I - building the game and its agents

### 1. Game foundations

The game began as a top-down embassy in Python and pygame: a game state, one action pipeline, and a scripted bot that
searched, collected and escaped. It grew to six rooms with doorways, then keys, keycards and safe codes placed from a
seed, a mission clock, victory conditions, and four animated trap types (bomb, electric, snare, alarm). The founding
rule never changed: **the game does not know whether an action came from a keyboard, a bot or a model.** A scripted
bot, a human and later an LLM all drive the same pipeline.

### 2. Partial information

Traps brought hidden state: inspection, seeded detection rolls and per-spy knowledge. An early version of the bot read
the true trap state and was "magically immune" to traps it had never detected. From then on, one function builds what
a spy may know (its own room, the objects it has inspected or searched, its own and detected traps, its inventory,
the opponent last seen), and that observation is the only way a controller sees the world. **The engine owns the
truth; a controller only sees its observation and proposes an action; the engine decides what happens.** That
boundary is what makes a bot, an imitator and an LLM comparable.

### 3. Memory

Spies got exploration memory (which rooms and objects they have seen, what they found where), backtracking and a
strategic trap economy. Later, the LLM controller got decision memory and strategic context. How that history is shown
to a model matters, and it is easy to misread: an apparent "shorter history helps" result turned out to be a
serialization-format confound once the representation was controlled.

### 4. The LLM controller

A model became a controller. It receives a compact serialization of the observation and a prompt with its
personality, and returns one action. That action is checked against a strict contract, repaired if invalid, and
replaced by a safe fallback if still invalid. Benchmarks showed that the action-execution rules had to be fixed
before episodes meant anything: when an action completes, and how a multi-step interaction resumes. The model
proposes; the engine decides.

### 5. Provider abstraction, and a first lesson

The same controller ran on Ollama, then on Amazon Bedrock, with telemetry for every call: latency, tokens, repairs,
fallbacks and errors. Long matches exposed context growth and request pacing, and prompt caching halved the input per
call. LLM-versus-LLM matches and personality experiments followed, and they taught a lesson that shaped everything
after:

```text
MODEL BEHAVIOUR  ≠  AGENT BEHAVIOUR  ≠  MATCH OUTCOME
```

A model can make sensible isolated decisions and still produce a poor agent, because each decision changes the world
it meets next. Match outcomes add the opponent, the seed and the layout on top. The first personality tournament (60
Sonnet 5.5 vs Opus 5.5 matches) is the cautionary tale. Its standings were confounded by the seed and the start rooms:
each seed favoured one side, and the saboteur played most of its mixed games on the favoured side. From then on,
trials counterbalanced sides and compared matched configurations.

## Part II - how much LLM does an agent need?

### 6. Surrogate policies: can a cheap model imitate the LLM?

Every LLM decision costs latency and money. Forty-eight REAL matches, with Claude Sonnet 5.5 and GPT-6.1 Sol on
Bedrock across four personalities on Level 1 (the Phase-A data), produced 4,062 decision records, 3,951 of them clean.
If many decisions are structurally similar, a much cheaper model might imitate them:

- **B1:** an opportunity-conditioned frequency table with backoff;
- **B2:** multinomial logistic regression on 54 model-visible features;
- **B3:** a shallow decision tree, kept for explanation.

Offline, holding out one seed at a time, B2 was the strongest: top-1 accuracy 0.891 and log loss 0.377, against
B1's 0.780 and 0.722. Each surrogate also has a target policy (which object or door a move goes to) and a coverage
label that says when a state is unlike its training data.

### 7. The closed-loop pathology

Put back into the game, the surrogates drive their own trajectories, and they behaved worse than their offline scores
promised. In closed loop they fought and were eliminated far more than the LLMs, they drifted into low-coverage
states, and B1 repeated itself badly. After replicates, B2 was selected over B1 (state-visitation distance to REAL
0.058 vs 0.128), but B2 alone was still not REAL. In later trials it spent 13.7% of its decisions in runs of four or
more identical actions (REAL 5.9%), and 32% on the new levels.

> **Local decision fidelity does not imply closed-loop trajectory fidelity.** Small local differences change the next
> observations, the states visited, the actions available, the encounters and the memory. Errors compound.

### 8. Hybrid routing: call the LLM only when it matters

Could the cheap policy handle the easy decisions, and call the LLM only when a decision looks hard?

```text
Observation → B2 → router ├─ accept B2's action
                          └─ escalate to the REAL LLM
```

Router replays and a provider-backed pilot led to **R2.1**, which escalates on a **repetition loop** or a **strong
B1/B2 disagreement**. An earlier "critical state" trigger cost calls with no local gain and was dropped. On held-out
configurations, R2.1 escalated 13.5% of decisions and passed 10 of its 11 predeclared gates.

Then the first **preregistered** trial: the design, seeds, metrics and decision rule were frozen before any provider
call. On Level 1 (fresh seeds 8001-8004, 32 configurations, REAL against B2 against Hybrid), the Hybrid was confirmed
(decision A). It was closer to REAL than B2 on state visitation (0.066 → 0.034, P 0.999) and on the trajectory metric
(0.640 → 0.487, P 0.983), using 463 provider calls against REAL's 2,650 (17.5%). The LLM became a scarce resource
instead of the controller for every decision.

The trial also produced the project's main infrastructure incident: the AWS session expired 50 s after the preflight,
and 56 matches failed. A recorded amendment archived them, added fail-fast guards and reran exactly those matches.
The data was never rewritten.

### 9. Environment shift: does it generalize?

Does the Hybrid only work in the layout its surrogate learned from? Levels 2 and 3 (nine and ten rooms) changed the
layouts, paths and state distributions. Call-free diagnostics showed the shift plainly: the share of B2's decisions
in low-coverage or out-of-distribution states went from 0.25 on Level 1 to 1.0 and 0.99, and predicted escalation
rose from 0.20 to 0.38-0.39.

A provider-backed pilot then gave a conflict. On Level 3 the Hybrid was closer to REAL on state visitation but
further on the historical trajectory metric (1.139 vs B2's 0.954). Rather than tune the system, the project
questioned the metric. An offline diagnostic found that it mixed counts with rates and compared timing on very thin
support: B2 rarely completed the mission, and its timeouts left no terminal event. A robustness study built
**trajectory-v2** (18 components in 7 families, rates per active minute or per opportunity, explicit timeouts,
symmetric support) and froze it before any ranking and before the next preregistration. The historical metric was
kept as it was, not retroactively replaced.

> When a metric gives a surprising result, first ask whether it measures what you think it measures.

The same pilot showed something subtler. An LLM correction can be locally more REAL-typical and still push the agent
into a future that is globally less REAL-like. Better local actions do not guarantee better global behaviour, which
is why every claim here rests on closed-loop measurement.

The preregistered confirmatory trial on fresh configurations of Levels 2 and 3 then gave **decision A, confirmed
generalization**:

| | B2 | Hybrid R2.1 |
|---|---|---|
| Level 2 state visitation | 0.106 | 0.060 |
| Level 2 trajectory-v2 | 1.112 | 0.628 |
| Level 3 state visitation | 0.101 | 0.048 |
| Level 3 trajectory-v2 | 1.127 | 0.558 |

P(Hybrid closer) was 1.0 in all four cells, with about 36% of REAL's provider calls. That is more than Level 1's
17.5%, because shift raises escalation pressure. Hybrid-v1 / R2.1 became the **CONFIRMED / GENERALIZED RESEARCH
BASELINE**.

### 10. Hard-example distillation: can the LLM teach the imitator?

Every escalation leaves a valuable record: a hard state, the surrogate's proposal and the LLM's decision. Could those
hard examples be distilled back into the surrogate, so that it needs the LLM less?

- **Offline (Stage 5A).** 854 hard examples from earlier escalations were promoted into training under frozen data
  rules, with the confirmatory data kept out. The selected candidate, **B2-v2**, improved development log loss
  (0.441 → 0.290) and lowered the expected escalation rate (0.271 → 0.237). Weighting the hard examples more made
  things worse.
- **Call-free closed loop (Stage 5B, 864 matches).** With R2.1 watching in shadow, B2-v2 triggered the router less
  often (0.358 → 0.292, -18.5%), with fewer timeouts and more completed missions. The gain was uneven: cautious -40%,
  saboteur -57%, opportunist -12%, and racer **+25%**.
- **With real providers (Stage 5C).** The final preregistered trial compared REAL, Hybrid-v1 and Hybrid-v2 on Levels 2
  and 3: 64 configurations after a feasibility amendment, 192 provider-backed matches.

```text
provider calls      REAL 7,504    Hybrid-v1 3,044    Hybrid-v2 2,599
reduction           14.6%  (95% CI 6.8%-22.0%),  P(Hybrid-v2 fewer calls) 0.9998
Hybrid-v2 / REAL    0.346
```

The provider reduction was real. Behavioural non-inferiority passed in three of four cells: trajectory-v2 on both
levels, and state visitation on Level 3. It was not established on Level-2 state visitation: the point difference
was +0.0008, the 95% CI [-0.0171, +0.0182], and the margin 0.0168. There was no measurable degradation, but the
interval was too wide to establish the frozen condition.

> **Decision B - provider reduction is real but Hybrid-v2 is not safely promotable under the frozen confirmatory rule.**

About 69% of the savings came from fewer escalation-prone decisions and about 31% from shorter trajectories; repairs
and retries contributed nothing measurable. B2-v2 changed closed-loop behaviour enough that the router met fewer hard
states. The gains were heterogeneous: provider calls went cautious 802 → 581, opportunist 967 → 913, racer 732 → 809
and saboteur 543 → 296. That pattern is consistent with uneven hard-example coverage across personalities, but the
trial did not establish that cause.

Infrastructure failed once more along the way: a provider service error tripped a fail-fast check in Stage 5C. It was
handled, as before, by a recorded amendment, never by editing results.

## Part III - after the research

### 11. Freezing the research

The research line was then frozen: one release manifest over every artifact, 33 claims with their evidence and status
(lessons A-K), research and incident registries, a final synthesis, and an offline reproduction of every stage that
makes no provider call. The final statuses:

```text
Hybrid-v1 / R2.1   CONFIRMED / GENERALIZED RESEARCH BASELINE          (the default research policy)
Hybrid-v2 / R2.1   MORE EFFICIENT BUT NOT SAFELY PROMOTABLE UNDER THE CONFIRMATORY RULE
MAIN PROVIDER-BACKED RESEARCH LINE CLOSED
```

### 12. The playable runtime

The frozen code was wrapped, never edited, into `python -m spy_vs_ai.play`. A setup screen, the command line or a
YAML file gives each spy its own policy, provider, model and personality. The runtime adds play and research views,
compatibility labels that keep runtime use apart from scientific validation, and offline replays that verify a match
reproduces exactly. A live smoke match (Hybrid-v1 on Sonnet 5.5 against REAL on GPT-6.1 Sol) made 41 Bedrock calls
with no error and replayed identically.

### 13. Gameplay polish

New spy animation frames, floor and wall tiles and decorative props came from an image-generation pipeline, in the
style of the existing sprites. Door traps, five trap variants and richer rooms were added as playable-mode rules, all
downstream of the frozen policies, so the research agents decide exactly as before.

## What it adds up to

- Per-decision imitation is not enough: closed-loop behaviour has to be measured directly.
- A cheap policy can handle a large share of decisions, and selective LLM reasoning recovers much of the missing
  fidelity.
- Environment shift raises escalation pressure.
- The router and the harness are part of the agent's intelligence.
- Hard LLM decisions can teach a cheaper policy, and the improvement can be uneven across behaviours.

The suggested architecture is layered by decision cost: deterministic logic, then a cheap learned policy, then
selective escalation, then frontier reasoning. The guiding question becomes: *what is the cheapest mechanism that
makes this decision well enough in closed loop?*

Natural follow-ups are new research questions, not unfinished requirements of this one: other decision layers,
other environments and harnesses, more balanced active learning, and multi-environment generalization. The findings
with all their numbers, the experiments, the claims and how to check them are in [RESULTS.md](RESULTS.md). The full
study is [the paper](../../paper/SPY_VS_AI_PAPER.md).
