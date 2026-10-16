# Glossary

**REAL** - a spy whose every decision is made by the full LLM controller (no surrogate). The behavioural reference
arm of every trial; the `real_llm` policy in the runtime.

**B1** (`table-v1`) - an opportunity-conditioned frequency table with deterministic backoff, one per research model
and personality. Superseded by B2 in Stage 3; still used inside the hybrids as the second opinion for the
disagreement trigger.

**B2-v1** (`logistic-v1`) - multinomial logistic regression on 54 model-visible features, trained on the Phase-A REAL
decisions. Selected in Stage 3 for closed-loop work; the surrogate inside Hybrid-v1.

**B2-v2** (Stage-5A candidate C1) - B2 retrained on Phase-A data plus 854 promoted hard examples. The surrogate inside
Hybrid-v2.

**B3** (`tree-v1`) - a shallow decision tree evaluated offline in Stage 3, kept for explanation, never deployed.

**Hybrid-v1** - B2-v1 + router R2.1, escalating to the LLM when R2.1 fires. Final status:
CONFIRMED / GENERALIZED RESEARCH BASELINE.

**Hybrid-v2** - B2-v2 + the same R2.1. Final status:
MORE EFFICIENT BUT NOT SAFELY PROMOTABLE UNDER THE CONFIRMATORY RULE.

**R0** - a router that never escalates. A hybrid with R0 is the call-free control arm: pure surrogate play through
the same controller.

**R2.1** - the frozen escalation router: escalate on a **repetition loop** OR a **strong B1/B2 disagreement**.
Coverage and critical-state signals are logged as diagnostics only.

**Escalation** - a hybrid decision for which the router calls the LLM instead of accepting the surrogate's action.

**Disagreement trigger** - strong disagreement between B1's and B2's proposals. The more useful local escalation
signal.

**Repetition trigger** - a loop signal (the surrogate repeating the same decision). Strong evidence of pathology,
weaker as a local correction signal.

**Coverage** - how well a state is represented in the surrogate's training data (`coverage-v1`: HIGH, MEDIUM, LOW,
OOD; a heuristic label, not a probability).

**LOW/OOD** - a low-coverage or out-of-distribution state (OOD: a core pattern never seen in training).

**State visitation** - the distribution of states (rooms, phases, situations) an agent visits in closed loop; compared
to REAL by total variation distance (TVD).

**Trajectory metric** (historical) - the standardized trajectory distance used in Stages 3-4B; kept as the
historical, authoritative metric of those stages.

**Trajectory-v2** - the trajectory metric built after the Stage-4B diagnostics (18 components in 7 families,
behaviour per active minute or per opportunity, explicit timeouts, symmetric support), frozen before any ranking
and before the Stage-4C preregistration.

**Opportunity rate** - how often a behaviour happens given an opportunity for it (e.g. P(search | search possible)).

**Closed loop** - the agent's own actions change the world, so they change its future observations. Agents are
evaluated this way.

**Open-loop replay** - evaluating a policy on recorded states without letting it change the future (logged-state
replay). Fast, but blind to compounding errors.

**Hard-example distillation** - using LLM decisions at hard (escalated) states as extra training data for the cheaper
surrogate. Stage 5.

**Matched configuration** - one experimental unit shared by every arm: level, seed, personality pairing and
orientation (which spy plays which side). Comparisons and bootstraps are over configurations.

**Orientation** - which identity plays Black and which White (A / B); counterbalanced because seeds favour a side.

**Bootstrap** - resampling configurations (10,000 resamples in the frozen analyses) to get confidence intervals and
probabilities such as P(Hybrid closer).

**Non-inferiority** - showing a new system is not worse than the reference by more than a frozen margin: the upper
confidence bound of the difference must lie below the margin.

**Preregistration** - the design, seeds, metrics, margins and decision rule frozen (hashed) before any provider call
of a trial.

**Amendment** - a recorded, hashed change to a preregistered design (e.g. M4D infrastructure recovery, Stage-5C
Amendments 01 / 02), made without reading outcomes.

**Provider call** - one request to a model provider (counted per spy, per match and per arm in a shared ledger).

**Physical call** - an actual request sent, including repairs and transport retries; one decision can cost several.

**Repair** - a follow-up request when the model's output does not satisfy the structured action contract.

**Fallback** - a safe default action committed when the normal model / validation path fails.

**Policy** - the decision architecture: scripted, B1, B2, Hybrid or REAL LLM.

**Provider** - where inference runs: Bedrock, OpenAI direct, Anthropic direct, Ollama (or the offline fake).

**Model** - the provider's own model id.

**Personality** - a behavioural prompt / context dimension: cautious, opportunist, racer, saboteur. It also selects
the surrogate files.

**Surrogate identity** - which research model's decisions a surrogate imitates (`sonnet55`, `gpt61sol`).

**Evaluation mode** - runtime execution semantics matching the frozen protocols (the world stops while a model
thinks, game-time mission clock, seeded commit order).

**Playable mode** - smooth live play (models decide while the world moves, wall-clock timer, enhanced gameplay).
Not scientifically equivalent to evaluation mode.

**Enhanced gameplay** - playable-runtime rules downstream of the frozen policies: door traps, trap variants and
decorative room props. Never part of Stages 3-5.

**Decision A / B** - outcomes of a frozen decision rule. Stage 4C A: confirmed generalization. Stage 5C B: efficiency
improves but a fidelity / subgroup guard or full integrity fails - provider reduction is real but not safely
promotable.

**FINAL-1** - the final scientific freeze and release (manifest, claims, registries, synthesis, offline
reproduction). **FINAL-2** - the playable multi-policy / multi-provider runtime built on top of it.
