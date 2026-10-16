# Architecture

How Spy vs AI is built: the engine and what a spy may see, the controllers that decide (bot, imitators, Hybrid,
LLM), the playable runtime around them, and the replay format. The game and the players are introduced in
[the guide](../README.md).

## Core principle: the engine owns the truth, controllers see observations

```text
GameState (src/state.py)                       world truth: positions, objects, traps, inventories, clocks
   │
   ▼
build_observation(state, actor) (src/observation.py)   what this spy may know: partial information
   │
   ▼
Observation                                    own room, visible objects, known locks, visited rooms, own and
   │                                            detected traps, inventory, mission progress, last-seen opponent,
   ▼                                            legal actions
Controller / policy                            scripted bot, surrogate, hybrid or LLM controller
   │
   ▼
GameAction (src/actions.py)                    a proposal: MOVE, INTERACT, INSPECT, PLACE_TRAP, DISARM, WAIT, ...
   │
   ▼
Engine (GameState.apply_action / update)       validates and applies; decides what actually happened
   │
   ▼
Renderer / telemetry / replay
```

A controller never reads `GameState`. An enemy trap it has not detected is not in its observation; an object's
contents are unknown until it is searched. The engine is the only authority: a model proposes, the engine disposes.
This is what let scripted bots, surrogates and different LLM providers be compared on equal terms.

## Actions

Engine actions (`ActionType`): `move`, `interact` (search furniture, open a door, try the exit), `inspect`,
`place_trap`, `disarm`, `wait`, `send_message`. The policy vocabulary the surrogates were trained on
(`workshop/policy_emulator/schema.py ACTIONS`) is `wait`, `move`, `interact`, `inspect`, `place_trap`, `disarm`,
`approach_opponent`, `avoid_opponent`, `enter_safe_code`: searching is `interact` on furniture, leaving is a `move`
/ `interact` to the exit, and picking up items happens as part of a search.

One decision becomes many frames: the action lifecycle (`src/agents/action_lifecycle.py`) turns a decided "move to
the desk" into per-frame movement along a path and reports how it ended (arrived, interrupted, stalled, unreachable).
Decisions are asked for at those boundaries, not every frame.

## Controllers

| Controller | Code | Decides with |
|---|---|---|
| Scripted bot | `src/bot_controller.py` `ScriptedBotController` | fixed priorities: disarm, inspect, search, trap, exit |
| LLM controller (REAL) | `src/llm_controller.py` `LLMController` | a model, through a provider client |
| Surrogate (B1 / B2) | `workshop/policy_emulator/surrogate.py` `SurrogateController` | a frozen policy file, no model |
| Hybrid | `workshop/policy_emulator/hybrid.py` `HybridController` | B2, and the LLM controller when the router escalates |

### The REAL LLM controller

```text
Observation → serialization (compact JSON payload) → prompt with personality → provider adapter → model
            → structured output → validation → repair if invalid → fallback if still invalid → GameAction
```

Provider errors, repairs, retries and fallbacks are part of the runtime and can change behaviour; they are counted
in the telemetry. Hidden reasoning is never requested.

### Surrogates

- **B1** (`table-v1`): an opportunity-conditioned frequency table with backoff, per research model and personality.
- **B2-v1** (`logistic-v1`): multinomial logistic regression on 54 model-visible features, trained on the Phase-A
  REAL decisions. Selected for closed-loop work in Stage 3.
- **B2-v2** (Stage-5A candidate C1): B2 retrained on Phase-A data plus 854 promoted hard examples (REAL-labelled
  decisions from earlier hybrid escalations).
- Each comes with a target policy (`target-v2.1`: which object or door a MOVE goes to) and a coverage label
  (`coverage-v1`: HIGH / MEDIUM / LOW / OOD). Surrogates never call a provider.

### Hybrid: selective escalation

```text
Observation
   │
   ▼
B2 proposal  (+ B1 proposal, for disagreement)
   │
   ▼
Router R2.1: repetition loop  OR  strong B1/B2 disagreement ?
   ├─ no  → B2's action                                    (no provider call)
   └─ yes → LLM controller → provider / model → validated action
```

R2.1 is frozen (`workshop/policies/phase-a-v2/hybrid-router-v2/router.json`, sha256:216f3c9a3c4b3f78). Hybrid-v1 is
B2-v1 + R2.1; Hybrid-v2 is B2-v2 + the same R2.1. R0 is the same controller with a router that never escalates (the
call-free control arm). Strong disagreement turned out to be the more useful of the two triggers (see
[RESULTS.md](../research/RESULTS.md#2-selective-escalation-works)).

## Policy, provider, model, personality

Four independent dimensions per spy:

| Dimension | Values | Meaning |
|---|---|---|
| policy | `scripted`, `b1`, `b2_v1`, `hybrid_v1`, `b2_v2`, `hybrid_v2`, `real_llm` | the decision architecture |
| provider | `bedrock`, `openai`, `anthropic`, `ollama` (+ `fake`) | where inference runs |
| model | any id of that provider | which model answers |
| personality | `cautious`, `opportunist`, `racer`, `saboteur` | behavioural prompt / context; selects the surrogate files |

## The playable runtime (`spy_vs_ai/`)

```text
CLI flags / YAML / setup screen (spy_vs_ai/play.py, ui/setup.py)
      │
      ▼
GameRuntimeConfig (runtime/config.py)          validated once, before anything starts
      │
      ├── policy registry (runtime/policy_registry.py)     frozen artifacts, loaded read-only, with their status
      └── provider registry (runtime/provider_registry.py) adapters loaded only for the providers in use
      │
      ▼
Black spy controller          White spy controller           (independent policy / provider / model / personality)
      │                               │
      └────────── engine (RuntimeGame, a LiveGame subclass: runtime/session.py) ──────────┘
                     │
                     ├── renderer (ui/research.py: play / research view, ui/enhanced.py)
                     └── telemetry + replay (runtime/recorder.py) → runs/<run-id>/
```

The frozen code is wrapped, never edited: `RuntimeGame` overrides only the per-spy construction hooks of the live
game (`workshop/live/session.py`), and the frozen policies are loaded through their provenance-checked loaders.

### Execution semantics: evaluation vs playable

| | evaluation mode | playable mode |
|---|---|---|
| decision timing | the world stops while a model thinks | REAL spies decide while the world moves (`live_async`); a hybrid keeps evaluation timing |
| mission clock | game time (model latency spends no mission time) | wall clock (game time with a hybrid) |
| commit order | seeded (barrier) | first answer first |
| enhanced gameplay | off | on |

Playable results are not scientifically equivalent to the frozen experiments, and the runtime says so on screen.

### Credential isolation

Bedrock reads the AWS credential chain, OpenAI direct `OPENAI_API_KEY`, Anthropic direct `ANTHROPIC_API_KEY`,
Ollama nothing. Only the providers a match uses are read; call-free policies read none. Outputs are scrubbed with
the selected providers' secret values and checked before they are written. See
[PROVIDERS_AND_MODELS.md](../play/PROVIDERS_AND_MODELS.md).

### Replay

The recorder stores the engine's inputs frame by frame (the actions applied, the clock steps, the updates) plus the
decision telemetry, never a secret. An offline replay rebuilds the same game with no agent and re-issues those
inputs, then compares the engine's event-log digest and the final state: REPRODUCED or DIVERGED, with zero provider
calls. The bundle is described in [Replay format](#replay-format) below.

### Enhanced gameplay layer

Door traps, trap variants and room dressing are playable-runtime rules downstream of the frozen policies. The policy
still emits `PLACE_TRAP`; a resolver (`runtime/traps.py`) chooses the target (the object, or a doorway) and the
variant (explosive, electric, gas, snare, alarm). Decorative props (`runtime/dressing.py`) are drawn by the renderer
only, never added to the world state or the observation. No action token or observation field was added, so nothing
needed retraining.

## Replay format

Every finished match writes `runs/<YYYYmmdd-HHMMSS>/replay_matchNN.json` unless `--no-replay` is given. A replay is an
**engine-input log plus the match's decision telemetry**: enough to rebuild the match exactly, offline, with no
agent, no provider and no network. The code is `spy_vs_ai/runtime/recorder.py` (writing) and
`spy_vs_ai/runtime/playback.py` (re-issuing and verifying).

**Why an input log.** The engine is deterministic given its seed and the exact sequence of calls it receives: every
random draw comes from seeded generators. Controllers are not deterministic, because a model's answer depends on
latency and the provider. So the recorder wraps the calls a frame makes into the engine, on the live `GameState`
instance only, and stores them. A replay re-issues exactly those calls into a freshly built game with the same level,
seed and rules.

**Top-level keys** (`schema: spy-vs-ai-replay-v1`):

| Key | Content |
|---|---|
| `schema`, `runtime_schema`, `kind` | format identifiers ("runtime match replay (not a scientific experiment record)") |
| `engine` | game-code fingerprint, runtime version, FINAL-1 manifest hash, Python and platform |
| `level`, `level_file` | the level id, and the level file's path and sha256 |
| `seed` | the engine seed |
| `config` | the runtime configuration (policies, providers, models, personalities, mode, `enhanced_gameplay`, ...); never a credential |
| `live_config` | the live-game settings the match ran on (timing, clocks, commit policy, rules) |
| `spies`, `execution` | each spy's identity, scientific status and compatibility label; the execution semantics |
| `frames`, `frame_count` | the engine calls, frame by frame, run-length encoded (below) |
| `events_at`, `telemetry`, `telemetry_index` | the kept telemetry events and where they occurred, so the replay reveals decisions in step with the frames |
| `decisions` | the decision records the research view shows (observation summary, proposals, router trigger, provider and model, returned and executed action, latency, tokens, repairs, fallbacks) |
| `outcome` | the match result (ranking, escape order) and the summary |
| `verification` | the engine event-log digest, the event count and the final spy states |
| `enhanced_gameplay`, `enhanced_traps` | whether the enhanced rules ran, and the door traps (target, variant, owner, trigger time, victim) |
| `historical` | re-simulated research matches only: the source match and the comparison |
| `curation` | shared examples only: what `tools.curate_example_replay` changed (local paths, profile names) |
| `started_utc`, `finished_utc`, `complete`, `secrets_recorded: false` | bookkeeping |

**Frames.** `frames` is a list of `[repeat, ops]`, where `ops` are the engine calls of one frame, in order:

- `["c", seconds]`: the wall-clock mission timer advanced;
- `["a", actor, action, dt]`: an action applied, encoded as
  `[type, direction, target_id, trap_type, message, accompanying_message, code]` with trailing nulls dropped;
- `["u", dt, budget_hold]`: the world updated;
- `["e", entries]`: entries a controller wrote into the engine's event log itself (the scripted bot logs its targets
  there).

Identical consecutive frames are stored once, with a repeat count: a model thinking for several seconds is one
entry. The kept telemetry events are: `surrogate_decision`, `hybrid_decision`, `llm_decision`, `opponent_decision`,
`provider_call`, `barrier_commit`, `live_async_commit`, `engine_event`, `spy_fight`, `match_result`,
`action_lifecycle`, `enhanced_trap`.

**Verification.** `python -m spy_vs_ai.replay <bundle> --headless --verify` rebuilds the game with no agent: every spy
is a `ReplayAgent` that never decides. It re-issues every frame and compares:

- the engine event-log digest and the event count;
- the final spy states (position, lifecycle, escaped, lives);
- the ranking and escape order, when the recording finished;
- the door traps, when the enhanced rules ran.

The verdict is **REPRODUCED** or **DIVERGED**, with exit code 1 under `--verify` when it diverges. A different level
file or game-code fingerprint is reported too. Replays recorded before the enhanced gameplay rules existed have no
flag and replay with those rules off.

**Offline guarantee.** The replay CLI runs under an offline guard. Credential-like environment variables are
removed; provider SDK imports, provider client construction and the network are refused; and the result reports 0
provider calls, 0 network attempts and 0 credential reads.

**What a replay never contains:** API keys, AWS credentials, authorization headers or any other secret value. Before
the bundle is written, it is scrubbed with the values of the match's own provider credentials and checked for
credential-shaped strings, and the write is refused if anything is found. Hidden model reasoning is never requested,
so it is never recorded.

**Historical research matches.** With the research archive, `python -m spy_vs_ai.replay --historical <match_dir>`
re-simulates a frozen Stage 4C, 5B or 5C match from its recorded telemetry. Each spy is fed its recorded validated
actions, with zero provider calls, and the engine events are compared with the stored ones. `--save PATH` writes the
reproduced bundle; the five research matches in `examples/replays/` (`01` to `05`) were made this way.

## The broader lesson

What an agent does is not the model alone:

```text
model + observation + representation + memory + prompt / personality + action schema + router
      + repair / fallback + scheduling + execution semantics + environment history
```

The LLM is one component of the agent, and the harness around it is part of the agent's behaviour.
